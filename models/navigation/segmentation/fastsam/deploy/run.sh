#!/usr/bin/env bash
# FastSAM-s 单脚本流程: 89 编译上传,开发板测试类别无关分割.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 用户配置: 留空的可选项使用仓库内默认路径.
if [[ ! -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    # 官方 FastSAM-s 权重: 留空则使用 original/FastSAM-s.pt.
    FASTSAM_WEIGHTS=""
    # 导出等价性检查使用的一张图片.
    FASTSAM_IMAGE=""
    # INT8 校准图片目录.
    FASTSAM_CALIBRATION_DIR=""
    # 临时构建目录: 辅助输入、缓存和程序放在仓库外.
    BUILD_WORK_DIR="/tmp/hailongcodex/$(date +%F)/fastsam"
    # 模型输出目录: 留空则使用 models/.
    MODEL_OUTPUT_DIR=""
    # 板端 COCO val2017 数据集目录.
    BOARD_DATASET_DIR=""
    # 板端部署目录.
    BOARD_DEPLOY_DIR=""
    # 板端结果目录: 留空则位于部署目录下.
    BOARD_RESULTS_DIR=""
    # ONNX 精度数据: 编译主机与 Docker 都可访问的全量数据集根目录,必须填写.
    ONNX_DATASET_DIR=""
    # 部署精度方案: 按实际编译信息填写,例如 w8a8、w8a16、fp16、fp32、mixed; 未确认用 unknown.
    DEPLOYMENT_PRECISION="unknown"
    # 实际权重类型: 例如 int8、fp16、mixed; 不根据输入输出或文件名推断.
    WEIGHT_DTYPE="unknown"
    # 实际激活类型: 例如 int16、fp16、mixed; 未确认用 unknown.
    ACTIVATION_DTYPE="unknown"
    # 精度依据: 编译配置/报告或混合层说明; 填写精度时同步填写,这些配置不改变编译策略.
    PRECISION_SOURCE=""
    # 板端 SSH 用户和地址.
    BOARD_HOST="root@192.168.0.92"
fi

# 板端阶段: 保留逐图检查点,同一 EVAL_RUN_ID 可显式续跑.
if [[ -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    source "${SCRIPT_DIR}/board_paths.conf"
    readonly RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    readonly RESULT_DIR="${BOARD_RESULTS_DIR}/${RUN_ID}"
    readonly RUN_DIR="${RESULT_DIR}/work"
    readonly RESUME="${EVAL_RESUME:-0}"
    readonly ANNOTATIONS="${BOARD_DATASET_DIR}/annotations/instances_val2017.json"
    test -s "${SCRIPT_DIR}/model_int8.dla"
    test -s "${SCRIPT_DIR}/runtime_config.csv"
    test -s "${ANNOTATIONS}"
    test "$(find "${BOARD_DATASET_DIR}/images" -maxdepth 1 -type f -name '*.jpg' | wc -l)" -eq 5000
    python3 -c 'import cv2, numpy, pycocotools'
    if [[ "${RESUME}" == "1" ]]; then
        test -d "${RUN_DIR}"
    else
        test ! -e "${RESULT_DIR}"
    fi
    read -r CONFIDENCE NMS_IOU MAX_DETECTIONS < <(
        python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["confidence"],p["nms_iou"],p["max_detections"])' \
            "${SCRIPT_DIR}/accuracy_protocol.json")
    mkdir -p "${RUN_DIR}/raw" "${RUN_DIR}/predictions_by_image" \
        "${RUN_DIR}/report"

    echo "[1/3] 在板端逐图执行 C++ NPU 推理并保存检查点."
    count=0
    while IFS= read -r -d '' image; do
        image_name="$(basename "${image}" .jpg)"
        checkpoint="${RUN_DIR}/predictions_by_image/${image_name}.json"
        if [[ -e "${checkpoint}" ]]; then
            if [[ "${RESUME}" != "1" ]]; then
                echo "[ERROR] 非续跑模式发现已有检查点: ${checkpoint}" >&2
                exit 2
            fi
        else
            image_dir="${RUN_DIR}/raw/${image_name}"
            test ! -e "${image_dir}"
            "${SCRIPT_DIR}/fastsam_board" \
                --model "${SCRIPT_DIR}/model_int8.dla" \
                --config "${SCRIPT_DIR}/runtime_config.csv" \
                --image "${image}" --output-dir "${image_dir}" \
                --confidence "${CONFIDENCE}" --iou "${NMS_IOU}" \
                --max-det "${MAX_DETECTIONS}"
            python3 "${SCRIPT_DIR}/full_accuracy_board.py" \
                --mode encode-one --image "${image}" \
                --images "${BOARD_DATASET_DIR}/images" \
                --annotations "${ANNOTATIONS}" \
                --work-dir "${RUN_DIR}" --run-id "${RUN_ID}"
            # 原始掩码已写入检查点,仅删除本次逐图中间文件.
            test "${image_dir}" = "${RUN_DIR}/raw/${image_name}"
            rm -r -- "${image_dir}"
        fi
        count=$((count + 1))
        if (( count % 50 == 0 || count == 5000 )); then
            echo "[PROGRESS] FastSAM ${count}/5000 张."
        fi
    done < <(find "${BOARD_DATASET_DIR}/images" -maxdepth 1 \
        -type f -name '*.jpg' -print0 | sort -z)
    test "${count}" -eq 5000

    echo "[2/3] 在板端计算类别无关 COCO segm AP."
    python3 "${SCRIPT_DIR}/full_accuracy_board.py" \
        --mode evaluate --images "${BOARD_DATASET_DIR}/images" \
        --annotations "${ANNOTATIONS}" --work-dir "${RUN_DIR}" \
        --run-id "${RUN_ID}" 2>&1 | tee "${RUN_DIR}/board_eval.log"

    # 所有原始数据仅在本次 work 下生成; 汇总成功后由工具清理.
    python3 "${SCRIPT_DIR}/summarize_board_result.py" \
        --model "fastsam" --work-dir "${RUN_DIR}" \
        --output "${RESULT_DIR}/summary.json" --run-id "${RUN_ID}" \
        --reference "${REFERENCE_ACCURACY:-}" \
        --reference-source "${REFERENCE_SOURCE:-用户提供的同协议参考基准}" \
        --precision "${DEPLOYMENT_PRECISION:-unknown}" \
        --weight-dtype "${WEIGHT_DTYPE:-unknown}" \
        --activation-dtype "${ACTIVATION_DTYPE:-unknown}" \
        --precision-source "${PRECISION_SOURCE:-}"
    exit 0
fi

if (( $# != 0 )); then
    echo "[ERROR] 在编译主机直接运行 bash deploy/run.sh,无需参数." >&2
    exit 2
fi

# 路径配置: 模型和校准集由用户提供,产物及板端目录均可覆盖.
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${ONNX_DATASET_DIR:?请在脚本顶部指定编译主机的全量精度数据集}"
test -d "${ONNX_DATASET_DIR}"
if [[ "${ONNX_DATASET_DIR}" != /* ]]; then
    echo "[ERROR] ONNX_DATASET_DIR 必须是编译主机与 Docker 共用的绝对路径." >&2
    exit 2
fi
readonly WEIGHTS="${FASTSAM_WEIGHTS:-${MODEL_ROOT}/original/FastSAM-s.pt}"
readonly SAMPLE_IMAGE="${FASTSAM_IMAGE:?请指定样例图片绝对路径}"
readonly CALIBRATION_DIR="${FASTSAM_CALIBRATION_DIR:?请指定校准图片目录}"
readonly MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-${MODEL_ROOT}/models}"
readonly BOARD_DATASET_DIR="${BOARD_DATASET_DIR:?请指定板端 COCO 数据集目录}"
readonly BOARD_DEPLOY_DIR="${BOARD_DEPLOY_DIR:?请指定 BOARD_DEPLOY_DIR}"
readonly BOARD_RESULTS_DIR="${BOARD_RESULTS_DIR:-${BOARD_DEPLOY_DIR}/results}"
readonly BOARD_HOST="${BOARD_HOST:-root@192.168.0.92}"
readonly CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
readonly NCC_ROOT="${NCC_ROOT:-/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host}"
readonly TOOLCHAIN_ROOT="${MTK_G720_CPP_TOOLCHAIN_ROOT:-/data/users/hailong.he/data/MTKG720/cpp_toolchain}"
readonly NEURON_INCLUDE="${MTK_NEURON_INCLUDE:-/data/users/hailong.he/data/MTKG720/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/include}"
readonly CXX="${CROSS_CXX:-aarch64-linux-gnu-g++}"
readonly SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

mkdir -p "${BUILD_WORK_DIR}/tmp"
export TMPDIR="${BUILD_WORK_DIR}/tmp"

test -s "${WEIGHTS}"
test -s "${SAMPLE_IMAGE}"
test -d "${CALIBRATION_DIR}"
if [[ ! "${BOARD_DEPLOY_DIR}" =~ ^/[A-Za-z0-9_./-]+$ ]]; then
    echo "[ERROR] BOARD_DEPLOY_DIR 必须是无空格的板端绝对路径." >&2
    exit 2
fi

echo "[1/3] 在 Docker 中导出、量化并编译 DLA."
docker exec -i -e BUILD_WORK_DIR="${BUILD_WORK_DIR}" -e PYTHONDONTWRITEBYTECODE=1 -e MODEL_ROOT="${MODEL_ROOT}" -e WEIGHTS="${WEIGHTS}" \
    -e SAMPLE_IMAGE="${SAMPLE_IMAGE}" \
    -e CALIBRATION_DIR="${CALIBRATION_DIR}" \
    -e MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR}" -e NCC_ROOT="${NCC_ROOT}" \
    "${CONTAINER}" bash -s <<'DOCKER_BUILD'
set -euo pipefail
mkdir -p "${MODEL_OUTPUT_DIR}" "${BUILD_WORK_DIR}/tmp" "${BUILD_WORK_DIR}/cache"
export TMPDIR="${BUILD_WORK_DIR}/tmp"
export XDG_CACHE_HOME="${BUILD_WORK_DIR}/cache"
export TORCH_HOME="${BUILD_WORK_DIR}/cache/torch"
cd "${BUILD_WORK_DIR}"
python "${MODEL_ROOT}/deploy/export_model.py" --weights "${WEIGHTS}" \
    --image "${SAMPLE_IMAGE}" \
    --output-dir "${BUILD_WORK_DIR}/export"
cp "${BUILD_WORK_DIR}/export/model_fp32.onnx" "${MODEL_OUTPUT_DIR}/model_fp32.onnx"
python "${MODEL_ROOT}/deploy/convert_int8.py" \
    --onnx "${MODEL_OUTPUT_DIR}/model_fp32.onnx" \
    --calibration-dir "${CALIBRATION_DIR}" --samples 100 \
    --output "${MODEL_OUTPUT_DIR}/model_int8.tflite"
export LD_LIBRARY_PATH="${NCC_ROOT}/lib:${LD_LIBRARY_PATH:-}"
"${NCC_ROOT}/bin/ncc-tflite" --arch=mdla5.3 \
    --suppress-output --disallow-bridge \
    "${MODEL_OUTPUT_DIR}/model_int8.tflite" -o "${MODEL_OUTPUT_DIR}/model_int8.dla"
DOCKER_BUILD

echo "[ONNX] 在编译主机 Docker 中评测全量浮点精度."
docker exec -i -e PYTHONDONTWRITEBYTECODE=1 \
    -e MODEL_ROOT="${MODEL_ROOT}" -e ONNX_DATASET_DIR="${ONNX_DATASET_DIR}" \
    -e ONNX_REFERENCE="${MODEL_OUTPUT_DIR}/model_fp32.onnx" -e BUILD_WORK_DIR="${BUILD_WORK_DIR}" \
    "${CONTAINER}" bash -s <<'ONNX_ACCURACY'
set -euo pipefail
test -d "${ONNX_DATASET_DIR}"
export TMPDIR="${BUILD_WORK_DIR}/tmp"
export XDG_CACHE_HOME="${BUILD_WORK_DIR}/cache"
export TORCH_HOME="${BUILD_WORK_DIR}/cache/torch"
REPO_ROOT="$(cd "${MODEL_ROOT}/../../../.." && pwd)"
python "${REPO_ROOT}/tools/accuracy/evaluate_onnx.py" \
    --model "fastsam" --onnx "${ONNX_REFERENCE}" \
    --dataset-root "${ONNX_DATASET_DIR}" \
    --output "${BUILD_WORK_DIR}/onnx_accuracy/summary.json"
ONNX_ACCURACY
# 将本次实测精度写入板端配置,不传递主机预测、耗时或内存.
REFERENCE_ACCURACY="$(docker exec "${CONTAINER}" python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["accuracy"])' \
    "${BUILD_WORK_DIR}/onnx_accuracy/summary.json")"
REFERENCE_SOURCE="本次 ONNX 浮点全量实测,使用与板端相同的评测协议"
test -s "${MODEL_OUTPUT_DIR}/runtime_config.csv"

echo "[2/3] 在编译主机交叉编译板端 C++ 测试程序."
readonly OPENCV_SOURCE="${TOOLCHAIN_ROOT}/opencv-4.9.0/opencv-4.9.0"
readonly OPENCV_BUILD="${TOOLCHAIN_ROOT}/opencv-4.9.0/build-aarch64-headers"
readonly TARGET_LIBS="${TOOLCHAIN_ROOT}/genio720-libs"
command -v "${CXX}" >/dev/null 2>&1
"${CXX}" -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic \
    -I"${OPENCV_BUILD}" \
    -I"${OPENCV_SOURCE}/modules/core/include" \
    -I"${OPENCV_SOURCE}/modules/imgproc/include" \
    -I"${OPENCV_SOURCE}/modules/imgcodecs/include" \
    -I"${NEURON_INCLUDE}" \
    "${SCRIPT_DIR}/inference_demo/fastsam_board.cpp" \
    "${TARGET_LIBS}/libneuronusdk_runtime.mtk.so.8" \
    "${TARGET_LIBS}/libopencv_imgcodecs.so.409" \
    "${TARGET_LIBS}/libopencv_imgproc.so.409" \
    "${TARGET_LIBS}/libopencv_core.so.409" \
    -Wl,--allow-shlib-undefined -pthread -ldl \
    -o "${BUILD_WORK_DIR}/fastsam_board"
file "${BUILD_WORK_DIR}/fastsam_board"

echo "[3/3] 上传模型、程序、评测代码和路径配置到板端."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" "mkdir -p '${BOARD_DEPLOY_DIR}'"
scp "${SSH_OPTIONS[@]}" "${MODEL_OUTPUT_DIR}/model_int8.dla" \
    "${MODEL_OUTPUT_DIR}/runtime_config.csv" \
    "${BUILD_WORK_DIR}/fastsam_board" \
    "${SCRIPT_DIR}/full_accuracy_board.py" \
    "${SCRIPT_DIR}/accuracy_protocol.json" "${SCRIPT_DIR}/run.sh" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "chmod 755 '${BOARD_DEPLOY_DIR}/fastsam_board'"
scp "${SSH_OPTIONS[@]}" \
    "${MODEL_ROOT}/../../../../tools/summarize_board_result.py" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/summarize_board_result.py"
printf 'BOARD_DATASET_DIR=%q\nBOARD_RESULTS_DIR=%q\nREFERENCE_ACCURACY=%q\nREFERENCE_SOURCE=%q\nDEPLOYMENT_PRECISION=%q\nWEIGHT_DTYPE=%q\nACTIVATION_DTYPE=%q\nPRECISION_SOURCE=%q\n' \
    "${BOARD_DATASET_DIR}" "${BOARD_RESULTS_DIR}" "${REFERENCE_ACCURACY}" "${REFERENCE_SOURCE}" "${DEPLOYMENT_PRECISION}" "${WEIGHT_DTYPE}" "${ACTIVATION_DTYPE}" "${PRECISION_SOURCE}" |
    ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
        "cat > '${BOARD_DEPLOY_DIR}/board_paths.conf'"
echo "[OK] 在板端运行: bash '${BOARD_DEPLOY_DIR}/run.sh'"
