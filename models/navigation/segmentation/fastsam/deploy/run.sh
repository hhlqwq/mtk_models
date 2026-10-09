#!/usr/bin/env bash
# 单脚本两步流程: 在编译主机编译并上传,在开发板执行测试.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 编译主机配置区: 只修改等号右侧的路径或名称.板端使用上传的 board_paths.conf.
if [[ ! -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    # 模型目录: 根据本脚本的位置自动确定.
    MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

    # 1. 模型与校准数据.
    # 官方 FastSAM-s 权重.
    FASTSAM_WEIGHTS="${MODEL_ROOT}/models/FastSAM-s.pt"
    # INT8 校准图片目录.
    FASTSAM_CALIBRATION_DIR="/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017/images"
    # 导出等价性检查使用的一张图片.
    FASTSAM_IMAGE="/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017/images/000000000139.jpg"

    # 2. 产物与临时目录.
    # 模型输出目录: 转换与编译产物保存在这里.
    MODEL_OUTPUT_DIR="${MODEL_ROOT}/models"
    # 临时构建目录: 缓存和中间文件保存在仓库外.
    BUILD_WORK_DIR="/tmp/hailongcodex/$(date +%F)/fastsam"

    # 3. 板端地址与数据.
    # 板端地址: SSH 用户和地址.
    BOARD_HOST="root@192.168.0.92"
    # 板端部署目录: 上传模型、程序和本脚本的目录,必须填写.
    BOARD_DEPLOY_DIR="/root/hailong.he/open_models/fastsam"
    # 板端结果目录: 保存本次测试汇总.
    BOARD_RESULTS_DIR="${BOARD_DEPLOY_DIR}/results"
    # 板端 COCO val2017 数据集目录.
    BOARD_DATASET_DIR="/root/hailong.he/datasets/coco/val2017"

    # 4. ONNX 精度数据.
    # 浮点精度数据: 编译主机与 Docker 可访问的数据集根目录,必须填写.
    ONNX_DATASET_DIR="/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017"

    # 5. 编译环境: 通常无需修改.
    # Docker 容器: 编译主机上的 Genio 720 编译环境.
    MTK_G720_CONTAINER="hhl_g720_8011"
    # C++ 工具链: 编译主机上的 AArch64 编译工具和 OpenCV 库目录.
    MTK_G720_CPP_TOOLCHAIN_ROOT="/data/users/hailong.he/data/MTKG720/cpp_toolchain"
    # 模型编译器: Docker 内 Neuron SDK host 目录,包含 bin/ 和 lib/.
    NCC_ROOT="/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host"
    # Runtime 头文件: 编译主机上的 Neuron Runtime include 目录.
    MTK_NEURON_INCLUDE="/data/users/hailong.he/data/MTKG720/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/include"
    # C++ 编译器: 编译主机上的 AArch64 交叉编译命令.
    CROSS_CXX="aarch64-linux-gnu-g++"
    # SSH 选项: 首次连接接受主机密钥,之后验证保存的密钥.
    SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)
fi

# 板端阶段: 保留逐图检查点,同一 EVAL_RUN_ID 可显式续跑.
if [[ -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    source "${SCRIPT_DIR}/board_paths.conf"
    readonly RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    readonly RESULT_DIR="${BOARD_RESULTS_DIR}/${RUN_ID}"
    readonly RUN_DIR="${RESULT_DIR}/work"
    # 在全量推理前检查少量示例的工具、清单和图片依赖.
    test -s "${SCRIPT_DIR}/examples/input/samples.json"
    test -s "${SCRIPT_DIR}/board/render_examples.py"
    python3 -c 'import cv2, numpy'
    readonly RESUME="${EVAL_RESUME:-0}"
    readonly ANNOTATIONS="${BOARD_DATASET_DIR}/annotations/instances_val2017.json"
    test -s "${SCRIPT_DIR}/models/model_int8.dla"
    test -s "${SCRIPT_DIR}/models/runtime_config.csv"
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
            "${SCRIPT_DIR}/board/accuracy_protocol.json")
    mkdir -p "${RUN_DIR}/raw" "${RUN_DIR}/predictions_by_image" \
        "${RUN_DIR}/report"

    echo "[开发板 1/3] 在板端逐图执行 C++ NPU 推理并保存检查点."
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
            "${SCRIPT_DIR}/board/fastsam_board" \
                --model "${SCRIPT_DIR}/models/model_int8.dla" \
                --config "${SCRIPT_DIR}/models/runtime_config.csv" \
                --image "${image}" --output-dir "${image_dir}" \
                --confidence "${CONFIDENCE}" --iou "${NMS_IOU}" \
                --max-det "${MAX_DETECTIONS}"
            python3 "${SCRIPT_DIR}/board/full_accuracy_board.py" \
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

    echo "[开发板 2/3] 在板端计算类别无关 COCO segm AP."
    python3 "${SCRIPT_DIR}/board/full_accuracy_board.py" \
        --mode evaluate --images "${BOARD_DATASET_DIR}/images" \
        --annotations "${ANNOTATIONS}" --work-dir "${RUN_DIR}" \
        --run-id "${RUN_ID}" 2>&1 | tee "${RUN_DIR}/board_eval.log"

    echo "[示例] 保存少量板端效果示例."
    python3 "${SCRIPT_DIR}/board/render_examples.py" \
        --model "fastsam" --input-dir "${SCRIPT_DIR}/examples/input" \
        --output-dir "${RESULT_DIR}/examples/output" --work-dir "${RUN_DIR}" \
        --dataset-root "${BOARD_DATASET_DIR}" --models-dir "${SCRIPT_DIR}/models"

    # 所有原始数据仅在本次 work 下生成; 汇总成功后由工具清理.
    python3 "${SCRIPT_DIR}/board/summarize_board_result.py" \
        --model "fastsam" --work-dir "${RUN_DIR}" \
        --output "${RESULT_DIR}/summary.json" --run-id "${RUN_ID}" \
        --reference "${REFERENCE_ACCURACY:-}" \
        --reference-source "${REFERENCE_SOURCE:-用户提供的同协议参考基准}"
    exit 0
fi

if (( $# != 0 )); then
    echo "[ERROR] 在编译主机直接运行 bash deploy/run.sh,无需参数." >&2
    exit 2
fi

# 编译主机阶段: 检查配置,转换模型,交叉编译并上传.
: "${ONNX_DATASET_DIR:?请在脚本顶部指定编译主机的全量精度数据集}"
test -d "${ONNX_DATASET_DIR}"
if [[ "${ONNX_DATASET_DIR}" != /* ]]; then
    echo "[ERROR] ONNX_DATASET_DIR 必须是编译主机与 Docker 共用的绝对路径." >&2
    exit 2
fi
readonly WEIGHTS="${FASTSAM_WEIGHTS}"
: "${FASTSAM_IMAGE:?请指定样例图片绝对路径}"
readonly SAMPLE_IMAGE="${FASTSAM_IMAGE}"
: "${FASTSAM_CALIBRATION_DIR:?请指定校准图片目录}"
readonly CALIBRATION_DIR="${FASTSAM_CALIBRATION_DIR}"
: "${BOARD_DATASET_DIR:?请指定板端 COCO 数据集目录}"
: "${BOARD_DEPLOY_DIR:?请指定 BOARD_DEPLOY_DIR}"
readonly CONTAINER="${MTK_G720_CONTAINER}"
readonly TOOLCHAIN_ROOT="${MTK_G720_CPP_TOOLCHAIN_ROOT}"
readonly NEURON_INCLUDE="${MTK_NEURON_INCLUDE}"
readonly CXX="${CROSS_CXX}"

mkdir -p "${BUILD_WORK_DIR}/tmp"
export TMPDIR="${BUILD_WORK_DIR}/tmp"

test -s "${WEIGHTS}"
test -s "${SAMPLE_IMAGE}"
test -d "${CALIBRATION_DIR}"
if [[ ! "${BOARD_DEPLOY_DIR}" =~ ^/[A-Za-z0-9_./-]+$ ]]; then
    echo "[ERROR] BOARD_DEPLOY_DIR 必须是无空格的板端绝对路径." >&2
    exit 2
fi

# 从配置的数据集生成示例清单,仅写入仓库外的构建目录.
mkdir -p "${BUILD_WORK_DIR}/examples"
docker exec -e PYTHONDONTWRITEBYTECODE=1 "${CONTAINER}" \
    python "${MODEL_ROOT}/../../../../tools/prepare_examples.py" \
    --model "fastsam" --dataset-root "${ONNX_DATASET_DIR}" \
    --output "${BUILD_WORK_DIR}/examples/samples.json"
docker cp "${CONTAINER}:${BUILD_WORK_DIR}/examples/samples.json" \
    "${BUILD_WORK_DIR}/examples/samples.json"

echo "[编译主机 1] 在 Docker 中导出、量化并编译 DLA."
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
python "${MODEL_ROOT}/deploy/host/export_model.py" --weights "${WEIGHTS}" \
    --image "${SAMPLE_IMAGE}" \
    --output-dir "${BUILD_WORK_DIR}/export"
cp "${BUILD_WORK_DIR}/export/model_fp32.onnx" "${MODEL_OUTPUT_DIR}/model_fp32.onnx"
python "${MODEL_ROOT}/deploy/host/convert_int8.py" \
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

echo "[编译主机 2] 在编译主机交叉编译板端 C++ 测试程序."
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
    "${SCRIPT_DIR}/board/fastsam_board.cpp" \
    "${TARGET_LIBS}/libneuronusdk_runtime.mtk.so.8" \
    "${TARGET_LIBS}/libopencv_imgcodecs.so.409" \
    "${TARGET_LIBS}/libopencv_imgproc.so.409" \
    "${TARGET_LIBS}/libopencv_core.so.409" \
    -Wl,--allow-shlib-undefined -pthread -ldl \
    -o "${BUILD_WORK_DIR}/fastsam_board"
file "${BUILD_WORK_DIR}/fastsam_board"

echo "[编译主机 上传] 上传模型、程序、评测代码和路径配置到板端."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "mkdir -p '${BOARD_DEPLOY_DIR}/models' '${BOARD_DEPLOY_DIR}/board'"
scp "${SSH_OPTIONS[@]}" \
    "${MODEL_OUTPUT_DIR}/model_int8.dla" \
    "${MODEL_OUTPUT_DIR}/runtime_config.csv" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/models/"
scp "${SSH_OPTIONS[@]}" \
    "${SCRIPT_DIR}/board/full_accuracy_board.py" \
    "${SCRIPT_DIR}/board/accuracy_protocol.json" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/board/"
scp "${SSH_OPTIONS[@]}" "${MODEL_ROOT}/../../../../tools/summarize_board_result.py" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/board/"
scp "${SSH_OPTIONS[@]}" \
    "${BUILD_WORK_DIR}/fastsam_board" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/board/fastsam_board"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" "chmod 755 '${BOARD_DEPLOY_DIR}/board/fastsam_board'"
scp "${SSH_OPTIONS[@]}" "${SCRIPT_DIR}/run.sh" "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/"
# 上传统一可视化工具和少量示例输入,不增加 Shell 入口.
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "mkdir -p '${BOARD_DEPLOY_DIR}/examples/input'"
scp "${SSH_OPTIONS[@]}" "${MODEL_ROOT}/../../../../tools/render_examples.py" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/board/render_examples.py"
scp "${SSH_OPTIONS[@]}" "${BUILD_WORK_DIR}/examples/samples.json" \
    "${MODEL_ROOT}/examples/input"/sample_* \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/examples/input/"
printf 'BOARD_DATASET_DIR=%q\nBOARD_RESULTS_DIR=%q\nREFERENCE_ACCURACY=%q\nREFERENCE_SOURCE=%q\n' \
    "${BOARD_DATASET_DIR}" "${BOARD_RESULTS_DIR}" "${REFERENCE_ACCURACY}" "${REFERENCE_SOURCE}" |
    ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
        "cat > '${BOARD_DEPLOY_DIR}/board_paths.conf'"
echo "[NEXT] 在板端运行: bash '${BOARD_DEPLOY_DIR}/run.sh'"
