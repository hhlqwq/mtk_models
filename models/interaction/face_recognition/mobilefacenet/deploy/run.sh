#!/usr/bin/env bash
# MobileFaceNet 单脚本流程: 89 编译上传,开发板测试.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 用户配置: 留空的可选项使用仓库内默认路径.
if [[ ! -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    # 原始 PyTorch 权重: 留空则使用 original/mobilefacenet.pt.
    WEIGHTS=""
    # 已对齐人脸的 INT8 校准图片目录.
    CALIBRATION_DIR=""
    # 临时构建目录: 辅助输入、缓存和程序放在仓库外.
    BUILD_WORK_DIR="/tmp/hailongcodex/$(date +%F)/mobilefacenet"
    # 模型输出目录: 留空则使用 models/.
    MODEL_OUTPUT_DIR=""
    # 板端已对齐 LFW 数据集目录.
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

# 上传的配置文件标识板端阶段,在板端直接执行 LFW 十折评测.
if [[ -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    source "${SCRIPT_DIR}/board_paths.conf"
    readonly RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    readonly RESULT_DIR="${BOARD_RESULTS_DIR}/${RUN_ID}"
    readonly RUN_DIR="${RESULT_DIR}/work"
    test -s "${SCRIPT_DIR}/model_int8.dla"
    test -s "${SCRIPT_DIR}/metadata.json"
    test -s "${BOARD_DATASET_DIR}/pairs.csv"
    test -d "${BOARD_DATASET_DIR}/images"
    test ! -e "${RESULT_DIR}"
    echo "[1/2] 在板端执行 LFW 十折全量人脸验证."
    python3 "${SCRIPT_DIR}/full_accuracy_board.py" \
        --dataset-root "${BOARD_DATASET_DIR}" \
        --model "${SCRIPT_DIR}/model_int8.dla" \
        --metadata "${SCRIPT_DIR}/metadata.json" \
        --run-dir "${RUN_DIR}" --run-id "${RUN_ID}"
    echo "[2/2] 使用交叉编译的 C++ 程序测量常驻模型推理耗时."
    inputs=("${RUN_DIR}/inputs/"*.bin)
    test -f "${inputs[0]}"
    "${SCRIPT_DIR}/benchmark_board" --model "${SCRIPT_DIR}/model_int8.dla" \
        --input "${inputs[0]}" --report "${RUN_DIR}/report/benchmark.json" \
        --warmup 10 --repeats 100
    # 所有原始数据仅在本次 work 下生成; 汇总成功后由工具清理.
    python3 "${SCRIPT_DIR}/summarize_board_result.py" \
        --model "mobilefacenet" --work-dir "${RUN_DIR}" \
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

# 路径配置: 用户可用同名环境变量指定权重、校准集、产物和板端目录.
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${ONNX_DATASET_DIR:?请在脚本顶部指定编译主机的全量精度数据集}"
test -d "${ONNX_DATASET_DIR}"
if [[ "${ONNX_DATASET_DIR}" != /* ]]; then
    echo "[ERROR] ONNX_DATASET_DIR 必须是编译主机与 Docker 共用的绝对路径." >&2
    exit 2
fi
readonly WEIGHTS="${WEIGHTS:-${MODEL_ROOT}/original/mobilefacenet.pt}"
readonly CALIBRATION_DIR="${CALIBRATION_DIR:?请指定对齐人脸校准目录}"
readonly MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-${MODEL_ROOT}/models}"
readonly BOARD_DATASET_DIR="${BOARD_DATASET_DIR:?请指定板端 LFW 数据集目录}"
readonly BOARD_DEPLOY_DIR="${BOARD_DEPLOY_DIR:?请指定 BOARD_DEPLOY_DIR}"
readonly BOARD_RESULTS_DIR="${BOARD_RESULTS_DIR:-${BOARD_DEPLOY_DIR}/results}"
readonly BOARD_HOST="${BOARD_HOST:-root@192.168.0.92}"
readonly CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
readonly MTK_SETUP_SCRIPT="${MTK_SETUP_SCRIPT:-/opt/mtk-build/setup_container.sh}"
readonly NCC_ROOT="${NCC_ROOT:-/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host}"
readonly TOOLCHAIN_ROOT="${MTK_G720_CPP_TOOLCHAIN_ROOT:-/data/users/hailong.he/data/MTKG720/cpp_toolchain}"
readonly NEURON_INCLUDE="${MTK_NEURON_INCLUDE:-/data/users/hailong.he/data/MTKG720/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/include}"
readonly CXX="${CROSS_CXX:-aarch64-linux-gnu-g++}"
readonly BINARY_OUTPUT="${BOARD_BINARY_OUTPUT:-${BUILD_WORK_DIR}/benchmark_board}"
readonly SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

mkdir -p "${BUILD_WORK_DIR}/tmp"
export TMPDIR="${BUILD_WORK_DIR}/tmp"

test -s "${WEIGHTS}"
test -d "${CALIBRATION_DIR}"
if [[ ! "${BOARD_DEPLOY_DIR}" =~ ^/[A-Za-z0-9_./-]+$ ]]; then
    echo "[ERROR] BOARD_DEPLOY_DIR 必须是无空格的板端绝对路径." >&2
    exit 2
fi

echo "[1/3] 在 Docker 中导出、量化、编译并生成输入元数据."
docker exec -i -e BUILD_WORK_DIR="${BUILD_WORK_DIR}" -e PYTHONDONTWRITEBYTECODE=1 \
    -e MODEL_ROOT="${MODEL_ROOT}" -e WEIGHTS="${WEIGHTS}" \
    -e CALIBRATION_DIR="${CALIBRATION_DIR}" \
    -e MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR}" \
    -e MTK_SETUP_SCRIPT="${MTK_SETUP_SCRIPT}" -e NCC_ROOT="${NCC_ROOT}" \
    "${CONTAINER}" bash -s <<'DOCKER_BUILD'
set -euo pipefail
mkdir -p "${MODEL_OUTPUT_DIR}" "${BUILD_WORK_DIR}/tmp" "${BUILD_WORK_DIR}/cache"
export TMPDIR="${BUILD_WORK_DIR}/tmp"
export XDG_CACHE_HOME="${BUILD_WORK_DIR}/cache"
export TORCH_HOME="${BUILD_WORK_DIR}/cache/torch"
cd "${BUILD_WORK_DIR}"
bash "${MTK_SETUP_SCRIPT}"
python "${MODEL_ROOT}/deploy/export_onnx.py" \
    --weights "${WEIGHTS}" \
    --output "${MODEL_OUTPUT_DIR}/model_fp32.onnx"
python "${MODEL_ROOT}/deploy/convert_int8.py" \
    --onnx "${MODEL_OUTPUT_DIR}/model_fp32.onnx" \
    --image-dir "${CALIBRATION_DIR}" \
    --output "${MODEL_OUTPUT_DIR}/model_int8.tflite"
export LD_LIBRARY_PATH="${NCC_ROOT}/lib:${LD_LIBRARY_PATH:-}"
"${NCC_ROOT}/bin/ncc-tflite" --arch=mdla5.3 \
    --suppress-output --disallow-bridge \
    "${MODEL_OUTPUT_DIR}/model_int8.tflite" -o "${MODEL_OUTPUT_DIR}/model_int8.dla"
python "${MODEL_ROOT}/deploy/prepare_input.py" \
    --tflite "${MODEL_OUTPUT_DIR}/model_int8.tflite" \
    --image-dir "${CALIBRATION_DIR}" \
    --output-dir "${BUILD_WORK_DIR}/board_input"
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
    --model "mobilefacenet" --onnx "${ONNX_REFERENCE}" \
    --dataset-root "${ONNX_DATASET_DIR}" \
    --output "${BUILD_WORK_DIR}/onnx_accuracy/summary.json"
ONNX_ACCURACY
# 将本次实测精度写入板端配置,不传递主机预测、耗时或内存.
REFERENCE_ACCURACY="$(docker exec "${CONTAINER}" python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["accuracy"])' \
    "${BUILD_WORK_DIR}/onnx_accuracy/summary.json")"
REFERENCE_SOURCE="本次 ONNX 浮点全量实测,使用与板端相同的评测协议"
test -s "${MODEL_OUTPUT_DIR}/model_int8.dla"
# Docker 与主机的临时目录独立,仅取回上传所需的元数据.
docker cp "${CONTAINER}:${BUILD_WORK_DIR}/board_input/metadata.json" "${BUILD_WORK_DIR}/metadata.json"
test -s "${BUILD_WORK_DIR}/metadata.json"

echo "[2/3] 在编译主机交叉编译 C++ 板端性能程序."
readonly TARGET_LIBS="${TOOLCHAIN_ROOT}/genio720-libs"
command -v "${CXX}" >/dev/null 2>&1
test -f "${NEURON_INCLUDE}/neuron/api/RuntimeAPI.h"
test -f "${TARGET_LIBS}/libneuronusdk_runtime.mtk.so.8"
mkdir -p "$(dirname "${BINARY_OUTPUT}")"
"${CXX}" -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wpedantic \
    -I"${NEURON_INCLUDE}" "${SCRIPT_DIR}/benchmark_board.cpp" \
    "${TARGET_LIBS}/libneuronusdk_runtime.mtk.so.8" \
    -Wl,--allow-shlib-undefined -pthread -ldl -o "${BINARY_OUTPUT}"
file "${BINARY_OUTPUT}"

echo "[3/3] 上传模型、程序、评测代码和路径配置到板端."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" "mkdir -p '${BOARD_DEPLOY_DIR}'"
scp "${SSH_OPTIONS[@]}" "${MODEL_OUTPUT_DIR}/model_int8.dla" \
    "${BUILD_WORK_DIR}/metadata.json" \
    "${SCRIPT_DIR}/full_accuracy_board.py" \
    "${SCRIPT_DIR}/face_utils.py" "${SCRIPT_DIR}/run.sh" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/"
scp "${SSH_OPTIONS[@]}" "${BINARY_OUTPUT}" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/benchmark_board"
scp "${SSH_OPTIONS[@]}" \
    "${MODEL_ROOT}/../../../../tools/summarize_board_result.py" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/summarize_board_result.py"
printf 'BOARD_DATASET_DIR=%q\nBOARD_RESULTS_DIR=%q\nREFERENCE_ACCURACY=%q\nREFERENCE_SOURCE=%q\nDEPLOYMENT_PRECISION=%q\nWEIGHT_DTYPE=%q\nACTIVATION_DTYPE=%q\nPRECISION_SOURCE=%q\n' \
    "${BOARD_DATASET_DIR}" "${BOARD_RESULTS_DIR}" "${REFERENCE_ACCURACY}" "${REFERENCE_SOURCE}" "${DEPLOYMENT_PRECISION}" "${WEIGHT_DTYPE}" "${ACTIVATION_DTYPE}" "${PRECISION_SOURCE}" |
    ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
        "cat > '${BOARD_DEPLOY_DIR}/board_paths.conf'"
echo "[OK] 在板端运行: bash '${BOARD_DEPLOY_DIR}/run.sh'"
