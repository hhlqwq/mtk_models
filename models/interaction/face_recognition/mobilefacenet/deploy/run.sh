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
    # 原始 PyTorch 权重.
    WEIGHTS="${MODEL_ROOT}/original/mobilefacenet.pt"
    # 已对齐人脸的 INT8 校准图片目录.
    CALIBRATION_DIR=""

    # 2. 产物与临时目录.
    # 模型输出目录: 转换与编译产物保存在这里.
    MODEL_OUTPUT_DIR="${MODEL_ROOT}/models"
    # 临时构建目录: 缓存和中间文件保存在仓库外.
    BUILD_WORK_DIR="/tmp/hailongcodex/$(date +%F)/mobilefacenet"
    # C++ 输出: 编译主机交叉编译生成的临时程序完整路径.
    BOARD_BINARY_OUTPUT="${BUILD_WORK_DIR}/benchmark_board"

    # 3. 板端地址与数据.
    # 板端地址: SSH 用户和地址.
    BOARD_HOST="root@192.168.0.92"
    # 板端部署目录: 上传模型、程序和本脚本的目录,必须填写.
    BOARD_DEPLOY_DIR=""
    # 板端结果目录: 保存本次测试汇总.
    BOARD_RESULTS_DIR="${BOARD_DEPLOY_DIR}/results"
    # 板端已对齐 LFW 数据集目录.
    BOARD_DATASET_DIR=""

    # 4. ONNX 精度数据.
    # 浮点精度数据: 编译主机与 Docker 可访问的数据集根目录,必须填写.
    ONNX_DATASET_DIR=""

    # 5. 编译环境: 通常无需修改.
    # Docker 容器: 编译主机上的 Genio 720 编译环境.
    MTK_G720_CONTAINER="hhl_g720_8011"
    # C++ 工具链: 编译主机上的 AArch64 编译工具和 OpenCV 库目录.
    MTK_G720_CPP_TOOLCHAIN_ROOT="/data/users/hailong.he/data/MTKG720/cpp_toolchain"
    # 环境脚本: Docker 内 MTK SDK 初始化脚本.
    MTK_SETUP_SCRIPT="/opt/mtk-build/setup_container.sh"
    # 模型编译器: Docker 内 Neuron SDK host 目录,包含 bin/ 和 lib/.
    NCC_ROOT="/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host"
    # Runtime 头文件: 编译主机上的 Neuron Runtime include 目录.
    MTK_NEURON_INCLUDE="/data/users/hailong.he/data/MTKG720/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/include"
    # C++ 编译器: 编译主机上的 AArch64 交叉编译命令.
    CROSS_CXX="aarch64-linux-gnu-g++"
    # SSH 选项: 首次连接接受主机密钥,之后验证保存的密钥.
    SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)
fi

# 上传的配置文件标识板端阶段,在板端直接执行 LFW 十折评测.
if [[ -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    source "${SCRIPT_DIR}/board_paths.conf"
    readonly RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    readonly RESULT_DIR="${BOARD_RESULTS_DIR}/${RUN_ID}"
    readonly RUN_DIR="${RESULT_DIR}/work"
    test -s "${SCRIPT_DIR}/models/model_int8.dla"
    test -s "${SCRIPT_DIR}/models/metadata.json"
    test -s "${BOARD_DATASET_DIR}/pairs.csv"
    test -d "${BOARD_DATASET_DIR}/images"
    test ! -e "${RESULT_DIR}"
    echo "[开发板 1/2] 在板端执行 LFW 十折全量人脸验证."
    python3 "${SCRIPT_DIR}/board/full_accuracy_board.py" \
        --dataset-root "${BOARD_DATASET_DIR}" \
        --model "${SCRIPT_DIR}/models/model_int8.dla" \
        --metadata "${SCRIPT_DIR}/models/metadata.json" \
        --run-dir "${RUN_DIR}" --run-id "${RUN_ID}"
    echo "[开发板 2/2] 使用交叉编译的 C++ 程序测量常驻模型推理耗时."
    inputs=("${RUN_DIR}/inputs/"*.bin)
    test -f "${inputs[0]}"
    "${SCRIPT_DIR}/board/benchmark_board" --model "${SCRIPT_DIR}/models/model_int8.dla" \
        --input "${inputs[0]}" --report "${RUN_DIR}/report/benchmark.json" \
        --warmup 10 --repeats 100
    # 所有原始数据仅在本次 work 下生成; 汇总成功后由工具清理.
    python3 "${SCRIPT_DIR}/board/summarize_board_result.py" \
        --model "mobilefacenet" --work-dir "${RUN_DIR}" \
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
: "${CALIBRATION_DIR:?请指定对齐人脸校准目录}"
: "${BOARD_DATASET_DIR:?请指定板端 LFW 数据集目录}"
: "${BOARD_DEPLOY_DIR:?请指定 BOARD_DEPLOY_DIR}"
readonly CONTAINER="${MTK_G720_CONTAINER}"
readonly TOOLCHAIN_ROOT="${MTK_G720_CPP_TOOLCHAIN_ROOT}"
readonly NEURON_INCLUDE="${MTK_NEURON_INCLUDE}"
readonly CXX="${CROSS_CXX}"
readonly BINARY_OUTPUT="${BOARD_BINARY_OUTPUT}"

mkdir -p "${BUILD_WORK_DIR}/tmp"
export TMPDIR="${BUILD_WORK_DIR}/tmp"

test -s "${WEIGHTS}"
test -d "${CALIBRATION_DIR}"
if [[ ! "${BOARD_DEPLOY_DIR}" =~ ^/[A-Za-z0-9_./-]+$ ]]; then
    echo "[ERROR] BOARD_DEPLOY_DIR 必须是无空格的板端绝对路径." >&2
    exit 2
fi

echo "[编译主机 1] 在 Docker 中导出、量化、编译并生成输入元数据."
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
python "${MODEL_ROOT}/deploy/host/export_onnx.py" \
    --weights "${WEIGHTS}" \
    --output "${MODEL_OUTPUT_DIR}/model_fp32.onnx"
python "${MODEL_ROOT}/deploy/host/convert_int8.py" \
    --onnx "${MODEL_OUTPUT_DIR}/model_fp32.onnx" \
    --image-dir "${CALIBRATION_DIR}" \
    --output "${MODEL_OUTPUT_DIR}/model_int8.tflite"
export LD_LIBRARY_PATH="${NCC_ROOT}/lib:${LD_LIBRARY_PATH:-}"
"${NCC_ROOT}/bin/ncc-tflite" --arch=mdla5.3 \
    --suppress-output --disallow-bridge \
    "${MODEL_OUTPUT_DIR}/model_int8.tflite" -o "${MODEL_OUTPUT_DIR}/model_int8.dla"
python "${MODEL_ROOT}/deploy/host/prepare_input.py" \
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

echo "[编译主机 2] 在编译主机交叉编译 C++ 板端性能程序."
readonly TARGET_LIBS="${TOOLCHAIN_ROOT}/genio720-libs"
command -v "${CXX}" >/dev/null 2>&1
test -f "${NEURON_INCLUDE}/neuron/api/RuntimeAPI.h"
test -f "${TARGET_LIBS}/libneuronusdk_runtime.mtk.so.8"
mkdir -p "$(dirname "${BINARY_OUTPUT}")"
"${CXX}" -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wpedantic \
    -I"${NEURON_INCLUDE}" "${SCRIPT_DIR}/board/benchmark_board.cpp" \
    "${TARGET_LIBS}/libneuronusdk_runtime.mtk.so.8" \
    -Wl,--allow-shlib-undefined -pthread -ldl -o "${BINARY_OUTPUT}"
file "${BINARY_OUTPUT}"

echo "[编译主机 上传] 上传模型、程序、评测代码和路径配置到板端."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "mkdir -p '${BOARD_DEPLOY_DIR}/models' '${BOARD_DEPLOY_DIR}/board'"
scp "${SSH_OPTIONS[@]}" \
    "${MODEL_OUTPUT_DIR}/model_int8.dla" \
    "${BUILD_WORK_DIR}/metadata.json" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/models/"
scp "${SSH_OPTIONS[@]}" \
    "${SCRIPT_DIR}/board/full_accuracy_board.py" \
    "${SCRIPT_DIR}/board/face_utils.py" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/board/"
scp "${SSH_OPTIONS[@]}" "${MODEL_ROOT}/../../../../tools/summarize_board_result.py" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/board/"
scp "${SSH_OPTIONS[@]}" "${BINARY_OUTPUT}" "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/board/benchmark_board"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" "chmod 755 '${BOARD_DEPLOY_DIR}/board/benchmark_board'"
scp "${SSH_OPTIONS[@]}" "${SCRIPT_DIR}/run.sh" "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/"
printf 'BOARD_DATASET_DIR=%q\nBOARD_RESULTS_DIR=%q\nREFERENCE_ACCURACY=%q\nREFERENCE_SOURCE=%q\n' \
    "${BOARD_DATASET_DIR}" "${BOARD_RESULTS_DIR}" "${REFERENCE_ACCURACY}" "${REFERENCE_SOURCE}" |
    ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
        "cat > '${BOARD_DEPLOY_DIR}/board_paths.conf'"
echo "[NEXT] 在板端运行: bash '${BOARD_DEPLOY_DIR}/run.sh'"
