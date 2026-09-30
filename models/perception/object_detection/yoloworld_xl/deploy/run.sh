#!/usr/bin/env bash
# YOLO-World XL 单脚本流程: 89 准备兼容 ONNX 并上传,92 板端测试.

set -euo pipefail
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 用户配置: 留空的可选项使用仓库内默认路径.
if [[ ! -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    # 官方 ONNX: 留空则使用 models/model_fp32.onnx.
    SOURCE_ONNX=""
    # 模型输出目录: 留空则使用 models/.
    MODEL_OUTPUT_DIR=""
    # 板端 COCO val2017 数据集目录.
    BOARD_DATASET_DIR=""
    # 板端部署目录.
    BOARD_DEPLOY_DIR=""
    # 板端结果目录: 留空则位于部署目录下.
    BOARD_RESULTS_DIR=""
    # 板端 SSH 用户和地址.
    BOARD_HOST="root@192.168.0.92"
    # 板端 ONNX Runtime 动态库路径.
    ORT_RUNTIME_LIB="/usr/lib/libonnxruntime.so.1.20.2"
fi

# 板端阶段: 此模型由 ONNX Runtime Neuron EP 在线编译,没有离线 DLA.
if [[ -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    source "${SCRIPT_DIR}/board_paths.conf"
    readonly RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    readonly RUN_DIR="${BOARD_RESULTS_DIR}/${RUN_ID}"
    readonly ANNOTATIONS="${BOARD_DATASET_DIR}/annotations/instances_val2017.json"
    test -s "${SCRIPT_DIR}/model_fp32_pure_npu.onnx"
    test -s "${ANNOTATIONS}"
    test "$(find "${BOARD_DATASET_DIR}/images" -maxdepth 1 -type f -name '*.jpg' | wc -l)" -eq 5000
    test -s "${ORT_RUNTIME_LIB}"
    python3 -c 'import pycocotools'
    test ! -e "${RUN_DIR}"
    mkdir -p "${RUN_DIR}/raw" "${RUN_DIR}/report"

    echo "[1/3] 在板端执行 5000 张 Neuron EP 推理."
    "${SCRIPT_DIR}/yoloworld_board_eval" \
        --model "${SCRIPT_DIR}/model_fp32_pure_npu.onnx" \
        --images "${BOARD_DATASET_DIR}/images" \
        --output-dir "${RUN_DIR}/raw" \
        2>&1 | tee "${RUN_DIR}/board_eval.log"
    echo "[2/3] 计算 COCO bbox AP."
    python3 "${SCRIPT_DIR}/evaluate_full_coco.py" \
        --annotations "${ANNOTATIONS}" \
        --results "${RUN_DIR}/raw/results.jsonl" \
        --profile "$(cat "${RUN_DIR}/raw/profile_path.txt")" \
        --output-dir "${RUN_DIR}/report" --run-id "${RUN_ID}" \
        2>&1 | tee "${RUN_DIR}/cocoeval.log"
    echo "[3/3] 记录运行环境."
    cp "${RUN_DIR}/board_eval.log" "${RUN_DIR}/cocoeval.log" \
        "${RUN_DIR}/report/"
    uname -a > "${RUN_DIR}/report/system.txt"
    echo "[OK] 报告: ${RUN_DIR}/report"
    exit 0
fi

if (( $# != 0 )); then
    echo "[ERROR] 在 89 直接运行 bash deploy/run.sh,无需参数." >&2
    exit 2
fi

# 路径配置: 原始模型、生成位置、板端数据与工具链均可覆盖.
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
readonly SOURCE_ONNX="${SOURCE_ONNX:-${MODEL_ROOT}/models/model_fp32.onnx}"
readonly MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-${MODEL_ROOT}/models}"
readonly BOARD_DATASET_DIR="${BOARD_DATASET_DIR:?请指定板端 COCO 数据集目录}"
readonly BOARD_DEPLOY_DIR="${BOARD_DEPLOY_DIR:?请指定 BOARD_DEPLOY_DIR}"
readonly BOARD_RESULTS_DIR="${BOARD_RESULTS_DIR:-${BOARD_DEPLOY_DIR}/results}"
readonly ORT_RUNTIME_LIB="${ORT_RUNTIME_LIB:-/usr/lib/libonnxruntime.so.1.20.2}"
readonly BOARD_HOST="${BOARD_HOST:-root@192.168.0.92}"
readonly CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
readonly TOOLCHAIN_ROOT="${MTK_G720_CPP_TOOLCHAIN_ROOT:-/data/users/hailong.he/data/MTKG720/cpp_toolchain}"
readonly ORT_HEADER="${ORT_HEADER:-${REPO_ROOT}/tools/third_party/onnxruntime/onnxruntime_c_api.h}"
readonly CXX="${CROSS_CXX:-aarch64-linux-gnu-g++}"
readonly BINARY_OUTPUT="${BOARD_BINARY_OUTPUT:-${SCRIPT_DIR}/inference_demo/yoloworld_board_eval}"
readonly SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

test -s "${SOURCE_ONNX}"
test -s "${ORT_HEADER}"
if [[ ! "${BOARD_DEPLOY_DIR}" =~ ^/[A-Za-z0-9_./-]+$ ]]; then
    echo "[ERROR] BOARD_DEPLOY_DIR 必须是无空格的板端绝对路径." >&2
    exit 2
fi

echo "[1/3] 在 Docker 中准备兼容 ONNX 并验证等价性."
docker exec -i -e MODEL_ROOT="${MODEL_ROOT}" \
    -e SOURCE_ONNX="${SOURCE_ONNX}" \
    -e MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR}" \
    "${CONTAINER}" bash -s <<'DOCKER_BUILD'
set -euo pipefail
mkdir -p "${MODEL_OUTPUT_DIR}"
python3 "${MODEL_ROOT}/deploy/prepare_onnx.py" \
    --input "${SOURCE_ONNX}" \
    --output "${MODEL_OUTPUT_DIR}/model_fp32_opset13.onnx" \
    --raw-output "${MODEL_OUTPUT_DIR}/model_fp32_raw.onnx"
python3 "${MODEL_ROOT}/deploy/prepare_pure_npu_onnx.py" \
    --input "${MODEL_OUTPUT_DIR}/model_fp32_raw.onnx" \
    --output "${MODEL_OUTPUT_DIR}/model_fp32_pure_npu.onnx"
python3 "${MODEL_ROOT}/deploy/verify_onnx_equivalence.py" \
    --source "${SOURCE_ONNX}" \
    --converted "${MODEL_OUTPUT_DIR}/model_fp32_opset13.onnx" \
    --raw "${MODEL_OUTPUT_DIR}/model_fp32_raw.onnx" \
    --report "${MODEL_OUTPUT_DIR}/onnx_equivalence.json"
DOCKER_BUILD
test -s "${MODEL_OUTPUT_DIR}/model_fp32_opset13.onnx"
test -s "${MODEL_OUTPUT_DIR}/model_fp32_pure_npu.onnx"

echo "[2/3] 在 89 交叉编译板端 C++ 测试程序."
readonly OPENCV_SOURCE="${TOOLCHAIN_ROOT}/opencv-4.9.0/opencv-4.9.0"
readonly OPENCV_BUILD="${TOOLCHAIN_ROOT}/opencv-4.9.0/build-aarch64-headers"
readonly TARGET_LIBS="${TOOLCHAIN_ROOT}/genio720-libs"
command -v "${CXX}" >/dev/null 2>&1
mkdir -p "$(dirname "${BINARY_OUTPUT}")"
"${CXX}" -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic \
    -Wno-deprecated-enum-enum-conversion \
    -I"$(dirname "${ORT_HEADER}")" -I"${OPENCV_BUILD}" \
    -I"${OPENCV_SOURCE}/modules/core/include" \
    -I"${OPENCV_SOURCE}/modules/imgproc/include" \
    -I"${OPENCV_SOURCE}/modules/imgcodecs/include" \
    "${SCRIPT_DIR}/inference_demo/yoloworld_board_eval.cpp" \
    "${TARGET_LIBS}/libopencv_imgcodecs.so.409" \
    "${TARGET_LIBS}/libopencv_imgproc.so.409" \
    "${TARGET_LIBS}/libopencv_core.so.409" \
    -Wl,--allow-shlib-undefined -pthread -ldl -o "${BINARY_OUTPUT}"
file "${BINARY_OUTPUT}"

echo "[3/3] 上传模型、程序、评测代码和路径配置到板端."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" "mkdir -p '${BOARD_DEPLOY_DIR}'"
scp "${SSH_OPTIONS[@]}" "${MODEL_OUTPUT_DIR}/model_fp32_pure_npu.onnx" \
    "${SCRIPT_DIR}/inference_demo/evaluate_full_coco.py" \
    "${SCRIPT_DIR}/run.sh" "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/"
scp "${SSH_OPTIONS[@]}" "${BINARY_OUTPUT}" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/yoloworld_board_eval"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "chmod 755 '${BOARD_DEPLOY_DIR}/yoloworld_board_eval'"
printf 'BOARD_DATASET_DIR=%q\nBOARD_RESULTS_DIR=%q\nORT_RUNTIME_LIB=%q\n' \
    "${BOARD_DATASET_DIR}" "${BOARD_RESULTS_DIR}" "${ORT_RUNTIME_LIB}" |
    ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
        "cat > '${BOARD_DEPLOY_DIR}/board_paths.conf'"
echo "[OK] 在板端运行: bash '${BOARD_DEPLOY_DIR}/run.sh'"
