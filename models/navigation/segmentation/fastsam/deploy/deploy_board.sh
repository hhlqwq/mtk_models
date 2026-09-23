#!/usr/bin/env bash
# 在 89 宿主部署 C++ 可执行程序并执行一张真实图片的硬件冒烟.
set -euo pipefail
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly RUN_ID="${FASTSAM_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
[[ "${RUN_ID}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "非法 RUN_ID"; exit 1; }
readonly BOARD_MODEL_ROOT="${MTK_BOARD_OPEN_MODELS_ROOT:-/root/hailong.he/open_models}/fastsam"
readonly BOARD_MODEL_DIR="${BOARD_MODEL_ROOT}/models"
readonly BOARD_DIR="${BOARD_MODEL_ROOT}/runs/${RUN_ID}"
readonly BOARD_IMAGE_DIR="${MTK_BOARD_DATASETS_ROOT:-/root/hailong.he/datasets}/fastsam/${RUN_ID}"
: "${FASTSAM_IMAGE:?请指定与导出基线相同的图片}"
for name in model_int8.dla model_int8.json runtime_config.csv deployment_manifest.json; do
    test -s "${MODEL_ROOT}/models/${name}"
done
test -s "${SCRIPT_DIR}/inference_demo/fastsam_board"
test -s "${FASTSAM_IMAGE}"
echo "[1/4] 核对部署清单与原始图片."
python "${SCRIPT_DIR}/verify_deployment.py" \
    --model-dir "${MODEL_ROOT}/models" --image "${FASTSAM_IMAGE}"
echo "[2/4] 创建独立板端目录并传输 C++ 程序及模型."
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "${BOARD_HOST}" \
    "test ! -e '${BOARD_DIR}' && mkdir -p '${BOARD_DIR}' '${BOARD_MODEL_DIR}' '${BOARD_IMAGE_DIR}'"
scp "${SCRIPT_DIR}/inference_demo/fastsam_board" \
    "${BOARD_HOST}:${BOARD_DIR}/"
scp "${MODEL_ROOT}/models/model_int8.dla" \
    "${MODEL_ROOT}/models/runtime_config.csv" \
    "${BOARD_HOST}:${BOARD_MODEL_DIR}/"
scp "${FASTSAM_IMAGE}" "${BOARD_HOST}:${BOARD_IMAGE_DIR}/"
echo "[3/4] 在 Genio 720 上调用 Neuron Runtime 硬件推理."
readonly IMAGE_NAME="$(basename -- "${FASTSAM_IMAGE}")"
readonly BOARD_IMAGE="${BOARD_IMAGE_DIR}/${IMAGE_NAME}"
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "${BOARD_HOST}" \
    "cd '${BOARD_DIR}' && chmod +x fastsam_board && \
    ./fastsam_board --model '${BOARD_MODEL_DIR}/model_int8.dla' \
    --config '${BOARD_MODEL_DIR}/runtime_config.csv' \
    --image '${BOARD_IMAGE}' --output-dir result > smoke.log 2>&1 && \
    sha256sum fastsam_board '${BOARD_MODEL_DIR}/model_int8.dla' \
    '${BOARD_MODEL_DIR}/runtime_config.csv' '${BOARD_IMAGE}' \
    > SHA256SUMS"
echo "[4/4] 回收本次冒烟证据."
mkdir -p "${MODEL_ROOT}/examples/output/runs/${RUN_ID}"
scp -r "${BOARD_HOST}:${BOARD_DIR}/result" \
    "${BOARD_HOST}:${BOARD_DIR}/smoke.log" \
    "${BOARD_HOST}:${BOARD_DIR}/SHA256SUMS" \
    "${MODEL_ROOT}/examples/output/runs/${RUN_ID}/"
echo "[OK] 请审核 results.json、掩码图和 smoke.log 后更新状态."
