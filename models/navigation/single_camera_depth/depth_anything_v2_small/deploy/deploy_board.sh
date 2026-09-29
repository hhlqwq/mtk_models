#!/usr/bin/env bash

set -euo pipefail
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly RUN_ID="${DEPTH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly BOARD_HOST="root@192.168.0.92"
readonly BOARD_DIR="/root/hailong.he/open_models/depth_anything_v2_small/smoke/${RUN_ID}"
readonly RUN_DIR="${MODEL_ROOT}/examples/output/runs/${RUN_ID}"
readonly IMAGE_ROOT="/data/users/hailong.he/github/mtk_models/models/perception/object_detection/yolov5s/examples/input"

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "运行编号仅允许字母、数字、下划线和连字符。" >&2
    exit 1
fi
test -s "${MODEL_ROOT}/models/model_int8.dla"
test -s "${MODEL_ROOT}/models/model_int8.json"
test -s "${IMAGE_ROOT}/000000000001.jpg"
test -s "${IMAGE_ROOT}/000000000002.jpg"
test ! -e "${RUN_DIR}"
ssh -o BatchMode=yes "${BOARD_HOST}" "test ! -e '${BOARD_DIR}'"
mkdir -p "${RUN_DIR}/output"
echo "[1/4] 在现有容器准备原生对齐 INT8 输入。"
docker exec hhl_g720_8011 python "${SCRIPT_DIR}/prepare_input.py" \
    --metadata "${MODEL_ROOT}/models/model_int8.json" \
    --image "${IMAGE_ROOT}/000000000001.jpg" \
        "${IMAGE_ROOT}/000000000002.jpg" \
    --output-dir "${RUN_DIR}/inputs"
echo "[2/4] 传输模型和输入至 Genio 720。"
ssh -o BatchMode=yes "${BOARD_HOST}" "mkdir -p '${BOARD_DIR}/inputs'"
scp -o BatchMode=yes "${MODEL_ROOT}/models/model_int8.dla" \
    "${SCRIPT_DIR}/run_board.sh" "${BOARD_HOST}:${BOARD_DIR}/"
scp -o BatchMode=yes "${RUN_DIR}"/inputs/image_*.bin \
    "${BOARD_HOST}:${BOARD_DIR}/inputs/"
echo "[3/4] 执行三次真实硬件推理。"
ssh -o BatchMode=yes "${BOARD_HOST}" "sh '${BOARD_DIR}/run_board.sh'"
scp -o BatchMode=yes "${BOARD_HOST}:${BOARD_DIR}/output/*" \
    "${RUN_DIR}/output/"
echo "[4/4] 对比官方 PyTorch 参考并保存报告。"
docker exec hhl_g720_8011 python "${SCRIPT_DIR}/verify_board.py" \
    --run-dir "${RUN_DIR}" --metadata "${MODEL_ROOT}/models/model_int8.json" \
    --upstream "${MODEL_ROOT}/original/upstream" \
    --weights "${MODEL_ROOT}/original/depth_anything_v2_vits.pth"
echo "[OK] 板端证据: ${RUN_DIR}/output/report.json"
