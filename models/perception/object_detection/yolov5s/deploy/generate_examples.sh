#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly COCO_ROOT="${COCO_ROOT:-/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017}"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_DIR="${MTK_BOARD_ROOT:-/root/hailong.he}/yolov5s_examples"
readonly CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
readonly INPUT_DIR="${MODEL_ROOT}/examples/input/generated"
readonly OUTPUT_DIR="${MODEL_ROOT}/examples/output/generated"
readonly BINARY="${SCRIPT_DIR}/inference_demo/yolov5s_board_eval"
readonly -a SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

echo "[1/5] 选择五张 COCO 验证图片."
rm -rf "${INPUT_DIR}" "${OUTPUT_DIR}"
mkdir -p "${INPUT_DIR}" "${OUTPUT_DIR}"
find "${COCO_ROOT}/images" -maxdepth 1 -type f -name '*.jpg' -print0 \
    | sort -z | head -z -n 5 \
    | xargs -0 -I {} cp "{}" "${INPUT_DIR}/"
test "$(find "${INPUT_DIR}" -maxdepth 1 -type f -name '*.jpg' | wc -l)" -eq 5

echo "[2/5] 创建板端五图示例目录."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "rm -rf '${BOARD_DIR}' && mkdir -p '${BOARD_DIR}/images' '${BOARD_DIR}/output'"
echo "[3/5] 部署 DLA、C++ 推理器和五张图片."
scp "${SSH_OPTIONS[@]}" "${MODEL_ROOT}/models/model_int8.dla" "${BINARY}" \
    "${BOARD_HOST}:${BOARD_DIR}/"
scp "${SSH_OPTIONS[@]}" "${INPUT_DIR}"/*.jpg \
    "${BOARD_HOST}:${BOARD_DIR}/images/"
echo "[4/5] 在 Genio 720 执行五张图片推理."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "'${BOARD_DIR}/yolov5s_board_eval' --model '${BOARD_DIR}/model_int8.dla' \
      --images '${BOARD_DIR}/images' --output-dir '${BOARD_DIR}/output' \
      --limit 5 --warmup 2 --progress-interval 1"
scp "${SSH_OPTIONS[@]}" "${BOARD_HOST}:${BOARD_DIR}/output/predictions.jsonl" \
    "${BOARD_HOST}:${BOARD_DIR}/output/timing_summary.json" "${OUTPUT_DIR}/"
echo "[5/5] 生成检测框图片和单图 JSON."
docker exec "${CONTAINER}" python3 \
    "/workspace/models/perception/object_detection/yolov5s/deploy/inference_demo/render_examples.py" \
    --input-dir "/workspace/models/perception/object_detection/yolov5s/examples/input/generated" \
    --predictions "/workspace/models/perception/object_detection/yolov5s/examples/output/generated/predictions.jsonl" \
    --output-dir "/workspace/models/perception/object_detection/yolov5s/examples/output/generated" \
    --count 5
echo "[OK] YOLOv5s 五图示例: ${OUTPUT_DIR}"
