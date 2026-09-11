#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_DIR="${MTK_BOARD_ROOT:-/root/hailong.he}/yolov5s_public_examples"
readonly CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
readonly INPUT_DIR="${MODEL_ROOT}/examples/input/public"
readonly OUTPUT_DIR="${MODEL_ROOT}/examples/output/public"
readonly BINARY="${SCRIPT_DIR}/inference_demo/yolov5s_board_eval"
readonly -a SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

test "$(find "${INPUT_DIR}" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 3
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "[1/6] 交叉编译支持 PNG 的 Genio 720 推理器."
bash "${SCRIPT_DIR}/build_board_cpp.sh"
echo "[2/6] 创建干净的板端三图运行目录."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "rm -rf '${BOARD_DIR}' && mkdir -p '${BOARD_DIR}/images' '${BOARD_DIR}/output'"
echo "[3/6] 部署 DLA、C++ 推理器和三张公开图片."
scp "${SSH_OPTIONS[@]}" "${MODEL_ROOT}/models/model_int8.dla" "${BINARY}" \
    "${BOARD_HOST}:${BOARD_DIR}/"
scp "${SSH_OPTIONS[@]}" "${INPUT_DIR}"/*.png \
    "${BOARD_HOST}:${BOARD_DIR}/images/"
echo "[4/6] 在 Genio 720 执行三张图片推理."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "'${BOARD_DIR}/yolov5s_board_eval' --model '${BOARD_DIR}/model_int8.dla' \
      --images '${BOARD_DIR}/images' --output-dir '${BOARD_DIR}/output' \
      --limit 3 --warmup 2 --progress-interval 1 --confidence 0.25 \
      --iou 0.45 --max-det 100"
echo "[5/6] 回收板端检测和耗时结果."
scp "${SSH_OPTIONS[@]}" "${BOARD_HOST}:${BOARD_DIR}/output/predictions.jsonl" \
    "${BOARD_HOST}:${BOARD_DIR}/output/timing_summary_current_run.json" \
    "${OUTPUT_DIR}/"
echo "[6/6] 生成检测框图片和单图 JSON."
docker exec "${CONTAINER}" python3 \
    "/workspace/models/perception/object_detection/yolov5s/deploy/inference_demo/render_examples.py" \
    --input-dir "/workspace/models/perception/object_detection/yolov5s/examples/input/public" \
    --predictions "/workspace/models/perception/object_detection/yolov5s/examples/output/public/predictions.jsonl" \
    --output-dir "/workspace/models/perception/object_detection/yolov5s/examples/output/public" \
    --count 3
echo "[OK] YOLOv5s 三张公开图片已在 Genio 720 完成测试: ${OUTPUT_DIR}"
