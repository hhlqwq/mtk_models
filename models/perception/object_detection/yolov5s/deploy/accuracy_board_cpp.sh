#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_ROOT="${MTK_BOARD_ROOT:-/root/hailong.he}"
readonly BOARD_MODEL_DIR="${BOARD_ROOT}/yolov5s_cpp"
readonly BOARD_DATASET="${BOARD_ROOT}/datasets/coco/val2017"
readonly RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
readonly BOARD_OUTPUT="${BOARD_MODEL_DIR}/runs/${RUN_ID}"
readonly LOCAL_OUTPUT="${MODEL_ROOT}/examples/output/board_cpp_accuracy/${RUN_ID}"
readonly BINARY="${SCRIPT_DIR}/inference_demo/yolov5s_board_eval"

echo "[1/6] 交叉编译板端 C++ 评测程序。"
bash "${SCRIPT_DIR}/build_board_cpp.sh"

echo "[2/6] 检查板端 COCO 数据集。"
image_count="$(ssh "${BOARD_HOST}" \
    "find '${BOARD_DATASET}/images' -maxdepth 1 -type f -name '*.jpg' | wc -l")"
test "${image_count}" -eq 5000
ssh "${BOARD_HOST}" "test -f '${BOARD_DATASET}/annotations/instances_val2017.json'"

echo "[3/6] 部署 C++ 程序、DLA 和板端指标脚本。"
ssh "${BOARD_HOST}" "mkdir -p '${BOARD_MODEL_DIR}' '${BOARD_OUTPUT}'"
scp "${BINARY}" "${MODEL_ROOT}/models/model_int8.dla" \
    "${SCRIPT_DIR}/inference_demo/evaluate_coco.py" \
    "${BOARD_HOST}:${BOARD_MODEL_DIR}/"

echo "[4/6] 在板端执行 5000 张 C++ 预处理、NPU 推理和后处理。"
ssh "${BOARD_HOST}" \
    "'${BOARD_MODEL_DIR}/yolov5s_board_eval' \
        --model '${BOARD_MODEL_DIR}/model_int8.dla' \
        --images '${BOARD_DATASET}/images' \
        --output-dir '${BOARD_OUTPUT}' \
        --warmup 20 --progress-interval 50 \
        2>&1 | tee '${BOARD_OUTPUT}/board_eval.log'"

echo "[5/6] 在板端使用 pycocotools 计算 COCO bbox 指标。"
ssh "${BOARD_HOST}" \
    "python3 '${BOARD_MODEL_DIR}/evaluate_coco.py' \
        --annotations '${BOARD_DATASET}/annotations/instances_val2017.json' \
        --predictions '${BOARD_OUTPUT}/predictions.json' \
        --processed-ids '${BOARD_OUTPUT}/processed_ids.txt' \
        --metrics '${BOARD_OUTPUT}/coco_metrics.json' \
        --summary-log '${BOARD_OUTPUT}/coco_summary.log' \
        2>&1 | tee '${BOARD_OUTPUT}/cocoeval.log'"

echo "[6/6] 仅回传指标、摘要和日志证据。"
mkdir -p "${LOCAL_OUTPUT}"
scp "${BOARD_HOST}:${BOARD_OUTPUT}/coco_metrics.json" \
    "${BOARD_HOST}:${BOARD_OUTPUT}/coco_summary.log" \
    "${BOARD_HOST}:${BOARD_OUTPUT}/board_eval.log" \
    "${BOARD_HOST}:${BOARD_OUTPUT}/cocoeval.log" \
    "${BOARD_HOST}:${BOARD_OUTPUT}/timing_summary_current_run.json" \
    "${BOARD_HOST}:${BOARD_OUTPUT}/timings.csv" \
    "${LOCAL_OUTPUT}/"
echo "[OK] 板端 C++ 全量精度测试完成: ${LOCAL_OUTPUT}"
