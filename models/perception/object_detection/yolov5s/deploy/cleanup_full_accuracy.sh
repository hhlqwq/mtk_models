#!/usr/bin/env bash
# 用户上传报告后,单独清理 YOLOv5s 本次板端大体积文件.

set -euo pipefail
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly RUN_ID="${EVAL_RUN_ID:?请设置待清理的 EVAL_RUN_ID}"
MODEL_KEY=yolov5s RUN_ID="${RUN_ID}" \
    bash "${SCRIPT_DIR}/../../../../../tools/evaluation/finalize_board_result.sh"
