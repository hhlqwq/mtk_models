#!/usr/bin/env bash
# 一条命令完成 YOLOv5s 交叉编译及 92 全量 COCO 精度测试,不清理现场.

set -euo pipefail
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/accuracy_board_cpp.sh"
