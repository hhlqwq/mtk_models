#!/usr/bin/env bash
# 在 89 的 Genio 720 Docker 容器中完成转换和 DLA 编译.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/2] 使用本地资源导出 ONNX 并转换 INT8 TFLite."
bash "${SCRIPT_DIR}/convert.sh"
echo "[2/2] 编译 Genio 720 DLA."
bash "${SCRIPT_DIR}/build.sh"
echo "[OK] YOLOv5s 模型准备完成."
