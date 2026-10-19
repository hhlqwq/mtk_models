#!/usr/bin/env bash
# 兼容原有环境检查入口,公共环境随 docker/enter_container.sh 加载.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${ROOT}/verify_environment.py" "${1:-}"
"${NCC_ROOT:?请通过 docker/enter_container.sh 进入容器}/bin/ncc-tflite" --version
