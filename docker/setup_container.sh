#!/usr/bin/env bash
set -euo pipefail
# 兼容原入口, 现在只校验, 不修改环境.
python /opt/mtk-build/verify_environment.py "${1:-}"
/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/bin/ncc-tflite --version
