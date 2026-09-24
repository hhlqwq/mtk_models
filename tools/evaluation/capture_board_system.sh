#!/usr/bin/env bash
# 在 92 记录本次评测的系统和 Runtime 身份,不修改系统设置.

set -euo pipefail

readonly REPORT_DIR="${1:?请指定本次报告目录}"
test -s "${REPORT_DIR}/summary.json"

{
    echo "captured_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "kernel=$(uname -a)"
    echo "[os_release]"
    cat /etc/os-release
    echo "[memory]"
    free -h
    echo "[disk]"
    df -h "${REPORT_DIR}"
    echo "[runtime_libraries]"
    ldconfig -p 2>/dev/null | grep -E \
        'lib(neuronusdk_runtime|onnxruntime|opencv_core)' || true
} > "${REPORT_DIR}/system.txt"
