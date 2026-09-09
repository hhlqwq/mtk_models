#!/usr/bin/env bash

set -euo pipefail

echo "[ERROR] RTMPose 官方 MMPose 版本和权重尚未锁定,已禁止使用历史 Qualcomm ONNX 作为正式源模型." >&2
echo "[NEXT] 请先记录官方配置、源码版本、权重地址、许可证和 SHA-256,再实现原始框架导出流程." >&2
exit 2
