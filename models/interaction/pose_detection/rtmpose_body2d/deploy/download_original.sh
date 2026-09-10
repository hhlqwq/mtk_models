#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="${MODEL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
readonly WEIGHTS_NAME="rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth"
readonly EXPECTED_SHA256="3da02694cd6479d3b333ff42ebd0723f96bfa06adac1db1e2e815ed2e9e1b02d"
readonly CONFIG_RELATIVE="configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py"

if [[ -f "${MODEL_ROOT}/original/${WEIGHTS_NAME}" ]]; then
    readonly WEIGHTS_PATH="${MODEL_ROOT}/original/${WEIGHTS_NAME}"
elif [[ -f "${MODEL_ROOT}/models/${WEIGHTS_NAME}" ]]; then
    readonly WEIGHTS_PATH="${MODEL_ROOT}/models/${WEIGHTS_NAME}"
else
    echo "[ERROR] 未找到 OpenMMLab 官方权重: ${WEIGHTS_NAME}" >&2
    echo "[NEXT] 请由用户下载后放入 original/ 或 models/." >&2
    exit 2
fi

echo "[1/4] 校验 OpenMMLab RTMPose-M 官方权重."
echo "${EXPECTED_SHA256}  ${WEIGHTS_PATH}" | sha256sum --check --status || {
    echo "[ERROR] RTMPose 权重 SHA-256 不匹配: ${WEIGHTS_PATH}" >&2
    exit 1
}

if [[ -n "${MMPOSE_ROOT:-}" ]]; then
    readonly CONFIG_PATH="${MMPOSE_ROOT}/${CONFIG_RELATIVE}"
else
    readonly CONFIG_PATH="$(python - "${CONFIG_RELATIVE}" <<'PY'
import sys
from pathlib import Path

import mmpose

relative_path = Path(sys.argv[1])
package_root = Path(mmpose.__file__).resolve().parent
candidates = [
    package_root.parent / relative_path,
    package_root / ".mim" / relative_path,
]
for candidate in candidates:
    if candidate.is_file():
        print(candidate)
        break
else:
    raise FileNotFoundError(
        "未在 MMPose v1.3.2 安装目录中找到官方 RTMPose 配置.")
PY
)"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[ERROR] 缺少 MMPose v1.3.2 官方配置: ${CONFIG_PATH}" >&2
    exit 2
fi

echo "[2/4] 从官方 PyTorch 权重自行导出 ONNX."
python "${SCRIPT_DIR}/export_onnx.py" \
    --config "${CONFIG_PATH}" \
    --weights "${WEIGHTS_PATH}" \
    --output "${MODEL_ROOT}/models/model_fp32.onnx"

echo "[3/4] 生成 MTK 兼容候选并验证数值等价性."
cp "${MODEL_ROOT}/models/model_fp32.onnx" \
    "${MODEL_ROOT}/models/model_mtk_compatible.onnx"
python "${SCRIPT_DIR}/prepare_onnx.py" \
    --source "${MODEL_ROOT}/models/model_fp32.onnx" \
    --output "${MODEL_ROOT}/models/model_mtk_compatible.onnx"
python "${SCRIPT_DIR}/verify_onnx_equivalence.py" \
    --reference "${MODEL_ROOT}/models/model_fp32.onnx" \
    --converted "${MODEL_ROOT}/models/model_mtk_compatible.onnx"

echo "[4/4] 记录本地权重和 ONNX 校验值."
sha256sum "${WEIGHTS_PATH}" \
    "${MODEL_ROOT}/models/model_fp32.onnx" \
    "${MODEL_ROOT}/models/model_mtk_compatible.onnx" \
    > "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] RTMPose 开源上游 ONNX 已离线准备."
