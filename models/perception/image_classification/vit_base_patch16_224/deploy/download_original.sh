#!/usr/bin/env bash

# 名称为兼容统一交付结构而保留: 实际只执行离线校验、解压和 SHA-256 记录,
# 不包含任何网络下载。vit-onnx-float.zip 由用户下载到本机后同步到 89。
set -euo pipefail

readonly MODEL_ROOT="/workspace/models/perception/image_classification/vit_base_patch16_224"
readonly EXPECTED_ZIP="${MODEL_ROOT}/original/vit-onnx-float.zip"
readonly USER_ZIP="${MODEL_ROOT}/models/vit-onnx-float.zip"
readonly EXPECTED_SHA256="72b7d02dd5c3d1e09c59196ba14364549aac9e8ed1f8212ea5f5c78f7424632f"

echo "[1/3] 定位用户放置的 Qualcomm ViT FP32 ONNX 归档。"
if [[ -f "${EXPECTED_ZIP}" ]]; then
    ARCHIVE="${EXPECTED_ZIP}"
elif [[ -f "${USER_ZIP}" ]]; then
    ARCHIVE="${USER_ZIP}"
    mkdir -p "${MODEL_ROOT}/original"
    mv "${ARCHIVE}" "${EXPECTED_ZIP}"
    ARCHIVE="${EXPECTED_ZIP}"
else
    echo "[ERROR] 未找到 vit-onnx-float.zip, 请按 README 先由用户下载并放置。" >&2
    exit 2
fi
echo "${EXPECTED_SHA256}  ${ARCHIVE}" | sha256sum --check --status || {
    echo "[ERROR] ViT 归档 SHA-256 不匹配: ${ARCHIVE}" >&2
    exit 1
}

echo "[2/3] 离线解压并提取 ONNX。"
rm -rf "${MODEL_ROOT}/original/exported"
mkdir -p "${MODEL_ROOT}/original/exported"
unzip -o -q "${ARCHIVE}" -d "${MODEL_ROOT}/original/exported"
mapfile -t ONNX_PATHS < <(find "${MODEL_ROOT}/original/exported" \
    -type f -name '*.onnx' | sort)
if [[ "${#ONNX_PATHS[@]}" -ne 1 ]]; then
    echo "归档内 ONNX 数量错误: 期望 1, 实际 ${#ONNX_PATHS[@]}。" >&2
    exit 1
fi
readonly ONNX_PATH="${ONNX_PATHS[0]}"
# Qualcomm 导出的 vit.onnx 权重存放在同目录 vit.data 外部文件中,
# 直接复制 .onnx 会得到无法加载的残缺模型, 必须先合并为单文件。
python - "${ONNX_PATH}" "${MODEL_ROOT}/models/model_fp32.onnx" <<'PY'
"""将带外部数据的 ONNX 合并为单文件, 无外部数据时直接复制。"""
import shutil
import sys

import onnx

source, target = sys.argv[1], sys.argv[2]
# 先只加载图结构检测外部权重: onnx.load 默认会把外部数据读入内存,
# 并将 data_location 重置为 DEFAULT, 无法用于判断。
stub = onnx.load(source, load_external_data=False)
has_external = any(
    tensor.data_location == onnx.TensorProto.EXTERNAL
    for tensor in stub.graph.initializer)
if has_external:
    del stub
    model = onnx.load(source)  # 自动加载同目录外部权重
    onnx.save(model, target, save_as_external_data=False)
    print(f"[INFO] 已合并外部权重为单文件: {source} -> {target}")
else:
    shutil.copyfile(source, target)
    print(f"[INFO] 无外部权重, 直接复制: {source} -> {target}")
PY

# Qualcomm 导出为 IR v10 / opset 21 且含 Gelu 算子, 超出 mtk_converter
# (onnx 1.13.1) 的 IR v3..v8 / opset <=18 支持范围, 需等价降级。
python "${MODEL_ROOT}/deploy/downgrade_onnx.py" \
    --model "${MODEL_ROOT}/models/model_fp32.onnx"

readonly LABELS_PATH="$(find "${MODEL_ROOT}/original/exported" \
    -type f -name 'labels.txt' | sort | head -n 1)"
if [[ -n "${LABELS_PATH}" ]]; then
    cp "${LABELS_PATH}" "${MODEL_ROOT}/original/labels.txt"
fi

echo "[3/3] 记录来源校验值。"
sha256sum "${ARCHIVE}" "${MODEL_ROOT}/models/model_fp32.onnx" \
    > "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] ViT FP32 ONNX 已准备: ${ONNX_PATH}"
