#!/usr/bin/env bash

set -euo pipefail

readonly MODEL_ROOT="/workspace/models/perception/object_detection/yolov5s"
readonly CALIBRATION_DIR="/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017/images"
readonly SOURCE_DIR="${MODEL_ROOT}/original/yolov5"
readonly CONSTRAINTS_FILE="${MODEL_ROOT}/deploy/constraints-py311.txt"

test -f "${MODEL_ROOT}/models/yolov5s.pt"
test -d "${CALIBRATION_DIR}"

echo "[1/3] 安装 MTK 兼容的 YOLOv5 导出依赖。"
python -m pip install --no-cache-dir \
    torch==2.0.0+cu118 torchvision==0.15.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Torch 2.0 CUDA wheel 使用哈希化 NVRTC 文件名，cuDNN 仍按标准名动态加载。
readonly TORCH_LIB_DIR="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')"
readonly NVRTC_LIBRARY="$(find "${TORCH_LIB_DIR}" -maxdepth 1 -type f \
    -name 'libnvrtc-*.so.*' -print -quit)"
test -n "${NVRTC_LIBRARY}"
ln -sfn "$(basename "${NVRTC_LIBRARY}")" "${TORCH_LIB_DIR}/libnvrtc.so"

python -m pip install --no-cache-dir \
    --constraint "${CONSTRAINTS_FILE}" \
    --requirement "${SOURCE_DIR}/requirements.txt"

echo "[VERIFY] 检查 YOLOv5 和 MTK Converter 依赖。"
python -m pip check
python -c "import cv2, mtk_converter, numpy, torch, torchvision; import ultralytics.yolo; assert torch.cuda.is_available(), 'CUDA 不可用'; print('[VERIFY] numpy:', numpy.__version__); print('[VERIFY] opencv:', cv2.__version__); print('[VERIFY] torch:', torch.__version__); print('[VERIFY] torchvision:', torchvision.__version__); print('[VERIFY] cuda:', torch.version.cuda); print('[VERIFY] gpu:', torch.cuda.get_device_name(0)); print('[VERIFY] mtk_converter:', mtk_converter.__version__)"
python -c "import torch; layer=torch.nn.Conv2d(3, 8, 3).cuda(); data=torch.randn(1, 3, 64, 64).cuda(); output=layer(data); torch.cuda.synchronize(); print('[VERIFY] CUDA Conv2d:', tuple(output.shape))"

echo "[2/3] 导出 MTK 转换用 TorchScript 和 FP32 ONNX。"
cd "${SOURCE_DIR}"
python export.py \
    --weights "${MODEL_ROOT}/models/yolov5s.pt" \
    --img-size 640 640 \
    --batch-size 1 \
    --device 0 \
    --include torchscript onnx
mv "${MODEL_ROOT}/models/yolov5s.onnx" "${MODEL_ROOT}/models/model_fp32.onnx"

echo "[3/3] 使用 MTK PyTorch Converter 执行 INT8 PTQ。"
python "${MODEL_ROOT}/deploy/convert_int8.py" \
    --torchscript "${MODEL_ROOT}/models/yolov5s.torchscript" \
    --calibration-dir "${CALIBRATION_DIR}" \
    --output "${MODEL_ROOT}/models/model_int8.tflite"
sha256sum "${MODEL_ROOT}/models/model_fp32.onnx" \
    "${MODEL_ROOT}/models/yolov5s.torchscript" \
    "${MODEL_ROOT}/models/model_int8.tflite" \
    >> "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] YOLOv5s ONNX 和 INT8 TFLite 已生成。"
