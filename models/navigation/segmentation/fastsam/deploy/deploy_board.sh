#!/usr/bin/env bash
# 在 89 宿主执行,每次部署使用独立目录,保留已有板端证据.
set -euo pipefail
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly RUN_ID="${FASTSAM_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
[[ "${RUN_ID}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "非法 RUN_ID"; exit 1; }
readonly BOARD_DIR="/root/hailong.he/fastsam/runs/${RUN_ID}"
: "${FASTSAM_IMAGE:?请指定与导出基线相同的图片}"
for name in model_int8.dla model_int8.json model_fp32.onnx pytorch_reference.npz export_manifest.json deployment_manifest.json; do
    test -s "${MODEL_ROOT}/models/${name}"
done
test -f "${FASTSAM_IMAGE}"
echo "[1/4] 创建独立板端运行目录 ${BOARD_DIR}."
ssh -o BatchMode=yes "${BOARD_HOST}" "test ! -e '${BOARD_DIR}' && mkdir -p '${BOARD_DIR}/deploy/inference_demo' '${BOARD_DIR}/models'"
echo "[2/4] 传输模型、基线与 Python Demo."
scp "${MODEL_ROOT}/models/model_int8.dla" "${MODEL_ROOT}/models/model_int8.json" \
    "${MODEL_ROOT}/models/model_fp32.onnx" "${MODEL_ROOT}/models/pytorch_reference.npz" \
    "${MODEL_ROOT}/models/export_manifest.json" "${MODEL_ROOT}/models/deployment_manifest.json" \
    "${BOARD_HOST}:${BOARD_DIR}/models/"
scp "${SCRIPT_DIR}/fastsam_utils.py" "${SCRIPT_DIR}/compare_backends.py" "${BOARD_HOST}:${BOARD_DIR}/deploy/"
scp "${SCRIPT_DIR}/inference_demo/run_board.py" "${BOARD_HOST}:${BOARD_DIR}/deploy/inference_demo/"
scp "${FASTSAM_IMAGE}" "${BOARD_HOST}:${BOARD_DIR}/input.jpg"
echo "[3/4] 板端执行 ONNX 数值检查和 NPU 推理."
ssh -o BatchMode=yes "${BOARD_HOST}" "cd '${BOARD_DIR}' && \
    python3 deploy/inference_demo/run_board.py --backend onnx --model models/model_fp32.onnx --image input.jpg --output-dir onnx && \
    python3 deploy/compare_backends.py --reference models/pytorch_reference.npz --candidate onnx/raw_outputs.npz --export-manifest models/export_manifest.json --output onnx/comparison.json --strict-fp32 && \
    python3 deploy/inference_demo/run_board.py --backend npu --model models/model_int8.dla --metadata models/model_int8.json --image input.jpg --output-dir npu --iterations 100 && \
    python3 deploy/compare_backends.py --reference onnx/raw_outputs.npz --candidate npu/raw_outputs.npz --export-manifest models/export_manifest.json --output npu/comparison.json"
echo "[4/4] 回收本次运行证据."
mkdir -p "${MODEL_ROOT}/examples/output/runs/${RUN_ID}"
scp -r "${BOARD_HOST}:${BOARD_DIR}/onnx" "${BOARD_HOST}:${BOARD_DIR}/npu" \
    "${MODEL_ROOT}/examples/output/runs/${RUN_ID}/"
echo "[OK] 请审核比较结果、纯 NPU 日志和叠加图后更新交付状态."
