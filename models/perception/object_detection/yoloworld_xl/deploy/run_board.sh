#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_ROOT="${MTK_BOARD_ROOT:-/root/hailong.he}"
readonly RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
readonly BOARD_MODEL="${BOARD_ROOT}/yoloworld_xl/model/model_fp32_opset13.onnx"
readonly BOARD_IMAGES="${BOARD_ROOT}/yoloworld_xl/demo/public/images"
readonly BOARD_OUTPUT="${BOARD_ROOT}/yoloworld_xl/demo/public/${RUN_ID}"
readonly LOCAL_OUTPUT="${MODEL_ROOT}/examples/output/board/${RUN_ID}"
readonly SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "[ERROR] RUN_ID 只能包含字母、数字、点、下划线和连字符。" >&2
    exit 1
fi
if [[ -e "${LOCAL_OUTPUT}" ]]; then
    echo "[ERROR] 本地输出目录已存在: ${LOCAL_OUTPUT}。" >&2
    exit 1
fi

echo "[1/7] 部署模型、脚本和公开样例。"
bash "${SCRIPT_DIR}/deploy_board.sh"

echo "[2/7] 检查板端运行目录没有旧结果。"
if ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" "test -e '${BOARD_OUTPUT}'"; then
    echo "[ERROR] 板端输出目录已存在: ${BOARD_OUTPUT}。" >&2
    exit 1
fi

echo "[3/7] 保存板端环境和输入哈希。"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_OUTPUT}" "${BOARD_MODEL}" "${BOARD_IMAGES}" <<'BOARD_PREPARE'
set -euo pipefail
readonly output_dir="$1"
readonly model_path="$2"
readonly images_dir="$3"
mkdir -p "${output_dir}"
{
    echo "captured_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uname=$(uname -a)"
    cat /etc/os-release
    python3 -c 'import cv2,numpy,onnxruntime; print("opencv=" + cv2.__version__); print("numpy=" + numpy.__version__); print("onnxruntime=" + onnxruntime.__version__); print("providers=" + str(onnxruntime.get_available_providers()))'
} > "${output_dir}/system.txt"
sha256sum "${model_path}" > "${output_dir}/model_sha256.txt"
find "${images_dir}" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${output_dir}/images_sha256.txt"
BOARD_PREPARE

echo "[4/7] 执行 CPU EP 三图基线，每张图运行 1 次。"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "cd '${BOARD_ROOT}/yoloworld_xl/model' && \
     python3 run_board.py \
        --model '${BOARD_MODEL}' \
        --images '${BOARD_IMAGES}' \
        --output-dir '${BOARD_OUTPUT}/cpu' \
        --provider cpu --warmup 0 --repeat 1 --profile \
        2>&1 | tee '${BOARD_OUTPUT}/cpu.log'"

echo "[5/7] 执行 Neuron EP 三图验证，每张图预热后运行 10 次。"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "cd '${BOARD_ROOT}/yoloworld_xl/model' && \
     python3 run_board.py \
        --model '${BOARD_MODEL}' \
        --images '${BOARD_IMAGES}' \
        --output-dir '${BOARD_OUTPUT}/neuron' \
        --provider neuron --warmup 3 --repeat 10 --profile \
        2>&1 | tee '${BOARD_OUTPUT}/neuron.log'"

echo "[6/7] 固化板端输出哈希。"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "cd '${BOARD_OUTPUT}' && find . -type f ! -name outputs_sha256.txt \
        -print0 | sort -z | xargs -0 sha256sum > outputs_sha256.txt"

echo "[7/7] 回传三图结果和运行证据。"
mkdir -p "${LOCAL_OUTPUT}"
scp "${SSH_OPTIONS[@]}" -r "${BOARD_HOST}:${BOARD_OUTPUT}/." "${LOCAL_OUTPUT}/"
echo "[OK] YOLO-World XL 板端运行完成: ${LOCAL_OUTPUT}。"
