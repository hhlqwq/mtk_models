#!/usr/bin/env bash
# 在 89 编译 ViT,在 92 完整评测 ImageNet val 50000 张图片.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
readonly CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_DATASET="${MTK_BOARD_DATASETS_ROOT:-/root/hailong.he/datasets}/imagenet"
readonly BOARD_MODEL_ROOT="${MTK_BOARD_OPEN_MODELS_ROOT:-/root/hailong.he/open_models}/vit_base_patch16_224"
readonly BOARD_RUN="${BOARD_MODEL_ROOT}/eval/${RUN_ID}"
readonly BOARD_MODEL_DIR="${BOARD_MODEL_ROOT}/models/${RUN_ID}"
readonly TEMP_DIR="/tmp/hailongcodex/$(date +%Y%m%d)/vit_${RUN_ID}"
readonly CONTAINER_TEMP="${TEMP_DIR}/quantization.json"
readonly -a SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "[ERROR] EVAL_RUN_ID 非法." >&2
    exit 2
fi
test -s "${MODEL_ROOT}/models/model_int8.tflite"
bash "${SCRIPT_DIR}/../../../../../tools/evaluation/check_board_clock.sh"

echo "[1/5] 预检 92 的完整 ImageNet val、标签和 Runtime."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_DATASET}" "${BOARD_RUN}" <<'BOARD_PREFLIGHT'
set -euo pipefail
readonly dataset="$1"
readonly run_dir="$2"
test -d "${dataset}/val"
test -s "${dataset}/val_labels_0based.txt"
test ! -e "${run_dir}/report/summary.json"
command -v python3
test -s /usr/lib/libneuronusdk_runtime.mtk.so.8 || ldconfig -p | grep -q libneuronusdk_runtime
BOARD_PREFLIGHT

echo "[2/5] 在 89 编译 DLA 并提取编译图量化元数据."
if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null)" != "true" ]]; then
    echo "[ERROR] 编译容器未运行: ${CONTAINER}." >&2
    exit 2
fi
docker exec "${CONTAINER}" bash "${SCRIPT_DIR}/build.sh"
docker exec "${CONTAINER}" mkdir -p "${TEMP_DIR}"
docker exec "${CONTAINER}" python3 "${SCRIPT_DIR}/extract_quantization.py" \
    --tflite "${MODEL_ROOT}/models/model_int8.tflite" \
    --output "${CONTAINER_TEMP}"
mkdir -p "${TEMP_DIR}"
docker cp "${CONTAINER}:${CONTAINER_TEMP}" "${TEMP_DIR}/quantization.json"
bash "${SCRIPT_DIR}/build_board_cpp.sh"

echo "[3/5] 部署本次专属模型和板端评测代码."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "mkdir -p '${BOARD_MODEL_DIR}' '${BOARD_RUN}/tools'"
scp "${SSH_OPTIONS[@]}" \
    "${MODEL_ROOT}/models/model_int8.dla" \
    "${TEMP_DIR}/quantization.json" \
    "${SCRIPT_DIR}/vit_board_eval" \
    "${BOARD_HOST}:${BOARD_MODEL_DIR}/"
scp "${SSH_OPTIONS[@]}" "${SCRIPT_DIR}/evaluate_full_accuracy.py" \
    "${BOARD_HOST}:${BOARD_RUN}/tools/"

echo "[4/5] 在 92 固化输入及数据集哈希."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_RUN}" "${BOARD_MODEL_DIR}" "${BOARD_DATASET}" <<'BOARD_HASH'
set -euo pipefail
readonly run_dir="$1"
readonly model_dir="$2"
readonly dataset="$3"
find "${dataset}/val" -maxdepth 1 -type f -name '*.JPEG' -print0 \
    | sort -z | xargs -0 sha256sum > "${run_dir}/dataset_images_sha256.txt"
test "$(wc -l < "${run_dir}/dataset_images_sha256.txt")" -eq 50000
sha256sum "${model_dir}/model_int8.dla" \
    "${model_dir}/quantization.json" \
    "${model_dir}/vit_board_eval" \
    "${run_dir}/tools/evaluate_full_accuracy.py" \
    "${dataset}/val_labels_0based.txt" \
    "${run_dir}/dataset_images_sha256.txt" \
    > "${run_dir}/run_inputs_sha256.txt"
BOARD_HASH

echo "[5/5] 在 92 由 C++ 完成 NPU 推理,Python 仅统计指标."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_RUN}" "${BOARD_MODEL_DIR}" "${BOARD_DATASET}" "${RUN_ID}" <<'BOARD_EVAL'
set -euo pipefail
readonly run_dir="$1"
readonly model_dir="$2"
readonly dataset="$3"
readonly run_id="$4"
"${model_dir}/vit_board_eval" \
    --model "${model_dir}/model_int8.dla" \
    --images "${dataset}/val" \
    --quantization "${model_dir}/quantization.json" \
    --predictions "${run_dir}/predictions.jsonl" \
    2>&1 | tee "${run_dir}/board_eval.log"
python3 "${run_dir}/tools/evaluate_full_accuracy.py" \
    --predictions "${run_dir}/predictions.jsonl" \
    --labels "${dataset}/val_labels_0based.txt" \
    --report "${run_dir}/report" --run-id "${run_id}"
cp "${run_dir}/dataset_images_sha256.txt" \
    "${run_dir}/run_inputs_sha256.txt" \
    "${run_dir}/board_eval.log" \
    "${run_dir}/report/"
sha256sum "${run_dir}/predictions.jsonl" \
    > "${run_dir}/report/raw_predictions_sha256.txt"
BOARD_EVAL
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- "${BOARD_RUN}/report" \
    < "${SCRIPT_DIR}/../../../../../tools/evaluation/capture_board_system.sh"
echo "[OK] ViT 全量精度报告: ${BOARD_RUN}/report"
echo "[NEXT] 手动上传结果到 Git 后,运行 EVAL_RUN_ID=${RUN_ID} bash ${SCRIPT_DIR}/cleanup_full_accuracy.sh"
