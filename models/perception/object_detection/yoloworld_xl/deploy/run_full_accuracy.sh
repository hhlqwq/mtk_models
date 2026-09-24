#!/usr/bin/env bash
# 在 89 检查兼容 ONNX,在 92 完成 COCO val2017 全量 Neuron EP mAP 测试.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_DATASET="${MTK_BOARD_DATASETS_ROOT:-/root/hailong.he/datasets}/coco/val2017"
readonly BOARD_MODEL_ROOT="${MTK_BOARD_OPEN_MODELS_ROOT:-/root/hailong.he/open_models}/yoloworld_xl"
readonly BOARD_RUN="${BOARD_MODEL_ROOT}/eval/${RUN_ID}"
readonly BOARD_MODEL_DIR="${BOARD_MODEL_ROOT}/models/${RUN_ID}"
readonly CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
readonly -a SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "[ERROR] EVAL_RUN_ID 非法." >&2
    exit 2
fi
bash "${SCRIPT_DIR}/../../../../../tools/evaluation/check_board_clock.sh"

echo "[1/5] 检查 92 的 COCO 数据及指标依赖."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_DATASET}" "${BOARD_RUN}" <<'BOARD_PREFLIGHT'
set -euo pipefail
readonly dataset="$1"
readonly run_dir="$2"
test ! -e "${run_dir}"
test -s "${dataset}/annotations/instances_val2017.json"
test "$(find "${dataset}/images" -maxdepth 1 -type f -name '*.jpg' | wc -l)" -eq 5000
python3 -c 'import pycocotools'
test -s /usr/lib/libonnxruntime.so.1.20.2
BOARD_PREFLIGHT

echo "[2/5] 在 89 做来源校验、兼容图准备和 C++ 交叉编译."
bash "${SCRIPT_DIR}/download_original.sh"
test "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null)" = true
docker exec "${CONTAINER}" python3 "${SCRIPT_DIR}/prepare_onnx.py" \
    --input "${MODEL_ROOT}/models/model_fp32.onnx" \
    --output "${MODEL_ROOT}/models/model_fp32_opset13.onnx" \
    --raw-output "${MODEL_ROOT}/models/model_fp32_raw.onnx"
test -s "${MODEL_ROOT}/models/model_fp32_opset13.onnx"
bash "${SCRIPT_DIR}/build_board_cpp.sh"

echo "[3/5] 部署本次专属模型和板端评测代码."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "mkdir -p '${BOARD_MODEL_DIR}' '${BOARD_RUN}/tools' '${BOARD_RUN}/report'"
scp "${SSH_OPTIONS[@]}" \
    "${MODEL_ROOT}/models/model_fp32_opset13.onnx" \
    "${SCRIPT_DIR}/inference_demo/yoloworld_board_eval" \
    "${BOARD_HOST}:${BOARD_MODEL_DIR}/"
scp "${SSH_OPTIONS[@]}" \
    "${SCRIPT_DIR}/inference_demo/evaluate_full_coco.py" \
    "${BOARD_HOST}:${BOARD_RUN}/tools/"

echo "[4/5] 在 92 完成 5000 张 Neuron EP 推理及 COCO mAP."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_RUN}" "${BOARD_MODEL_DIR}" "${BOARD_DATASET}" "${RUN_ID}" <<'BOARD_EVAL'
set -euo pipefail
readonly run_dir="$1"
readonly model_dir="$2"
readonly dataset="$3"
readonly run_id="$4"
"${model_dir}/yoloworld_board_eval" \
    --model "${model_dir}/model_fp32_opset13.onnx" \
    --images "${dataset}/images" \
    --output-dir "${run_dir}/raw" \
    2>&1 | tee "${run_dir}/board_eval.log"
python3 "${run_dir}/tools/evaluate_full_coco.py" \
    --annotations "${dataset}/annotations/instances_val2017.json" \
    --results "${run_dir}/raw/results.jsonl" \
    --profile "$(cat "${run_dir}/raw/profile_path.txt")" \
    --output-dir "${run_dir}/report" --run-id "${run_id}" \
    2>&1 | tee "${run_dir}/cocoeval.log"
BOARD_EVAL

echo "[5/5] 在 92 固化模型、数据及原始预测哈希."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_RUN}" "${BOARD_MODEL_DIR}" "${BOARD_DATASET}" <<'BOARD_HASH'
set -euo pipefail
readonly run_dir="$1"
readonly model_dir="$2"
readonly dataset="$3"
find "${dataset}/images" -maxdepth 1 -type f -name '*.jpg' -print0 \
    | sort -z | xargs -0 sha256sum > "${run_dir}/report/dataset_images_sha256.txt"
test "$(wc -l < "${run_dir}/report/dataset_images_sha256.txt")" -eq 5000
sha256sum "${model_dir}/model_fp32_opset13.onnx" \
    "${model_dir}/yoloworld_board_eval" \
    "${run_dir}/tools/evaluate_full_coco.py" \
    "${run_dir}/report/dataset_images_sha256.txt" \
    "${dataset}/annotations/instances_val2017.json" \
    > "${run_dir}/report/run_inputs_sha256.txt"
profile_file="$(cat "${run_dir}/raw/profile_path.txt")"
test -s "${profile_file}"
sha256sum "${run_dir}/raw/results.jsonl" \
    "${run_dir}/raw/profile_path.txt" \
    "${profile_file}" \
    "${run_dir}/coco_predictions.json" \
    > "${run_dir}/report/raw_outputs_sha256.txt"
cp "${run_dir}/board_eval.log" "${run_dir}/cocoeval.log" \
    "${run_dir}/report/"
BOARD_HASH
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- "${BOARD_RUN}/report" \
    < "${SCRIPT_DIR}/../../../../../tools/evaluation/capture_board_system.sh"
echo "[OK] YOLO-World XL 板端全量精度报告: ${BOARD_RUN}/report"
echo "[NEXT] 手动上传结果到 Git 后,运行 EVAL_RUN_ID=${RUN_ID} bash ${SCRIPT_DIR}/cleanup_full_accuracy.sh"
