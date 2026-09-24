#!/usr/bin/env bash
# 在 89 交叉编译,在 92 完成 COCO-WholeBody 全量 NPU 精度与耗时测试.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_DATASET="${MTK_BOARD_DATASETS_ROOT:-/root/hailong.he/datasets}/coco/val2017"
readonly BOARD_MODEL_ROOT="${MTK_BOARD_OPEN_MODELS_ROOT:-/root/hailong.he/open_models}/rtmpose_body2d"
readonly BOARD_RUN="${BOARD_MODEL_ROOT}/eval/${RUN_ID}"
readonly BOARD_MODEL="${BOARD_MODEL_ROOT}/models/${RUN_ID}"
readonly BINARY="${SCRIPT_DIR}/inference_demo/rtmpose_board_eval"
readonly -a SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "[ERROR] EVAL_RUN_ID 非法." >&2
    exit 2
fi
test -s "${MODEL_ROOT}/models/model_int8.dla"
bash "${SCRIPT_DIR}/../../../../../tools/evaluation/check_board_clock.sh"

echo "[1/7] 在 92 检查全量 COCO 数据与 xtcocotools."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_DATASET}" "${BOARD_RUN}" <<'BOARD_PREFLIGHT'
set -euo pipefail
readonly dataset="$1"
readonly run_dir="$2"
test ! -e "${run_dir}"
test -s "${dataset}/annotations/coco_wholebody_val_v1.0.json"
test -s "${dataset}/person_detection_results/COCO_val2017_detections_AP_H_56_person.json"
test "$(find "${dataset}/images" -maxdepth 1 -type f -name '*.jpg' | wc -l)" -eq 5000
python3 -c 'import numpy, xtcocotools'
BOARD_PREFLIGHT

echo "[2/7] 在 89 交叉编译板端 C++ 程序."
bash "${SCRIPT_DIR}/build_board_cpp.sh"

echo "[3/7] 部署本次运行专属的程序、DLA 和指标代码."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" "mkdir -p '${BOARD_MODEL}' '${BOARD_RUN}/report'"
scp "${SSH_OPTIONS[@]}" \
    "${BINARY}" "${MODEL_ROOT}/models/model_int8.dla" \
    "${SCRIPT_DIR}/inference_demo/prepare_eval_manifest" \
    "${SCRIPT_DIR}/inference_demo/evaluate_coco_wholebody.py" \
    "${BOARD_HOST}:${BOARD_MODEL}/"

echo "[4/7] 在 92 生成并核对 104125 个检测框清单."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_MODEL}" "${BOARD_DATASET}" "${BOARD_RUN}" <<'BOARD_PREPARE'
set -euo pipefail
readonly model_dir="$1"
readonly dataset="$2"
readonly run_dir="$3"
"${model_dir}/prepare_eval_manifest" \
    --detections "${dataset}/person_detection_results/COCO_val2017_detections_AP_H_56_person.json" \
    --output "${run_dir}/person_detections.tsv"
test "$(($(wc -l < "${run_dir}/person_detections.tsv") - 1))" -eq 104125
find "${dataset}/images" -maxdepth 1 -type f -name '*.jpg' -print0 \
    | sort -z | xargs -0 sha256sum > "${run_dir}/dataset_images_sha256.txt"
test "$(wc -l < "${run_dir}/dataset_images_sha256.txt")" -eq 5000
{
    echo "captured_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    uname -a
    cat /etc/os-release
} > "${run_dir}/system.txt"
sha256sum "${model_dir}/model_int8.dla" \
    "${model_dir}/rtmpose_board_eval" \
    "${model_dir}/prepare_eval_manifest" \
    "${model_dir}/evaluate_coco_wholebody.py" \
    "${dataset}/annotations/coco_wholebody_val_v1.0.json" \
    "${dataset}/person_detection_results/COCO_val2017_detections_AP_H_56_person.json" \
    "${run_dir}/dataset_images_sha256.txt" \
    "${run_dir}/person_detections.tsv" > "${run_dir}/run_inputs_sha256.txt"
BOARD_PREPARE

echo "[5/7] 在 92 执行全部人体框的 C++ 预处理、NPU 推理与后处理."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_MODEL}" "${BOARD_DATASET}" "${BOARD_RUN}" <<'BOARD_INFER'
set -euo pipefail
readonly model_dir="$1"
readonly dataset="$2"
readonly run_dir="$3"
"${model_dir}/rtmpose_board_eval" \
    --model "${model_dir}/model_int8.dla" \
    --images "${dataset}/images" \
    --manifest "${run_dir}/person_detections.tsv" \
    --output-dir "${run_dir}" --warmup 20 --progress-interval 500 \
    2>&1 | tee "${run_dir}/board_eval.log"
test "$(wc -l < "${run_dir}/processed_ids.txt")" -eq 104125
BOARD_INFER

echo "[6/7] 在 92 计算 WholeBody AP/AR 并生成 Git 报告."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_MODEL}" "${BOARD_DATASET}" "${BOARD_RUN}" "${RUN_ID}" <<'BOARD_EVALUATE'
set -euo pipefail
readonly model_dir="$1"
readonly dataset="$2"
readonly run_dir="$3"
readonly run_id="$4"
python3 "${model_dir}/evaluate_coco_wholebody.py" \
    --annotations "${dataset}/annotations/coco_wholebody_val_v1.0.json" \
    --predictions "${run_dir}/predictions.jsonl" \
    --formatted "${run_dir}/wholebody_predictions.json" \
    --metrics "${run_dir}/coco_wholebody_metrics.json" \
    --summary-log "${run_dir}/coco_wholebody_summary.log" \
    2>&1 | tee "${run_dir}/cocoeval.log"
cd "${run_dir}"
sha256sum predictions.jsonl processed_ids.txt timing_summary.json \
    wholebody_predictions.json coco_wholebody_metrics.json \
    > run_outputs_sha256.txt
cp timing_summary.json coco_wholebody_metrics.json \
    coco_wholebody_summary.log board_eval.log cocoeval.log \
    run_inputs_sha256.txt run_outputs_sha256.txt \
    dataset_images_sha256.txt system.txt report/
python3 - "${run_dir}/coco_wholebody_metrics.json" \
    "${run_dir}/report/summary.json" "${run_id}" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if metrics.get("protocol", {}).get("detections_before_nms") != 104125:
    raise SystemExit("[ERROR] WholeBody 检测框覆盖不完整.")
summary = {
    "status": "complete",
    "model": "rtmpose_body2d",
    "run_id": sys.argv[3],
    "dataset": "coco_wholebody_val2017",
    "detections": 104125,
    "accuracy": metrics["metrics"],
}
Path(sys.argv[2]).write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
BOARD_EVALUATE

echo "[7/7] 完整报告保留在 92,等待用户手动上传和清理."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- "${BOARD_RUN}/report" \
    < "${SCRIPT_DIR}/../../../../../tools/evaluation/capture_board_system.sh"
echo "[OK] RTMPose 全量精度报告: ${BOARD_RUN}/report"
echo "[NEXT] 手动上传结果到 Git 后,运行 EVAL_RUN_ID=${RUN_ID} bash ${SCRIPT_DIR}/cleanup_full_accuracy.sh"
