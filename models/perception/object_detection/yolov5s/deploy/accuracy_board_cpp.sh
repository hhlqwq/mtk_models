#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_MODEL_ROOT="${MTK_BOARD_OPEN_MODELS_ROOT:-/root/hailong.he/open_models}/yolov5s"
readonly BOARD_DATASET="${MTK_BOARD_DATASETS_ROOT:-/root/hailong.he/datasets}/coco/val2017"
readonly RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
readonly BOARD_OUTPUT="${BOARD_MODEL_ROOT}/eval/${RUN_ID}"
readonly BOARD_MODEL_DIR="${BOARD_MODEL_ROOT}/models/${RUN_ID}"
readonly BOARD_RESULT="${BOARD_MODEL_ROOT}/results/${RUN_ID}"
readonly BINARY="${SCRIPT_DIR}/inference_demo/yolov5s_board_eval"
readonly EVALUATOR="${SCRIPT_DIR}/inference_demo/evaluate_coco.py"
readonly ANNOTATIONS="${BOARD_DATASET}/annotations/instances_val2017.json"
readonly SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "[ERROR] EVAL_RUN_ID 只能包含字母、数字、点、下划线和连字符." >&2
    exit 1
fi
bash "${SCRIPT_DIR}/../../../../../tools/evaluation/check_board_clock.sh"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "python3 -c 'import pycocotools'"

# 采集板端系统与 CPU 调频状态,并写入指定证据文件.
capture_board_state() {
    local output_path="$1"
    ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- "${output_path}" <<'BOARD_STATE'
set -euo pipefail
readonly output_path="$1"
{
    echo "captured_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uname=$(uname -a)"
    if [[ -r /etc/os-release ]]; then
        echo "[os_release]"
        cat /etc/os-release
    fi
    echo "[cpu_frequency]"
    shopt -s nullglob
    policies=(/sys/devices/system/cpu/cpufreq/policy*)
    if (( ${#policies[@]} == 0 )); then
        echo "unavailable"
    else
        for policy in "${policies[@]}"; do
            echo "policy=$(basename "${policy}")"
            for item in scaling_governor scaling_cur_freq scaling_min_freq \
                        scaling_max_freq cpuinfo_min_freq cpuinfo_max_freq; do
                if [[ -r "${policy}/${item}" ]]; then
                    echo "${item}=$(<"${policy}/${item}")"
                fi
            done
        done
    fi
} > "${output_path}"
BOARD_STATE
}

echo "[1/8] 交叉编译板端 C++ 评测程序."
bash "${SCRIPT_DIR}/build_board_cpp.sh"

echo "[2/8] 检查板端数据集及全新运行目录."
image_count="$(ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "find '${BOARD_DATASET}/images' -maxdepth 1 -type f -name '*.jpg' | wc -l")"
test "${image_count}" -eq 5000
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" "test -f '${ANNOTATIONS}'"
if ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" "test -e '${BOARD_OUTPUT}'"; then
    echo "[ERROR] 板端运行目录已存在,请使用新的 EVAL_RUN_ID: ${BOARD_OUTPUT}" >&2
    exit 1
fi

echo "[3/8] 部署 C++ 程序、DLA 和板端指标脚本."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "mkdir -p '${BOARD_MODEL_DIR}' '${BOARD_OUTPUT}'"
scp "${SSH_OPTIONS[@]}" "${BINARY}" "${MODEL_ROOT}/models/model_int8.dla" \
    "${EVALUATOR}" "${BOARD_HOST}:${BOARD_MODEL_DIR}/"

echo "[4/8] 固化运行输入、数据集和板端环境证据."
readonly GIT_COMMIT="$(git -C "${MODEL_ROOT}" rev-parse HEAD)"
readonly SOURCE_SHA256="$(sha256sum \
    "${SCRIPT_DIR}/inference_demo/yolov5s_board_eval.cpp" | awk '{print $1}')"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_MODEL_DIR}" "${BOARD_DATASET}" "${BOARD_OUTPUT}" \
    "${RUN_ID}" "${GIT_COMMIT}" "${SOURCE_SHA256}" <<'BOARD_MANIFEST'
set -euo pipefail
readonly model_dir="$1"
readonly dataset_dir="$2"
readonly output_dir="$3"
readonly run_id="$4"
readonly git_commit="$5"
readonly source_sha256="$6"
find "${dataset_dir}/images" -maxdepth 1 -type f -name '*.jpg' -print0 \
    | sort -z | xargs -0 sha256sum > "${output_dir}/dataset_images_sha256.txt"
{
    echo "schema_version=1"
    echo "run_id=${run_id}"
    echo "git_commit=${git_commit}"
    echo "cpp_source_sha256=${source_sha256}"
    echo "board_binary_sha256=$(sha256sum "${model_dir}/yolov5s_board_eval" | awk '{print $1}')"
    echo "dla_sha256=$(sha256sum "${model_dir}/model_int8.dla" | awk '{print $1}')"
    echo "evaluator_sha256=$(sha256sum "${model_dir}/evaluate_coco.py" | awk '{print $1}')"
    echo "annotations_sha256=$(sha256sum "${dataset_dir}/annotations/instances_val2017.json" | awk '{print $1}')"
    echo "dataset_images_manifest_sha256=$(sha256sum "${output_dir}/dataset_images_sha256.txt" | awk '{print $1}')"
    echo "dataset_image_count=$(wc -l < "${output_dir}/dataset_images_sha256.txt")"
    echo "input_shape=1,3,640,640"
    echo "warmup=20"
    echo "confidence=0.001"
    echo "iou=0.6"
    echo "max_detections=300"
} > "${output_dir}/run_inputs_manifest.txt"
BOARD_MANIFEST
capture_board_state "${BOARD_OUTPUT}/system_before.txt"

echo "[5/8] 在板端执行 5000 张 C++ 预处理、NPU 推理和后处理."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "'${BOARD_MODEL_DIR}/yolov5s_board_eval' \
        --model '${BOARD_MODEL_DIR}/model_int8.dla' \
        --images '${BOARD_DATASET}/images' \
        --output-dir '${BOARD_OUTPUT}' \
        --warmup 20 --progress-interval 50 \
        2>&1 | tee '${BOARD_OUTPUT}/board_eval.log'"

echo "[6/8] 在板端使用 pycocotools 计算 COCO bbox 指标."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "python3 '${BOARD_MODEL_DIR}/evaluate_coco.py' \
        --annotations '${ANNOTATIONS}' \
        --predictions '${BOARD_OUTPUT}/predictions.json' \
        --processed-ids '${BOARD_OUTPUT}/processed_ids.txt' \
        --metrics '${BOARD_OUTPUT}/coco_metrics.json' \
        --summary-log '${BOARD_OUTPUT}/coco_summary.log' \
        2>&1 | tee '${BOARD_OUTPUT}/cocoeval.log'"

echo "[7/8] 固化输出哈希和运行后系统状态."
capture_board_state "${BOARD_OUTPUT}/system_after.txt"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "cd '${BOARD_OUTPUT}' && sha256sum predictions.json processed_ids.txt \
        timings.csv timing_summary_current_run.json coco_metrics.json \
        coco_summary.log board_eval.log cocoeval.log system_before.txt \
        system_after.txt run_inputs_manifest.txt dataset_images_sha256.txt \
        > run_outputs_sha256.txt"

echo "[8/8] 在 92 整理轻量报告,本次不清理."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_OUTPUT}" "${RUN_ID}" <<'BOARD_REPORT'
set -euo pipefail
readonly run_dir="$1"
readonly run_id="$2"
mkdir -p "${run_dir}/report"
cp "${run_dir}/coco_metrics.json" \
    "${run_dir}/coco_summary.log" \
    "${run_dir}/board_eval.log" \
    "${run_dir}/cocoeval.log" \
    "${run_dir}/timing_summary_current_run.json" \
    "${run_dir}/timings.csv" \
    "${run_dir}/processed_ids.txt" \
    "${run_dir}/run_inputs_manifest.txt" \
    "${run_dir}/run_outputs_sha256.txt" \
    "${run_dir}/dataset_images_sha256.txt" \
    "${run_dir}/system_before.txt" \
    "${run_dir}/system_after.txt" \
    "${run_dir}/report/"
python3 - "${run_dir}/report/coco_metrics.json" \
    "${run_dir}/report/summary.json" "${run_id}" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if metrics.get("images") != 5000:
    raise SystemExit("[ERROR] COCO 全量图片数不等于 5000.")
summary = {
    "status": "complete",
    "model": "yolov5s",
    "run_id": sys.argv[3],
    "dataset": "coco_val2017",
    "images": metrics["images"],
    "accuracy": metrics["metrics"],
}
Path(sys.argv[2]).write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
BOARD_REPORT
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- "${BOARD_OUTPUT}/report" \
    < "${SCRIPT_DIR}/../../../../../tools/evaluation/capture_board_system.sh"
echo "[OK] YOLOv5s 板端全量精度报告: ${BOARD_OUTPUT}/report"
echo "[NEXT] 手动上传结果到 Git 后,运行 EVAL_RUN_ID=${RUN_ID} bash ${SCRIPT_DIR}/cleanup_full_accuracy.sh"
