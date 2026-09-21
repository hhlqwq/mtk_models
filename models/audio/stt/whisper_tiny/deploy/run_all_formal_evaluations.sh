#!/usr/bin/env bash

set -euo pipefail

readonly REPO_ROOT="${REPO_ROOT:-/data/users/hailong.he/github/mtk_models}"
readonly MODEL_ROOT="${REPO_ROOT}/models/audio/stt/whisper_tiny"
readonly DEPLOY_ROOT="${MODEL_ROOT}/deploy"
readonly EVAL_BASE="${EVAL_BASE:-${REPO_ROOT}/.eval/whisper_tiny}"
readonly BOARD_ROOT="${BOARD_ROOT:-/root/hailong.he/whisper_tiny}"
readonly DATE_TAG="${EVAL_DATE:-$(date +%Y%m%d)}"
readonly AISHELL_RUN_ID="${AISHELL_RUN_ID:-\
${DATE_TAG}_aishell1_test_fp16_v1}"
readonly LIBRISPEECH_RUN_ID="${LIBRISPEECH_RUN_ID:-\
${DATE_TAG}_librispeech_test_clean_fp16_v1}"
readonly COMBINED_RUN_ID="${COMBINED_RUN_ID:-${DATE_TAG}_bilingual_fp16_v1}"
readonly AISHELL_DATASET_ROOT="${AISHELL_DATASET_ROOT:-\
/data/users/hailong.he/nas_smb/Datasets/open_source/raw/Aishell/test}"
readonly AISHELL_ARCHIVE="${AISHELL_ARCHIVE:-\
/data/users/hailong.he/nas_smb/Datasets/open_source/raw/Aishell/data_aishell.tgz}"
readonly LIBRISPEECH_DATASET_ROOT="${LIBRISPEECH_DATASET_ROOT:-\
/data/users/hailong.he/nas_smb/Datasets/open_source/raw/LibriSpeech/test-clean/LibriSpeech/test-clean}"
readonly LIBRISPEECH_ARCHIVE="${LIBRISPEECH_ARCHIVE:-\
/data/users/hailong.he/nas_smb/Datasets/open_source/raw/LibriSpeech/test-clean.tar.gz}"
readonly REUSE_COMPLETE_RUNS="${REUSE_COMPLETE_RUNS:-1}"
readonly AISHELL_EVAL_ROOT="${EVAL_BASE}/${AISHELL_RUN_ID}"
readonly LIBRISPEECH_EVAL_ROOT="${EVAL_BASE}/${LIBRISPEECH_RUN_ID}"
readonly COMBINED_REPORT_ROOT="${EVAL_BASE}/${COMBINED_RUN_ID}/report"

# 检查数据目录和原始压缩包,避免运行中途才发现第二套数据缺失.
validate_input() {
    local dataset="$1"
    local dataset_root="$2"
    local archive="$3"
    if [[ ! -d "${dataset_root}" ]]; then
        echo "[ERROR] ${dataset} 数据目录不存在: ${dataset_root}" >&2
        return 2
    fi
    if [[ ! -f "${archive}" ]]; then
        echo "[ERROR] ${dataset} 原始压缩包不存在: ${archive}" >&2
        return 2
    fi
}

# 判断已有 summary.json 是否属于指定数据集且完整,用于安全断点复用.
summary_is_complete() {
    local summary_path="$1"
    local dataset="$2"
    python3 - "${summary_path}" "${dataset}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_dataset = sys.argv[2]
expected_metric = "cer" if expected_dataset == "aishell1" else "wer"
if not path.is_file():
    raise SystemExit(1)
summary = json.loads(path.read_text(encoding="utf-8"))
complete = (
    summary.get("dataset") == expected_dataset
    and summary.get("status") == "complete"
    and summary.get("successful_samples") == summary.get("expected_samples")
    and summary.get("failed_samples") == 0
    and summary.get("missing_samples") == 0
    and summary.get("accuracy", {}).get("metric") == expected_metric
    and summary.get("framework_accuracy", {}).get("metric") == expected_metric
    and summary.get("framework_parity") is not None
    and summary.get("performance", {}).get("overall") is not None
)
raise SystemExit(0 if complete else 1)
PY
}

# 执行或复用一个数据集的正式 Run.
run_dataset() {
    local index="$1"
    local dataset="$2"
    local run_id="$3"
    local dataset_root="$4"
    local archive="$5"
    local eval_root="$6"
    local board_eval_root="${BOARD_ROOT}/eval/${run_id}"
    local summary_path="${eval_root}/report/summary.json"

    echo "[SUITE ${index}/2] ${dataset}, Run ID: ${run_id}."
    if [[ "${REUSE_COMPLETE_RUNS}" == "1" ]] && \
            summary_is_complete "${summary_path}" "${dataset}"; then
        echo "[SKIP] 复用已完成正式 Run: ${summary_path}"
        return 0
    fi

    DATASET="${dataset}" \
    RUN_ID="${run_id}" \
    DATASET_ROOT="${dataset_root}" \
    ARCHIVE="${archive}" \
    EVAL_ROOT="${eval_root}" \
    BOARD_EVAL_ROOT="${board_eval_root}" \
    SKIP_BOARD_BUILD=1 \
        bash "${DEPLOY_ROOT}/run_formal_evaluation.sh"
}

if [[ "${AISHELL_RUN_ID}" == "${LIBRISPEECH_RUN_ID}" ]]; then
    echo "[ERROR] 两套数据集不能使用相同 RUN_ID." >&2
    exit 2
fi
if [[ "${REUSE_COMPLETE_RUNS}" != "0" && "${REUSE_COMPLETE_RUNS}" != "1" ]]; then
    echo "[ERROR] REUSE_COMPLETE_RUNS 只能为 0 或 1." >&2
    exit 2
fi

validate_input "aishell1" "${AISHELL_DATASET_ROOT}" "${AISHELL_ARCHIVE}"
validate_input "librispeech" \
    "${LIBRISPEECH_DATASET_ROOT}" "${LIBRISPEECH_ARCHIVE}"

echo "[PIPELINE] 交叉编译一次板端批量评测程序,供两套数据复用."
bash "${DEPLOY_ROOT}/build_board_cpp.sh"

run_dataset 1 "aishell1" "${AISHELL_RUN_ID}" \
    "${AISHELL_DATASET_ROOT}" "${AISHELL_ARCHIVE}" "${AISHELL_EVAL_ROOT}"
run_dataset 2 "librispeech" "${LIBRISPEECH_RUN_ID}" \
    "${LIBRISPEECH_DATASET_ROOT}" "${LIBRISPEECH_ARCHIVE}" \
    "${LIBRISPEECH_EVAL_ROOT}"

python3 "${DEPLOY_ROOT}/summarize_formal_evaluations.py" \
    --aishell-summary "${AISHELL_EVAL_ROOT}/report/summary.json" \
    --aishell-run-id "${AISHELL_RUN_ID}" \
    --librispeech-summary "${LIBRISPEECH_EVAL_ROOT}/report/summary.json" \
    --librispeech-run-id "${LIBRISPEECH_RUN_ID}" \
    --output-dir "${COMBINED_REPORT_ROOT}"

echo "[OK] AISHELL-1 与 LibriSpeech 两套正式评测全部完成."
echo "[OK] 联合报告: ${COMBINED_REPORT_ROOT}/report.md"
echo "[OK] 联合汇总: ${COMBINED_REPORT_ROOT}/summary.json"
