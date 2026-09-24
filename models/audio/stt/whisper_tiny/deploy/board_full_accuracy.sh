#!/usr/bin/env bash
# 在 Genio 720 板端以 C++ 准备 LibriSpeech 并推理,Python 仅计算 WER.

set -euo pipefail

readonly RUN_ID="${RUN_ID:?请设置本次运行 ID}"
readonly BOARD_RUN="${BOARD_RUN:?请设置板端运行目录}"
readonly BOARD_DATASETS_ROOT="${MTK_BOARD_DATASETS_ROOT:-/root/hailong.he/datasets}"
readonly LIBRISPEECH_ROOT="${LIBRISPEECH_ROOT:-${BOARD_DATASETS_ROOT}/librispeech/test-clean}"
readonly BOARD_CACHE="${BOARD_DATASETS_ROOT}/whisper_tiny/${RUN_ID}"
readonly MODEL_DIR="${MTK_BOARD_OPEN_MODELS_ROOT:-/root/hailong.he/open_models}/whisper_tiny/models/${RUN_ID}"
readonly TOOL_DIR="${BOARD_RUN}/tools"
readonly REPORT_DIR="${BOARD_RUN}/report"

# 执行单数据集全量流程,板端非指标计算均由 C++ 完成.
run_dataset() {
    local dataset="$1"
    local root="$2"
    local expected="$3"
    local cache_dir="${BOARD_CACHE}/${dataset}"
    local output_dir="${BOARD_RUN}/${dataset}"
    local sample_count

    echo "[DATASET] ${dataset}: C++ 构建清单、解码音频并生成 Mel."
    "${MODEL_DIR}/prepare_board_audio" \
        --dataset-root "${root}" \
        --filters "${MODEL_DIR}/mel_filters_f32.bin" \
        --output-dir "${cache_dir}"
    sample_count="$(wc -l < "${cache_dir}/source_manifest.jsonl")"
    if [[ "${sample_count}" -ne "${expected}" ]]; then
        echo "[ERROR] ${dataset} 样本数 ${sample_count},预期 ${expected}." >&2
        return 2
    fi

    echo "[DATASET] ${dataset}: 板端双 DLA 常驻推理."
    mkdir -p "${output_dir}/logs" "${REPORT_DIR}"
    /usr/bin/time -v -o "${output_dir}/logs/resource_usage.txt" \
        "${MODEL_DIR}/whisper_board_eval" \
        "${MODEL_DIR}/encoder_fp32.dla" \
        "${MODEL_DIR}/decoder_step_fp32.dla" \
        "${cache_dir}/board_manifest.tsv" \
        "${MODEL_DIR}/decode_config.txt" \
        "${output_dir}/board_predictions.jsonl" \
        > "${output_dir}/logs/board.log" 2>&1

    echo "[DATASET] ${dataset}: 板端计算 WER 与耗时."
    python3 "${TOOL_DIR}/evaluate_accuracy.py" \
        --dataset "${dataset}" \
        --source-manifest "${cache_dir}/source_manifest.jsonl" \
        --board-predictions "${output_dir}/board_predictions.jsonl" \
        --preprocess-metrics "${cache_dir}/preprocess_metrics.jsonl" \
        --output-dir "${REPORT_DIR}"
    cp "${output_dir}/logs/resource_usage.txt" \
        "${output_dir}/logs/board.log" \
        "${REPORT_DIR}/"
}

test ! -e "${REPORT_DIR}/summary.json"
test -d "${LIBRISPEECH_ROOT}"
test -s "${MODEL_DIR}/encoder_fp32.dla"
test -s "${MODEL_DIR}/decoder_step_fp32.dla"
test -s "${MODEL_DIR}/prepare_board_audio"
test -s "${MODEL_DIR}/mel_filters_f32.bin"
test -s "${MODEL_DIR}/decode_config.txt"
python3 -c 'from whisper.normalizers import EnglishTextNormalizer; from whisper.tokenizer import get_tokenizer'

echo "[SUITE] LibriSpeech test-clean: 2620 条."
run_dataset librispeech "${LIBRISPEECH_ROOT}" 2620
echo "[SUITE] 固化单数据集报告身份."
python3 - "${REPORT_DIR}/summary.json" "${RUN_ID}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
summary = json.loads(path.read_text(encoding="utf-8"))
if (summary.get("status") != "complete" or
        summary.get("dataset") != "librispeech" or
        summary.get("expected_samples") != 2620 or
        summary.get("successful_samples") != 2620):
    raise SystemExit("[ERROR] LibriSpeech 全量报告不完整.")
summary["model"] = "whisper_tiny"
summary["run_id"] = sys.argv[2]
path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
PY
sha256sum "${MODEL_DIR}/encoder_fp32.dla" \
    "${MODEL_DIR}/decoder_step_fp32.dla" \
    "${MODEL_DIR}/whisper_board_eval" \
    "${MODEL_DIR}/prepare_board_audio" \
    "${MODEL_DIR}/mel_filters_f32.bin" \
    "${MODEL_DIR}/decode_config.txt" \
    > "${REPORT_DIR}/model_sha256.txt"
find "${LIBRISPEECH_ROOT}" -type f -name '*.flac' -print0 \
    | sort -z | xargs -0 sha256sum > "${REPORT_DIR}/dataset_audio_sha256.txt"
test "$(wc -l < "${REPORT_DIR}/dataset_audio_sha256.txt")" -eq 2620
sha256sum "${BOARD_CACHE}/librispeech/source_manifest.jsonl" \
    "${BOARD_CACHE}/librispeech/board_manifest.tsv" \
    "${REPORT_DIR}/dataset_audio_sha256.txt" \
    > "${REPORT_DIR}/run_inputs_sha256.txt"
echo "[OK] 板端 LibriSpeech 全量报告: ${REPORT_DIR}"
