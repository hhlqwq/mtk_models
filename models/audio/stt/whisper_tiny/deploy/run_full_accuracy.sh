#!/usr/bin/env bash
# 在 89 编译 Whisper-Tiny,在 92 完成 LibriSpeech 正式精度测试.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_MODEL_ROOT="${MTK_BOARD_OPEN_MODELS_ROOT:-/root/hailong.he/open_models}/whisper_tiny"
readonly BOARD_RUN="${BOARD_MODEL_ROOT}/eval/${RUN_ID}"
readonly BOARD_MODEL_DIR="${BOARD_MODEL_ROOT}/models/${RUN_ID}"
readonly BOARD_DATASETS_ROOT="${MTK_BOARD_DATASETS_ROOT:-/root/hailong.he/datasets}"
readonly LIBRISPEECH_ROOT="${LIBRISPEECH_ROOT:-${BOARD_DATASETS_ROOT}/librispeech/test-clean}"
readonly ASSETS_DIR="/tmp/hailongcodex/$(date +%Y%m%d)/whisper_${RUN_ID}"
readonly CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
readonly -a SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "[ERROR] EVAL_RUN_ID 只能包含字母、数字、点、下划线和连字符." >&2
    exit 2
fi
if [[ ! "${LIBRISPEECH_ROOT}" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    echo "[ERROR] 板端数据目录必须是无空格的绝对路径: ${LIBRISPEECH_ROOT}." >&2
    exit 2
fi
test -s "${MODEL_ROOT}/models/encoder_fp32.dla"
test -s "${MODEL_ROOT}/models/decoder_step_fp32.dla"
bash "${SCRIPT_DIR}/../../../../../tools/evaluation/check_board_clock.sh"

echo "[1/4] 检查 92 数据、依赖和本次结果目录."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${LIBRISPEECH_ROOT}" "${BOARD_RUN}" <<'BOARD_PREFLIGHT'
set -euo pipefail
readonly librispeech_root="$1"
readonly run_dir="$2"
test -d "${librispeech_root}"
test ! -e "${run_dir}/report/summary.json"
command -v ffmpeg
command -v /usr/bin/time
python3 -c 'from whisper.normalizers import EnglishTextNormalizer; from whisper.tokenizer import get_tokenizer'
BOARD_PREFLIGHT

echo "[2/4] 在 89 编译 C++ 程序并导出静态滤波器和解码规则."
bash "${SCRIPT_DIR}/build_board_cpp.sh"
bash "${SCRIPT_DIR}/build_board_audio_cpp.sh"
test "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null)" = true
docker exec "${CONTAINER}" mkdir -p "${ASSETS_DIR}"
docker exec "${CONTAINER}" python3 "${SCRIPT_DIR}/export_board_assets.py" \
    --output-dir "${ASSETS_DIR}"
mkdir -p "${ASSETS_DIR}"
docker cp "${CONTAINER}:${ASSETS_DIR}/mel_filters_f32.bin" "${ASSETS_DIR}/mel_filters_f32.bin"
docker cp "${CONTAINER}:${ASSETS_DIR}/decode_config.txt" "${ASSETS_DIR}/decode_config.txt"
test "$(stat -c %s "${ASSETS_DIR}/mel_filters_f32.bin")" -eq 64320

echo "[3/4] 只部署代码、权重和 DLA,不在 89 处理测试数据."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "mkdir -p '${BOARD_MODEL_DIR}' '${BOARD_RUN}/tools'"
scp "${SSH_OPTIONS[@]}" \
    "${MODEL_ROOT}/models/encoder_fp32.dla" \
    "${MODEL_ROOT}/models/decoder_step_fp32.dla" \
    "${SCRIPT_DIR}/inference_demo/whisper_board_eval" \
    "${SCRIPT_DIR}/prepare_board_audio" \
    "${ASSETS_DIR}/mel_filters_f32.bin" \
    "${ASSETS_DIR}/decode_config.txt" \
    "${BOARD_HOST}:${BOARD_MODEL_DIR}/"
scp "${SSH_OPTIONS[@]}" \
    "${SCRIPT_DIR}/evaluate_accuracy.py" \
    "${SCRIPT_DIR}/board_full_accuracy.sh" \
    "${BOARD_HOST}:${BOARD_RUN}/tools/"

echo "[4/4] 在 92 执行 LibriSpeech 全量测试."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${RUN_ID}" "${BOARD_RUN}" "${LIBRISPEECH_ROOT}" <<'BOARD_EXEC'
set -euo pipefail
RUN_ID="$1" BOARD_RUN="$2" LIBRISPEECH_ROOT="$3" \
    bash "$2/tools/board_full_accuracy.sh"
BOARD_EXEC
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- "${BOARD_RUN}/report" \
    < "${SCRIPT_DIR}/../../../../../tools/evaluation/capture_board_system.sh"
echo "[OK] Whisper-Tiny LibriSpeech 板端报告: ${BOARD_RUN}/report"
echo "[NEXT] 手动上传结果到 Git 后,运行 EVAL_RUN_ID=${RUN_ID} bash ${SCRIPT_DIR}/cleanup_full_accuracy.sh"
