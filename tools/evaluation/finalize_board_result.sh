#!/usr/bin/env bash
# 在板端保留精度报告,校验后清理本次大体积缓存.

set -euo pipefail

readonly MODEL_KEY="${MODEL_KEY:?请设置模型目录名}"
readonly RUN_ID="${RUN_ID:?请设置本次运行 ID}"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_ROOT="${MTK_BOARD_OPEN_MODELS_ROOT:-/root/hailong.he/open_models}"
readonly DATASETS_ROOT="${MTK_BOARD_DATASETS_ROOT:-/root/hailong.he/datasets}"
readonly BOARD_RUN="${BOARD_ROOT}/${MODEL_KEY}/eval/${RUN_ID}"
readonly BOARD_MODEL="${BOARD_ROOT}/${MODEL_KEY}/models/${RUN_ID}"
readonly BOARD_CACHE="${DATASETS_ROOT}/${MODEL_KEY}/${RUN_ID}"
readonly BOARD_RESULT="${BOARD_ROOT}/${MODEL_KEY}/results/${RUN_ID}"
readonly MAX_REPORT_BYTES="${MAX_REPORT_BYTES:-20971520}"
readonly -a SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

if [[ "${CONFIRM_RESULTS_UPLOADED:-0}" != "1" ]]; then
    echo "[ERROR] 请先手动上传结果,确认后设置 CONFIRM_RESULTS_UPLOADED=1." >&2
    exit 2
fi
if [[ "${BOARD_ROOT}" != "/root/hailong.he/open_models" ]] ||
        [[ "${DATASETS_ROOT}" != "/root/hailong.he/datasets" ]]; then
    echo "[ERROR] 清理脚本只允许固定板端根目录." >&2
    exit 2
fi
if [[ ! "${MODEL_KEY}" =~ ^[a-z][a-z0-9_]*$ ]] ||
        [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
        [[ ! "${MAX_REPORT_BYTES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] 模型、运行 ID 或报告上限格式不正确." >&2
    exit 2
fi
echo "[FINALIZE 1/2] 检查 92 的完整报告和体积."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_RUN}" "${MODEL_KEY}" "${RUN_ID}" "${MAX_REPORT_BYTES}" <<'BOARD_CHECK'
set -euo pipefail
readonly run_dir="$1"
readonly model_key="$2"
readonly run_id="$3"
readonly max_bytes="$4"
readonly report_dir="${run_dir}/report"
test -s "${report_dir}/summary.json"
test ! -L "${report_dir}"
python3 - "${report_dir}/summary.json" "${model_key}" "${run_id}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (summary.get("status") != "complete" or
        summary.get("model") != sys.argv[2] or
        summary.get("run_id") != sys.argv[3]):
    raise SystemExit("[ERROR] 报告状态或运行身份不匹配.")
PY
size="$(du -sb "${report_dir}" | cut -f 1)"
if (( size > max_bytes )); then
    echo "[ERROR] 报告超出 Git 上限: ${size} > ${max_bytes} 字节." >&2
    exit 2
fi
cd "${report_dir}"
find . -type l -print -quit | grep -q . && {
    echo "[ERROR] 报告包含符号链接." >&2
    exit 2
}
find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
BOARD_CHECK

echo "[FINALIZE 2/2] 板端保留报告并清理本次大文件."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_RUN}" "${BOARD_MODEL}" "${BOARD_CACHE}" "${BOARD_RESULT}" \
    "${BOARD_ROOT}/${MODEL_KEY}/eval" \
    "${BOARD_ROOT}/${MODEL_KEY}/models" \
    "${DATASETS_ROOT}/${MODEL_KEY}" \
    "${BOARD_ROOT}/${MODEL_KEY}/results" <<'BOARD_CLEAN'
set -euo pipefail
readonly run_dir="$1"
readonly model_dir="$2"
readonly cache_dir="$3"
readonly result_dir="$4"
readonly run_parent="$5"
readonly model_parent="$6"
readonly cache_parent="$7"
readonly result_parent="$8"
[[ -d "${run_dir}" && "$(dirname "${run_dir}")" == "${run_parent}" ]]
test ! -L "${run_dir}"
test ! -L "${run_dir}/report"
[[ "$(dirname "${model_dir}")" == "${model_parent}" ]]
[[ "$(dirname "${cache_dir}")" == "${cache_parent}" ]]
[[ "$(dirname "${result_dir}")" == "${result_parent}" ]]
test ! -e "${result_dir}"
(cd "${run_dir}/report" && sha256sum --check --status SHA256SUMS)
mkdir -p "${result_parent}"
mv -- "${run_dir}/report" "${result_dir}"
(cd "${result_dir}" && sha256sum --check --status SHA256SUMS)
rm -rf -- "${run_dir}"
if [[ -d "${model_dir}" ]]; then
    rm -rf -- "${model_dir}"
fi
if [[ -d "${cache_dir}" ]]; then
    rm -rf -- "${cache_dir}"
fi
BOARD_CLEAN
echo "[OK] 板端报告: ${BOARD_RESULT}."
echo "[WARN] 报告仅在 92 上.刷机前请手动取走并上传 Git."
