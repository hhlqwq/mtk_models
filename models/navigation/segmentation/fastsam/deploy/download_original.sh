#!/usr/bin/env bash
# 兼容仓库命名规范,只校验离线权重,不执行任何下载.
set -euo pipefail
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly WEIGHTS="${FASTSAM_WEIGHTS:-${MODEL_ROOT}/original/FastSAM-s.pt}"
: "${FASTSAM_WEIGHTS_SHA256:?请设置已核实的 SHA-256}"
test -s "${WEIGHTS}"
printf '%s  %s\n' "${FASTSAM_WEIGHTS_SHA256}" "${WEIGHTS}" | sha256sum --check -
echo "[OK] 本地权重完整性校验通过.来源地址见 original/source_url.txt."
