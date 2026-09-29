#!/usr/bin/env bash
# 在 89 部署模型,在 92 完成 LFW 十折全量人脸验证.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly RUN_ID="${EVAL_RUN_ID:?请设置本次运行 ID}"
readonly RESUME="${EVAL_RESUME:-0}"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_ROOT="/root/hailong.he/open_models/mobilefacenet"
readonly BOARD_MODEL_DIR="${BOARD_ROOT}/models/${RUN_ID}"
readonly BOARD_RUN="${BOARD_ROOT}/eval/${RUN_ID}"
readonly BOARD_DATASET="/root/hailong.he/datasets/lfw"
readonly SMOKE_METADATA="${MODEL_ROOT}/examples/output/runs/20260924T020219Z/metadata.json"
readonly -a SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "[ERROR] EVAL_RUN_ID 非法." >&2
    exit 2
fi
if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
    echo "[ERROR] EVAL_RESUME 只能为 0 或 1." >&2
    exit 2
fi
test -s "${MODEL_ROOT}/models/model_int8.dla"
test -s "${SMOKE_METADATA}"
bash "${SCRIPT_DIR}/../../../../../tools/evaluation/check_board_clock.sh"

echo "[1/3] 核对 92 的 LFW 十折数据与 Python 依赖."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_DATASET}" "${BOARD_RUN}" "${RESUME}" <<'BOARD_PREFLIGHT'
set -euo pipefail
test -s "$1/pairs.csv"
test -s "$1/source_manifest.json"
test -d "$1/images"
python3 -c 'import cv2, numpy'
if [[ "$3" == "1" ]]; then test -d "$2"; else test ! -e "$2"; fi
BOARD_PREFLIGHT

if [[ "${RESUME}" == "0" ]]; then
    echo "[2/3] 部署本次专属 DLA、量化元数据与评测代码."
    ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
        "mkdir -p '${BOARD_MODEL_DIR}'"
    scp "${SSH_OPTIONS[@]}" \
        "${MODEL_ROOT}/models/model_int8.dla" \
        "${SCRIPT_DIR}/full_accuracy_board.py" \
        "${SCRIPT_DIR}/face_utils.py" \
        "${BOARD_HOST}:${BOARD_MODEL_DIR}/"
    scp "${SSH_OPTIONS[@]}" "${SMOKE_METADATA}" \
        "${BOARD_HOST}:${BOARD_MODEL_DIR}/metadata.json"
else
    echo "[2/3] 续跑,沿用本次已部署的模型和评测代码."
fi

echo "[3/3] 在 92 完成 LFW 全量十折评测."
resume_arg=()
if [[ "${RESUME}" == "1" ]]; then resume_arg+=(--resume); fi
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "python3 '${BOARD_MODEL_DIR}/full_accuracy_board.py' \
    --dataset-root '${BOARD_DATASET}' \
    --model '${BOARD_MODEL_DIR}/model_int8.dla' \
    --metadata '${BOARD_MODEL_DIR}/metadata.json' \
    --run-dir '${BOARD_RUN}' --run-id '${RUN_ID}' ${resume_arg[*]}"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- "${BOARD_RUN}/report" \
    < "${SCRIPT_DIR}/../../../../../tools/evaluation/capture_board_system.sh"
echo "[OK] LFW 十折全量报告: ${BOARD_RUN}/report"
