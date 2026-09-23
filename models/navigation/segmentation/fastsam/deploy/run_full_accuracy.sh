#!/usr/bin/env bash
# 在 89 编译 FastSAM,在 92 测完整 COCO val2017 类别无关分割 AP.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
readonly RESUME="${EVAL_RESUME:-0}"
readonly CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_DATASET="${MTK_BOARD_DATASETS_ROOT:-/root/hailong.he/datasets}/coco/val2017"
readonly BOARD_MODEL_ROOT="${MTK_BOARD_OPEN_MODELS_ROOT:-/root/hailong.he/open_models}/fastsam"
readonly BOARD_RUN="${BOARD_MODEL_ROOT}/eval/${RUN_ID}"
readonly BOARD_MODEL_DIR="${BOARD_MODEL_ROOT}/models/${RUN_ID}"
readonly -a SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "[ERROR] EVAL_RUN_ID 非法." >&2
    exit 2
fi
if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
    echo "[ERROR] EVAL_RESUME 只能为 0 或 1." >&2
    exit 2
fi
host_epoch="$(date -u +%s)"
board_epoch="$(ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" 'date -u +%s')"
if [[ ! "${board_epoch}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] 无法读取 92 的 UTC 时间." >&2
    exit 2
fi
clock_delta=$((host_epoch - board_epoch))
if (( clock_delta < 0 )); then clock_delta=$((-clock_delta)); fi
if (( clock_delta > 600 )); then
    echo "[ERROR] 92 与 89 的时钟相差 ${clock_delta} 秒,请先校准板端时间." >&2
    exit 2
fi

echo "[1/5] 在 92 预检完整 COCO 实例标注、图片和评测依赖."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_DATASET}" "${BOARD_RUN}" "${RESUME}" <<'BOARD_PREFLIGHT'
set -euo pipefail
readonly dataset="$1"
readonly run_dir="$2"
readonly resume="$3"
if [[ "${resume}" == "1" ]]; then
    test -d "${run_dir}"
else
    test ! -e "${run_dir}"
fi
test -s "${dataset}/annotations/instances_val2017.json"
test "$(find "${dataset}/images" -maxdepth 1 -type f -name '*.jpg' | wc -l)" -eq 5000
python3 -c 'import cv2, numpy, pycocotools'
BOARD_PREFLIGHT

if [[ "${RESUME}" == "0" ]]; then
    echo "[2/5] 在 89 编译 DLA 和 AArch64 C++ 程序."
    if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null)" != "true" ]]; then
        echo "[ERROR] 编译容器未运行: ${CONTAINER}." >&2
        exit 2
    fi
    docker exec "${CONTAINER}" bash "${SCRIPT_DIR}/build.sh"
    bash "${SCRIPT_DIR}/build_board_cpp.sh"
    test -s "${MODEL_ROOT}/models/runtime_config.csv"

    echo "[3/5] 部署本次专属 DLA、量化配置和评测代码."
    ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
        "mkdir -p '${BOARD_MODEL_DIR}' '${BOARD_RUN}/tools' '${BOARD_RUN}/report'"
    scp "${SSH_OPTIONS[@]}" \
        "${MODEL_ROOT}/models/model_int8.dla" \
        "${MODEL_ROOT}/models/runtime_config.csv" \
        "${SCRIPT_DIR}/inference_demo/fastsam_board" \
        "${BOARD_HOST}:${BOARD_MODEL_DIR}/"
    scp "${SSH_OPTIONS[@]}" "${SCRIPT_DIR}/full_accuracy_board.py" \
        "${BOARD_HOST}:${BOARD_RUN}/tools/"
else
    echo "[2-3/5] 续跑,沿用该 run 已部署的模型和评测代码."
fi

echo "[4/5] 在 92 完成 5000 张类别无关实例分割 AP 与耗时统计."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_RUN}" "${BOARD_MODEL_DIR}" "${BOARD_DATASET}" "${RUN_ID}" "${RESUME}" <<'BOARD_EVAL'
set -euo pipefail
readonly run_dir="$1"
readonly model_dir="$2"
readonly dataset="$3"
readonly run_id="$4"
readonly resume="$5"
resume_arg=()
if [[ "${resume}" == "1" ]]; then resume_arg+=(--resume); fi
python3 "${run_dir}/tools/full_accuracy_board.py" \
    --binary "${model_dir}/fastsam_board" \
    --model "${model_dir}/model_int8.dla" \
    --config "${model_dir}/runtime_config.csv" \
    --images "${dataset}/images" \
    --annotations "${dataset}/annotations/instances_val2017.json" \
    --work-dir "${run_dir}" --run-id "${run_id}" "${resume_arg[@]}" \
    2>&1 | tee "${run_dir}/board_eval.log"
BOARD_EVAL

echo "[5/5] 在 92 固化输入与原始预测哈希,不清理现场."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_RUN}" "${BOARD_MODEL_DIR}" "${BOARD_DATASET}" <<'BOARD_HASH'
set -euo pipefail
readonly run_dir="$1"
readonly model_dir="$2"
readonly dataset="$3"
sha256sum "${model_dir}/model_int8.dla" \
    "${model_dir}/runtime_config.csv" \
    "${model_dir}/fastsam_board" \
    "${run_dir}/tools/full_accuracy_board.py" \
    "${dataset}/annotations/instances_val2017.json" \
    > "${run_dir}/report/run_inputs_sha256.txt"
sha256sum "${run_dir}/coco_segm_predictions.json" \
    "${run_dir}/processed_ids.txt" \
    > "${run_dir}/report/raw_outputs_sha256.txt"
cp "${run_dir}/board_eval.log" "${run_dir}/report/"
BOARD_HASH
echo "[OK] FastSAM 板端全量精度报告: ${BOARD_RUN}/report"
echo "[NEXT] 手动上传结果到 Git 后,运行 EVAL_RUN_ID=${RUN_ID} bash ${SCRIPT_DIR}/cleanup_full_accuracy.sh"
