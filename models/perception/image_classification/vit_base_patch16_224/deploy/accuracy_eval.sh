#!/usr/bin/env bash
# ViT-Base Patch16 224 精度评测驱动 (89 宿主机执行).
# 用法: bash accuracy_eval.sh [prepare|board|compare|all]
# 环境变量: EVAL_RUN_ID / START / TOTAL(默认 1000) / CHUNK /
# IMAGENET_LABELS / ACCURACY_EXCLUDE_START / ACCURACY_EXCLUDE_COUNT /
# MTK_BOARD_HOST / MTK_BOARD_ROOT.

set -euo pipefail

readonly PROJECT_ROOT="/data/users/hailong.he/github/mtk_models"
readonly CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_ROOT="${MTK_BOARD_ROOT:-/root/hailong.he}"
readonly MODEL_DIR="${PROJECT_ROOT}/models/perception/image_classification/vit_base_patch16_224"
readonly EVAL_PY="${PROJECT_ROOT}/tools/accuracy/vit_val_agreement.py"
readonly START="${START:-0}"
readonly TOTAL="${TOTAL:-1000}"
readonly CHUNK="${CHUNK:-500}"
readonly FP32_PROVIDER="${FP32_PROVIDER:-cpu}"
readonly ACCURACY_EXCLUDE_START="${ACCURACY_EXCLUDE_START:-1000}"
readonly ACCURACY_EXCLUDE_COUNT="${ACCURACY_EXCLUDE_COUNT:-100}"
readonly STAGE="${1:-all}"
readonly IMAGES_DIR="/data/users/hailong.he/nas_smb/Datasets/open_source/raw/ILSVRC2012/val"

if [[ -z "${EVAL_RUN_ID:-}" ]]; then
    if [[ "${STAGE}" != "all" ]]; then
        echo "分阶段执行必须显式设置 EVAL_RUN_ID." >&2
        exit 1
    fi
    EVAL_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ ! "${EVAL_RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "EVAL_RUN_ID 只能包含字母、数字、点、下划线和连字符." >&2
    exit 1
fi
readonly EVAL_RUN_ID
readonly WORK="${PROJECT_ROOT}/.eval/vit_base_patch16_224/runs/${EVAL_RUN_ID}"
readonly WORK_C="${WORK}"
readonly BOARD_EVAL="${BOARD_ROOT}/vit_eval/runs/${EVAL_RUN_ID}"
readonly RUN_CONFIG="${WORK}/run_inputs_sha256.txt"
readonly -a SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

docker_run() {
    docker exec "${CONTAINER}" python "${EVAL_PY}" \
        --work-dir "${WORK_C}" "$@"
}

write_or_check_config() {
    local candidate
    candidate="$(mktemp)"
    {
        echo "run_id=${EVAL_RUN_ID}"
        echo "start=${START}"
        echo "total=${TOTAL}"
        echo "images_dir=${IMAGES_DIR}"
        echo "fp32_provider=${FP32_PROVIDER}"
        sha256sum "${MODEL_DIR}/models/model_fp32.onnx" \
            "${MODEL_DIR}/models/model_mtk_compatible.onnx" \
            "${MODEL_DIR}/models/model_int8.tflite" \
            "${MODEL_DIR}/models/model_int8.dla" \
            "${PROJECT_ROOT}/tools/accuracy/vit_val_agreement.py" \
            "${MODEL_DIR}/deploy/inference_demo/board_eval_loop.sh"
        if [[ -n "${IMAGENET_LABELS:-}" ]]; then
            sha256sum "${IMAGENET_LABELS}"
        fi
    } > "${candidate}"
    if [[ -f "${RUN_CONFIG}" ]]; then
        if ! cmp -s "${candidate}" "${RUN_CONFIG}"; then
            echo "运行输入与已有 EVAL_RUN_ID 不一致,请使用新的运行 ID." >&2
            rm -f "${candidate}"
            exit 1
        fi
    else
        mv "${candidate}" "${RUN_CONFIG}"
        candidate=""
    fi
    [[ -z "${candidate}" ]] || rm -f "${candidate}"
}

run_prepare() {
    echo "[prepare] 生成 INT8 输入与 FP32 基线 logits."
    rm -rf "${WORK}/npu_bins" "${WORK}/fp32_logits"
    rm -f "${WORK}/manifest.jsonl" "${WORK}/prepare.done"
    mkdir -p "${WORK}/npu_bins" "${WORK}/fp32_logits"
    local done_count=0
    while [ "${done_count}" -lt "${TOTAL}" ]; do
        local remaining=$((TOTAL - done_count))
        local size=$((remaining < CHUNK ? remaining : CHUNK))
        local global_start=$((START + done_count))
        docker_run --stage prepare --start "${global_start}" --count "${size}" \
            --onnx-provider "${FP32_PROVIDER}"
        done_count=$((done_count + size))
        echo "  已准备 ${done_count}/${TOTAL}"
    done
    local manifest_count
    manifest_count="$(wc -l < "${WORK}/manifest.jsonl")"
    [[ "${manifest_count}" -eq "${TOTAL}" ]]
    touch "${WORK}/prepare.done"
}

run_board() {
    echo "[board] 推送 DLA、输入并批量推理."
    test -f "${WORK}/prepare.done"
    ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
        "mkdir -p '${BOARD_EVAL}/inputs' '${BOARD_EVAL}/outputs'"
    scp "${SSH_OPTIONS[@]}" -q "${MODEL_DIR}/models/model_int8.dla" \
        "${BOARD_HOST}:${BOARD_EVAL}/model_int8.dla"
    scp "${SSH_OPTIONS[@]}" -q \
        "${MODEL_DIR}/deploy/inference_demo/board_eval_loop.sh" \
        "${BOARD_HOST}:${BOARD_EVAL}/board_eval_loop.sh"
    tar -C "${WORK}/npu_bins" -cf - . | \
        ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
        "tar -C '${BOARD_EVAL}/inputs' -m -xf -"
    ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
        "sh '${BOARD_EVAL}/board_eval_loop.sh' \
        '${BOARD_EVAL}/model_int8.dla' '${BOARD_EVAL}/inputs' \
        '${BOARD_EVAL}/outputs'"
    echo "[board] 回传输出,板端原始证据保留在 ${BOARD_EVAL}."
    mkdir -p "${WORK}/npu_outputs"
    ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
        "tar -C '${BOARD_EVAL}/outputs' -cf - ." \
        | tar -C "${WORK}/npu_outputs" -xf -
    local output_count
    output_count="$(find "${WORK}/npu_outputs" -maxdepth 1 \
        -type f -name '*_0.bin' -size +0c | wc -l)"
    [[ "${output_count}" -eq "${TOTAL}" ]]
    touch "${WORK}/board.done"
}

run_compare() {
    echo "[compare] NPU 与 FP32 基线对齐分析."
    test -f "${WORK}/prepare.done"
    test -f "${WORK}/board.done"
    if [[ -n "${IMAGENET_LABELS:-}" ]]; then
        echo "  使用已映射的 0-based ImageNet 标签报告绝对 Top-1/Top-5."
        docker_run --stage compare --start "${START}" --count "${TOTAL}" \
            --exclude-accuracy-start "${ACCURACY_EXCLUDE_START}" \
            --exclude-accuracy-count "${ACCURACY_EXCLUDE_COUNT}" \
            --labels "${IMAGENET_LABELS}"
    else
        docker_run --stage compare --start "${START}" --count "${TOTAL}"
    fi
}

mkdir -p "${WORK}"
write_or_check_config
case "${STAGE}" in
    prepare) run_prepare ;;
    board) run_board ;;
    compare) run_compare ;;
    all) run_prepare; run_board; run_compare ;;
    *) echo "未知阶段: ${STAGE}" >&2; exit 1 ;;
esac
echo "[OK] ViT 评测驱动结束, run_id=${EVAL_RUN_ID}, 产物位于 ${WORK}."
