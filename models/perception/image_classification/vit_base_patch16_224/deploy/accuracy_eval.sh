#!/usr/bin/env bash
# ViT-Base Patch16 224 精度评测驱动 (89 宿主机执行)。
# 用法: bash accuracy_eval.sh [prepare|board|compare|all]
# 环境变量: TOTAL(评测张数, 默认 1000) / CHUNK / MTK_BOARD_HOST / MTK_BOARD_ROOT。

set -euo pipefail

readonly PROJECT_ROOT="/data/users/hailong.he/github/mtk_models"
readonly CONTAINER="hhl_g720_311"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_ROOT="${MTK_BOARD_ROOT:-/root/hailong.he}"
readonly MODEL_DIR="${PROJECT_ROOT}/models/perception/image_classification/vit_base_patch16_224"
readonly WORK="${PROJECT_ROOT}/.eval/vit_base_patch16_224"
readonly WORK_C="/workspace/.eval/vit_base_patch16_224"
readonly EVAL_PY="/workspace/tools/accuracy/vit_val_agreement.py"
readonly TOTAL="${TOTAL:-1000}"
readonly CHUNK="${CHUNK:-500}"
readonly STAGE="${1:-all}"
readonly BOARD_EVAL="${BOARD_ROOT}/vit_eval"

docker_run() {
    docker exec "${CONTAINER}" python "${EVAL_PY}" "$@"
}

run_prepare() {
    echo "[prepare] 生成 INT8 输入与 FP32 基线 logits。"
    local done_count=0
    while [ "${done_count}" -lt "${TOTAL}" ]; do
        local remaining=$((TOTAL - done_count))
        local size=$((remaining < CHUNK ? remaining : CHUNK))
        docker_run --stage prepare --start "${done_count}" --count "${size}"
        done_count=$((done_count + size))
        echo "  已准备 ${done_count}/${TOTAL}"
    done
}

run_board() {
    echo "[board] 推送 DLA、输入并批量推理。"
    ssh "${BOARD_HOST}" "mkdir -p '${BOARD_EVAL}/inputs' '${BOARD_EVAL}/outputs'"
    scp -q "${MODEL_DIR}/models/model_int8.dla" \
        "${BOARD_HOST}:${BOARD_EVAL}/model_int8.dla"
    scp -q "${MODEL_DIR}/deploy/inference_demo/board_eval_loop.sh" \
        "${BOARD_HOST}:${BOARD_EVAL}/board_eval_loop.sh"
    tar -C "${WORK}/npu_bins" -cf - . | ssh "${BOARD_HOST}" \
        "tar -C '${BOARD_EVAL}/inputs' -xf -"
    ssh "${BOARD_HOST}" "sh '${BOARD_EVAL}/board_eval_loop.sh' \
        '${BOARD_EVAL}/model_int8.dla' '${BOARD_EVAL}/inputs' \
        '${BOARD_EVAL}/outputs'"
    echo "[board] 回传输出并清理。"
    mkdir -p "${WORK}/npu_outputs"
    ssh "${BOARD_HOST}" "tar -C '${BOARD_EVAL}/outputs' -cf - ." \
        | tar -C "${WORK}/npu_outputs" -xf -
    ssh "${BOARD_HOST}" "rm -rf '${BOARD_EVAL}'"
}

run_compare() {
    echo "[compare] NPU 与 FP32 基线对齐分析。"
    if [[ -f "${WORK}/labels.txt" ]]; then
        echo "  检测到 ${WORK}/labels.txt, 同时报告绝对 Top-1/Top-5。"
        docker_run --stage compare --count "${TOTAL}" \
            --labels "${WORK_C}/labels.txt"
    else
        docker_run --stage compare --count "${TOTAL}"
    fi
}

mkdir -p "${WORK}"
case "${STAGE}" in
    prepare) run_prepare ;;
    board) run_board ;;
    compare) run_compare ;;
    all) run_prepare; run_board; run_compare ;;
    *) echo "未知阶段: ${STAGE}" >&2; exit 1 ;;
esac
echo "[OK] ViT 评测驱动结束, 产物位于 ${WORK}。"
