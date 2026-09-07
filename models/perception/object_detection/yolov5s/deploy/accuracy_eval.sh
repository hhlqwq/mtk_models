#!/usr/bin/env bash
# YOLOv5s COCO val2017 正式精度评测驱动 (在 89 服务器宿主机执行)。
# 用法: bash accuracy_eval.sh [all|npu|fp32|evaluate]
# 环境变量: MTK_BOARD_HOST / MTK_BOARD_ROOT / TOTAL / CHUNK。

set -euo pipefail

readonly PROJECT_ROOT="/data/users/hailong.he/github/mtk_models"
readonly CONTAINER="hhl_g720_311"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_ROOT="${MTK_BOARD_ROOT:-/root/hailong.he}"
readonly MODEL_DIR="${PROJECT_ROOT}/models/perception/object_detection/yolov5s"
readonly WORK="${PROJECT_ROOT}/.eval/yolov5s"
readonly WORK_C="/workspace/.eval/yolov5s"
readonly EVAL_PY="/workspace/tools/accuracy/yolov5s_val_coco.py"
readonly TOTAL="${TOTAL:-5000}"
readonly CHUNK="${CHUNK:-500}"
readonly STAGE="${1:-all}"
readonly BOARD_EVAL="${BOARD_ROOT}/eval"

docker_run() {
    docker exec "${CONTAINER}" python "${EVAL_PY}" "$@"
}

run_npu() {
    echo "[1/5] 生成 ${TOTAL} 张 COCO val2017 INT8 输入。"
    local done_count=0
    while [ "${done_count}" -lt "${TOTAL}" ]; do
        local remaining=$((TOTAL - done_count))
        local size=$((remaining < CHUNK ? remaining : CHUNK))
        docker_run --stage prepare --start "${done_count}" --count "${size}" \
            --bins-dir "${WORK_C}/npu_bins"
        done_count=$((done_count + size))
        echo "  已准备 ${done_count}/${TOTAL}"
    done

    echo "[2/5] 推送 DLA、输入和板端循环脚本。"
    ssh "${BOARD_HOST}" "mkdir -p '${BOARD_EVAL}/inputs' '${BOARD_EVAL}/outputs'"
    scp -q "${MODEL_DIR}/models/model_int8.dla" \
        "${BOARD_HOST}:${BOARD_EVAL}/model_int8.dla"
    scp -q "${MODEL_DIR}/deploy/inference_demo/board_eval_loop.sh" \
        "${BOARD_HOST}:${BOARD_EVAL}/board_eval_loop.sh"
    tar -C "${WORK}/npu_bins" -cf - . | ssh "${BOARD_HOST}" \
        "tar -C '${BOARD_EVAL}/inputs' -xf -"

    echo "[3/5] 板端 neuronrt 批量推理 (进度每 50 张打印一次)。"
    ssh "${BOARD_HOST}" "sh '${BOARD_EVAL}/board_eval_loop.sh' \
        '${BOARD_EVAL}/model_int8.dla' '${BOARD_EVAL}/inputs' \
        '${BOARD_EVAL}/outputs'"

    echo "[4/5] 回传板端输出。"
    mkdir -p "${WORK}/npu_outputs"
    ssh "${BOARD_HOST}" "tar -C '${BOARD_EVAL}/outputs' -cf - ." \
        | tar -C "${WORK}/npu_outputs" -xf -
    echo "  清理板端评测数据。"
    ssh "${BOARD_HOST}" "rm -rf '${BOARD_EVAL}'"

    echo "[5/5] 解码 NPU 输出。"
    docker_run --stage decode --backend npu \
        --bins-dir "${WORK_C}/npu_outputs"
}

run_fp32() {
    echo "[fp32] PyTorch 基线推理 + 解码。"
    docker_run --stage decode --backend torch
    echo "[fp32] ONNX Runtime 基线推理 + 解码。"
    docker_run --stage decode --backend onnx
}

run_evaluate() {
    for backend in npu torch onnx; do
        echo "[eval] ${backend}"
        docker_run --stage evaluate --backend "${backend}" \
            > "${WORK}/${backend}_eval.txt" 2>&1 || true
        tail -4 "${WORK}/${backend}_eval.txt"
        cat "${WORK}/${backend}_summary.json"
    done
}

mkdir -p "${WORK}"
case "${STAGE}" in
    npu) run_npu ;;
    fp32) run_fp32 ;;
    evaluate) run_evaluate ;;
    all) run_npu; run_fp32; run_evaluate ;;
    *) echo "未知阶段: ${STAGE}" >&2; exit 1 ;;
esac
echo "[OK] 评测驱动结束, 产物位于 ${WORK}。"
