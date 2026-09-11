#!/usr/bin/env bash
# YOLOv5s COCO val2017 正式精度评测驱动 (在 89 服务器宿主机执行).
# 用法: bash accuracy_eval.sh [all|npu|fp32|evaluate]
# 环境变量: MTK_BOARD_HOST / MTK_BOARD_ROOT / EVAL_RUN_ID / TOTAL / CHUNK.

set -euo pipefail

readonly PROJECT_ROOT="/data/users/hailong.he/github/mtk_models"
readonly CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_ROOT="${MTK_BOARD_ROOT:-/root/hailong.he}"
readonly MODEL_DIR="${PROJECT_ROOT}/models/perception/object_detection/yolov5s"
readonly RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
readonly WORK_BASE="${PROJECT_ROOT}/.eval/yolov5s/runs"
readonly WORK="${WORK_BASE}/${RUN_ID}"
readonly WORK_C="${WORK}"
readonly EVAL_PY="${PROJECT_ROOT}/tools/accuracy/yolov5s_val_coco.py"
readonly TOTAL="${TOTAL:-5000}"
readonly CHUNK="${CHUNK:-500}"
readonly STAGE="${1:-all}"
readonly BOARD_EVAL="${BOARD_ROOT}/eval/yolov5s/${RUN_ID}"
readonly HASH_FILE="${WORK}/run_inputs_sha256.txt"
readonly CONFIG_FILE="${WORK}/run_config.txt"

if ! [[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "EVAL_RUN_ID 必须以字母或数字开头,且只能包含字母、数字、点、下划线和连字符: ${RUN_ID}" >&2
    exit 1
fi
if ! [[ "${TOTAL}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${CHUNK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TOTAL 和 CHUNK 必须是正整数." >&2
    exit 1
fi

# 在容器中执行统一评测工具,并固定到本次运行目录.
docker_run() {
    docker exec "${CONTAINER}" python "${EVAL_PY}" \
        --work-dir "${WORK_C}" --expected-images "${TOTAL}" "$@"
}

# 初始化隔离运行目录,并阻止同一运行 ID 混用不同模型产物.
initialize_run() {
    local model_files=(
        "${MODEL_DIR}/models/yolov5s.pt"
        "${MODEL_DIR}/models/model_fp32.onnx"
        "${MODEL_DIR}/models/model_int8.tflite"
        "${MODEL_DIR}/models/model_int8.dla"
        "${PROJECT_ROOT}/tools/accuracy/yolov5s_val_coco.py"
        "${MODEL_DIR}/deploy/inference_demo/board_eval_loop.sh"
    )
    local model_file
    mkdir -p "${WORK}"
    for model_file in "${model_files[@]}"; do
        test -f "${model_file}"
    done
    if [[ -f "${HASH_FILE}" ]]; then
        sha256sum --check --status "${HASH_FILE}" || {
            echo "运行目录对应的模型或评测脚本哈希已变化,请使用新的 EVAL_RUN_ID." >&2
            exit 1
        }
    else
        sha256sum "${model_files[@]}" > "${HASH_FILE}"
    fi
    if [[ -f "${CONFIG_FILE}" ]]; then
        grep -Fxq "TOTAL=${TOTAL}" "${CONFIG_FILE}" || {
            echo "运行目录的 TOTAL 已变化,请使用新的 EVAL_RUN_ID." >&2
            exit 1
        }
    else
        printf 'TOTAL=%s\n' "${TOTAL}" > "${CONFIG_FILE}"
    fi
}

# 生成输入、执行板端推理并解码 NPU 输出.
run_npu() {
    echo "[1/5] 生成 ${TOTAL} 张 COCO val2017 INT8 输入."
    local done_count=0
    while [ "${done_count}" -lt "${TOTAL}" ]; do
        local remaining=$((TOTAL - done_count))
        local size=$((remaining < CHUNK ? remaining : CHUNK))
        docker_run --stage prepare --start "${done_count}" --count "${size}" \
            --bins-dir "${WORK_C}/npu_bins"
        done_count=$((done_count + size))
        echo "  已准备 ${done_count}/${TOTAL}"
    done

    echo "[2/5] 推送 DLA、输入和板端循环脚本."
    ssh "${BOARD_HOST}" "mkdir -p '${BOARD_EVAL}/inputs' '${BOARD_EVAL}/outputs'"
    scp -q "${MODEL_DIR}/models/model_int8.dla" \
        "${BOARD_HOST}:${BOARD_EVAL}/model_int8.dla"
    scp -q "${MODEL_DIR}/deploy/inference_demo/board_eval_loop.sh" \
        "${BOARD_HOST}:${BOARD_EVAL}/board_eval_loop.sh"
    tar -C "${WORK}/npu_bins" -cf - . | ssh "${BOARD_HOST}" \
        "tar -C '${BOARD_EVAL}/inputs' -xf -"

    echo "[3/5] 板端 neuronrt 批量推理 (进度每 50 张打印一次)."
    ssh "${BOARD_HOST}" "sh '${BOARD_EVAL}/board_eval_loop.sh' \
        '${BOARD_EVAL}/model_int8.dla' '${BOARD_EVAL}/inputs' \
        '${BOARD_EVAL}/outputs'"

    echo "[4/5] 回传板端输出."
    mkdir -p "${WORK}/npu_outputs"
    ssh "${BOARD_HOST}" "tar -C '${BOARD_EVAL}/outputs' -cf - ." \
        | tar -C "${WORK}/npu_outputs" -xf -
    echo "  清理本次运行的板端评测数据."
    ssh "${BOARD_HOST}" "rm -rf '${BOARD_EVAL}'"

    echo "[5/5] 解码 NPU 输出."
    docker_run --stage decode --backend npu \
        --bins-dir "${WORK_C}/npu_outputs"
}

# 执行 PyTorch 和 ONNX 两个 FP32 基线.
run_fp32() {
    echo "[fp32] PyTorch 基线推理 + 解码."
    docker_run --stage decode --backend torch
    echo "[fp32] ONNX Runtime 基线推理 + 解码."
    docker_run --stage decode --backend onnx
}

# 校验覆盖范围并计算三个后端的 COCO 指标.
run_evaluate() {
    for backend in npu torch onnx; do
        echo "[eval] ${backend}"
        docker_run --stage evaluate --backend "${backend}" \
            > "${WORK}/${backend}_eval.txt" 2>&1
        tail -4 "${WORK}/${backend}_eval.txt"
        cat "${WORK}/${backend}_summary.json"
    done
}

initialize_run
echo "[INFO] 评测运行 ID: ${RUN_ID}"
echo "[INFO] 评测目录: ${WORK}"
case "${STAGE}" in
    npu) run_npu ;;
    fp32) run_fp32 ;;
    evaluate) run_evaluate ;;
    all) run_npu; run_fp32; run_evaluate ;;
    *) echo "未知阶段: ${STAGE}" >&2; exit 1 ;;
esac
echo "[OK] 评测驱动结束, 产物位于 ${WORK}."
