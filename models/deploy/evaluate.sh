#!/usr/bin/env bash
# 顺序评价已有模型;容器准备全量参考,板端执行各模型约定的全量评价.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 只由最外层总入口保存日志,末尾回传不再打开 compile.log.
if [[ "${MTK_DEPLOY_LOG_ACTIVE:-0}" != 1 ]]; then
    LOG_FILE="${SCRIPT_DIR}/evaluate.log"
    : > "${LOG_FILE}"
    set +e
    {
        printf '[日志] %s; 开始: %s; 本次覆盖旧日志.\n' "${LOG_FILE}" "$(date '+%F %T %z')"
        PYTHONUNBUFFERED=1 MTK_DEPLOY_LOG_ACTIVE=1 bash "${BASH_SOURCE[0]}" "$@"
    } 2>&1 | tee -- "${LOG_FILE}"
    LOG_STATUSES=("${PIPESTATUS[@]}")
    # 优先保留原流程退出码;流程成功时仍须检查日志是否写入成功.
    if (( LOG_STATUSES[0] != 0 )); then
        exit "${LOG_STATUSES[0]}"
    fi
    exit "${LOG_STATUSES[1]}"
fi
MODELS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS=(
    perception/image_classification/resnet101
    perception/image_classification/vit_base_patch16_224
    perception/object_detection/yolov5s
    perception/object_detection/yolov8n
    interaction/face_recognition/mobilefacenet
    interaction/pose_detection/rtmpose_body2d
    navigation/segmentation/fastsam
    navigation/single_camera_depth/depth_anything_v2_small
    audio/ambient_sound/yamnet
    audio/stt/whisper_tiny
    pointcloud/three_d_fusion/mit_bevfusion
    gen_ai/llm/minicpm5_2b
    gen_ai/llm/deepseek_r1_distill_qwen_1_5b
)

if [[ -f "${MODELS_ROOT}/../tools/storage.py" ]]; then
    : "${MTK_MODELS_ROOT:?请先通过 docker/enter_container.sh 进入编译容器}"
    LOCATION=容器
else
    LOCATION=板端
fi

INDEX=0
MODEL=总入口
trap 'STATUS=$?; echo "[失败] ${LOCATION}: ${MODEL},退出码 ${STATUS}." >&2; exit "${STATUS}"' ERR
for MODEL in "${MODELS[@]}"; do
    INDEX=$((INDEX + 1))
    if [[ "${LOCATION}" == 容器 ]]; then
        ENTRY="${MODELS_ROOT}/${MODEL}/deploy/evaluate.sh"
    else
        ENTRY="${MODELS_ROOT}/${MODEL##*/}/evaluate.sh"
    fi
    echo "[${LOCATION} ${INDEX}/${#MODELS[@]}] ${MODEL##*/}: ${ENTRY}"
    bash "${ENTRY}"
done

if [[ "${LOCATION}" == 容器 ]]; then
    echo "[参考上传完成] 所有模型全量浮点评估及必要参考上传已完成."
    echo "[下一步] 登录开发板,在部署根目录的 deploy/ 下执行 bash evaluate.sh."
else
    echo "[全量评价完成] 各模型结果分别保存在其 results/summary.json."
    bash "${SCRIPT_DIR}/compile.sh" --return-results full
fi
