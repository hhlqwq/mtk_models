#!/usr/bin/env bash
# 顺序完成所有模型转换、量化、编译、上传及浮点子集评估;板端执行同子集 NPU 评估.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
    : "${MTK_BOARD_HOST:?缺少本地平台配置中的板端连接信息}"
    : "${MTK_BOARD_OPEN_MODELS_ROOT:?缺少本地平台配置中的板端部署根目录}"
    BOARD_DEPLOY_DIR="${MTK_BOARD_OPEN_MODELS_ROOT%/}/deploy"
    printf -v QUOTED_BOARD_DEPLOY_DIR '%q' "${BOARD_DEPLOY_DIR}"
    UPLOAD_CONFIG="$(python "${MTK_MODELS_ROOT}/tools/storage.py" --write-upload-config)"
    echo "[总入口] 上传 compile.sh、evaluate.sh、upload.sh 和本地回传配置到板端."
    ssh -o BatchMode=yes "${MTK_BOARD_HOST}" "mkdir -p -- ${QUOTED_BOARD_DEPLOY_DIR}"
    scp -o BatchMode=yes "${SCRIPT_DIR}/compile.sh" "${SCRIPT_DIR}/evaluate.sh" \
        "${SCRIPT_DIR}/upload.sh" "${UPLOAD_CONFIG}" \
        "${MTK_BOARD_HOST}:${QUOTED_BOARD_DEPLOY_DIR}/"
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
        ENTRY="${MODELS_ROOT}/${MODEL}/deploy/compile.sh"
    else
        ENTRY="${MODELS_ROOT}/${MODEL##*/}/compile.sh"
    fi
    echo "[${LOCATION} ${INDEX}/${#MODELS[@]}] ${MODEL##*/}: ${ENTRY}"
    bash "${ENTRY}"
done

if [[ "${LOCATION}" == 容器 ]]; then
    echo "[上传完成] 请登录开发板执行: bash ${BOARD_DEPLOY_DIR}/compile.sh"
    echo "[浮点子集完成] 所有模型浮点子集评估已完成;待板端执行同子集 NPU 评估及精度损失统计."
else
    echo "[少量样本完成] 所有模型入口执行结束,未执行全量评价."
    bash "${SCRIPT_DIR}/upload.sh" subset
fi
