#!/usr/bin/env bash
# 板端回传已有示例和唯一正式报告,不传模型或过程文件.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/upload.local.sh"
: "${MTK_RESULTS_HOST:?缺少本地回传账户}"
: "${MTK_RESULTS_ROOT:?缺少编译服务器仓库根目录}"
SCOPE="${1:-all}"
if [[ "${SCOPE}" != all && "${SCOPE}" != examples && "${SCOPE}" != subset ]] || (( $# > 1 )); then
    echo "用法: bash upload.sh [examples|subset]; 默认回传全量报告和已有子集报告,subset 回传子集报告,均附带示例." >&2
    exit 2
fi
if [[ "${MTK_RESULTS_ROOT}" != /* || "${MTK_RESULTS_ROOT}" == / ]]; then
    echo "[错误] 回传目标必须为编译服务器仓库根目录的绝对路径." >&2
    exit 2
fi
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
KEY="${MTK_RESULTS_KEY:-${HOME}/.ssh/mtk_results_ed25519}"
if [[ ! -s "${KEY}" ]]; then
    echo "[错误] 缺少板端专用回传密钥: ${KEY}." >&2
    exit 2
fi
printf -v QUOTED_KEY '%q' "${KEY}"
if ssh -V 2>&1 | grep -qi Dropbear; then
    # Dropbear 不支持 BatchMode;无控制终端并拒绝交互认证,仍检查 known_hosts.
    export SSH_ASKPASS=/bin/false DISPLAY=mtk-upload
    SSH=(setsid --wait ssh -T -i "${KEY}")
    RSYNC_SSH="setsid --wait ssh -T -i ${QUOTED_KEY}"
else
    SSH=(ssh -i "${KEY}" -o BatchMode=yes -o StrictHostKeyChecking=yes)
    RSYNC_SSH="ssh -i ${QUOTED_KEY} -o BatchMode=yes -o StrictHostKeyChecking=yes"
fi
STATUS=0
INDEX=0
for MODEL in "${MODELS[@]}"; do
    INDEX=$((INDEX + 1))
    SOURCE="${MODELS_ROOT}/${MODEL##*/}"
    TARGET="${MTK_RESULTS_ROOT%/}/models/${MODEL}"
    echo "[回传 ${INDEX}/${#MODELS[@]}] ${MODEL##*/}"
    if [[ -L "${SOURCE}/examples/output" || -L "${SOURCE}/results/summary.json" || -L "${SOURCE}/results/summary_sub.json" ]]; then
        echo "[错误] 回传源不允许为符号链接: ${MODEL}." >&2
        exit 2
    fi
    printf -v QUOTED_TARGET '%q' "${TARGET}"
    "${SSH[@]}" "${MTK_RESULTS_HOST}" \
        "mkdir -p -- ${QUOTED_TARGET}/examples/output ${QUOTED_TARGET}/results" </dev/null
    if [[ -d "${SOURCE}/examples/output" ]] && \
            [[ -n "$(find "${SOURCE}/examples/output" -type f -size +0c -print -quit)" ]]; then
        rsync -rlt --no-perms --omit-dir-times --no-links --protect-args --info=progress2 \
            -e "${RSYNC_SSH}" "${SOURCE}/examples/output/" \
            "${MTK_RESULTS_HOST}:${TARGET}/examples/output/"
    else
        echo "[缺失] ${MODEL}/examples/output/ 没有非空示例." >&2
        STATUS=1
    fi
    REPORTS=()
    if [[ "${SCOPE}" == all ]]; then
        REPORTS+=(summary.json)
    fi
    if [[ "${SCOPE}" == subset ]] || \
            [[ "${SCOPE}" == all && -s "${SOURCE}/results/summary_sub.json" ]]; then
        REPORTS+=(summary_sub.json)
    fi
    for REPORT in "${REPORTS[@]}"; do
        if [[ -s "${SOURCE}/results/${REPORT}" ]]; then
            rsync -rlt --no-perms --omit-dir-times --protect-args --info=progress2 \
                -e "${RSYNC_SSH}" "${SOURCE}/results/${REPORT}" \
                "${MTK_RESULTS_HOST}:${TARGET}/results/${REPORT}"
        else
            echo "[缺失] ${MODEL}/results/${REPORT} 未生成,不回传空报告." >&2
            STATUS=1
        fi
    done
done
if (( STATUS == 0 )); then
    echo "[回传完成] 已同步 ${SCOPE} 范围的结果,未删除服务器其他文件."
fi
exit "${STATUS}"
