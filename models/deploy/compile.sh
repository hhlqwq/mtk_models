#!/usr/bin/env bash
# 顺序完成所有模型转换、量化、编译、上传及浮点子集评估;板端执行同子集 NPU 评估.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RETURN_ONLY=0
RESULT_SCOPE=subset
if (( $# )); then
    if [[ "$#" != 2 || "$1" != --return-results || "$2" != full ]]; then
        echo "[错误] 正常编译无参数;结果回传分支只由 evaluate.sh 调用." >&2
        exit 2
    fi
    RETURN_ONLY=1
    RESULT_SCOPE=all
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

if [[ -f "${MODELS_ROOT}/../tools/storage.py" ]]; then
    (( RETURN_ONLY == 0 )) || { echo "[错误] 结果回传只能在板端执行." >&2; exit 2; }
    : "${MTK_MODELS_ROOT:?请先通过 docker/enter_container.sh 进入编译容器}"
    : "${MTK_BOARD_HOST:?缺少本地平台配置中的板端连接信息}"
    : "${MTK_BOARD_OPEN_MODELS_ROOT:?缺少本地平台配置中的板端部署根目录}"
    BOARD_DEPLOY_DIR="${MTK_BOARD_OPEN_MODELS_ROOT%/}/deploy"
    printf -v QUOTED_BOARD_DEPLOY_DIR '%q' "${BOARD_DEPLOY_DIR}"
    UPLOAD_CONFIG="$(python "${MTK_MODELS_ROOT}/tools/storage.py" --write-upload-config)"
    echo "[总入口] 上传 compile.sh、evaluate.sh 和本地回传配置到板端."
    ssh -o BatchMode=yes "${MTK_BOARD_HOST}" "mkdir -p -- ${QUOTED_BOARD_DEPLOY_DIR}"
    scp -o BatchMode=yes "${SCRIPT_DIR}/compile.sh" "${SCRIPT_DIR}/evaluate.sh" \
        "${UPLOAD_CONFIG}" \
        "${MTK_BOARD_HOST}:${QUOTED_BOARD_DEPLOY_DIR}/"
    LOCATION=容器
else
    LOCATION=板端
fi

if (( RETURN_ONLY == 0 )); then
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

    fi

fi

if [[ "${LOCATION}" == 板端 ]]; then
    source "${SCRIPT_DIR}/platform.local.sh"
    : "${MTK_RESULTS_HOST:?缺少本地回传账户}"
    : "${MTK_RESULTS_ROOT:?缺少编译服务器仓库根目录}"
    SCOPE="${RESULT_SCOPE}"
    if [[ "${MTK_RESULTS_ROOT}" != /* || "${MTK_RESULTS_ROOT}" == / ]]; then
        echo "[错误] 回传目标必须为编译服务器仓库根目录的绝对路径." >&2
        exit 2
    fi
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
        if [[ -L "${SOURCE}/examples/output" || -L "${SOURCE}/results/summary.json" || -L "${SOURCE}/results/summary_test.json" ]]; then
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
                [[ "${SCOPE}" == all && -s "${SOURCE}/results/summary_test.json" ]]; then
            REPORTS+=(summary_test.json)
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

fi
