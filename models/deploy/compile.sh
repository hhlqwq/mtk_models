#!/usr/bin/env bash
# 顺序编译上传所有模型;上传到板端后,同一入口只运行少量样本检查.
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
    UPLOAD_CONFIG="$(python - <<'PY'
from datetime import date
import os
from pathlib import Path
import shlex
from tools.storage import platform_environment

config = platform_environment()
values = {name: config.get(name, "") for name in ("MTK_RESULTS_HOST", "MTK_RESULTS_ROOT")}
if not all(values.values()) or not values["MTK_RESULTS_ROOT"].startswith("/"):
    raise ValueError("请在本地平台配置填写 results_host 和 results_root 绝对路径.")
path = Path("/tmp/hailongcodex") / date.today().isoformat() / f"results_upload_{os.getuid()}"
path.mkdir(parents=True, exist_ok=True)
target = path / "upload.local.sh"
target.write_text("".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items()), encoding="utf-8")
target.chmod(0o600)
print(target)
PY
)"
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
    echo "[待板端检查] 尚未执行板端推理或生成展示示例."
else
    echo "[少量样本完成] 所有模型入口执行结束,未执行全量评价."
    bash "${SCRIPT_DIR}/upload.sh" examples
fi
