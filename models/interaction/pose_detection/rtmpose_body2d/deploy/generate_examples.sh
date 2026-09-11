#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_DIR="${MTK_BOARD_ROOT:-/root/hailong.he}/rtmpose_public_examples"
readonly INPUT_DIR="${MODEL_ROOT}/examples/input/public"
readonly WORK_DIR="${MODEL_ROOT}/examples/input/generated"
readonly RAW_DIR="${MODEL_ROOT}/examples/output/board_raw"
readonly OUTPUT_DIR="${MODEL_ROOT}/examples/output/public"
readonly -a SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

test "$(find "${INPUT_DIR}" -maxdepth 1 -type f -name '*.jpg' | wc -l)" -eq 3
test -f "${INPUT_DIR}/annotations.json"
docker exec "${CONTAINER}" sh -c \
    "rm -rf \
      '/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/input/generated' \
      '/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/output/board_raw' \
      '/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/output/public' && \
     mkdir -p \
      '/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/input/generated' \
      '/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/output/board_raw' \
      '/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/output/public' && \
     chmod 0777 \
      '/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/input/generated' \
      '/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/output/board_raw' \
      '/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/output/public'"

echo "[1/6] 使用公开人体框生成三份 INT8 输入."
docker exec "${CONTAINER}" python3 \
    "/workspace/models/interaction/pose_detection/rtmpose_body2d/deploy/inference_demo/prepare_input.py" \
    --image-dir "/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/input/public" \
    --annotations "/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/input/public/annotations.json" \
    --tflite "/workspace/models/interaction/pose_detection/rtmpose_body2d/models/model_int8.tflite" \
    --output-dir "/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/input/generated" \
    --metadata "/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/input/generated/metadata.json" \
    --selection-mode all --count 3
echo "[2/6] 创建干净的板端三图运行目录."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "rm -rf '${BOARD_DIR}' && mkdir -p '${BOARD_DIR}/inputs' '${BOARD_DIR}/output'"
echo "[3/6] 部署 DLA、输入和逐图运行脚本."
scp "${SSH_OPTIONS[@]}" "${MODEL_ROOT}/models/model_int8.dla" \
    "${MODEL_ROOT}/deploy/inference_demo/run_examples.sh" \
    "${BOARD_HOST}:${BOARD_DIR}/"
scp "${SSH_OPTIONS[@]}" "${WORK_DIR}"/*.bin \
    "${BOARD_HOST}:${BOARD_DIR}/inputs/"
echo "[4/6] 在 Genio 720 NPU 逐张推理."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "chmod +x '${BOARD_DIR}/run_examples.sh' && '${BOARD_DIR}/run_examples.sh'"
scp "${SSH_OPTIONS[@]}" "${BOARD_HOST}:${BOARD_DIR}/output/*.bin" "${RAW_DIR}/"
echo "[5/6] 反量化 SimCC 输出并生成 133 点结果."
docker exec "${CONTAINER}" python3 \
    "/workspace/models/interaction/pose_detection/rtmpose_body2d/deploy/inference_demo/postprocess_keypoints.py" \
    --metadata "/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/input/generated/metadata.json" \
    --input-dir "/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/input/generated" \
    --output-dir "/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/output/board_raw" \
    --result-dir "/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/output/public"
echo "[6/6] 比较 FP32 ONNX 与板端 NPU 输出."
docker exec "${CONTAINER}" python3 \
    "/workspace/models/interaction/pose_detection/rtmpose_body2d/deploy/inference_demo/compare_backends.py" \
    --onnx "/workspace/models/interaction/pose_detection/rtmpose_body2d/models/model_mtk_compatible.onnx" \
    --metadata "/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/input/generated/metadata.json" \
    --input-dir "/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/input/generated" \
    --output-dir "/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/output/board_raw" \
    --result "/workspace/models/interaction/pose_detection/rtmpose_body2d/examples/output/public/backend_comparison.json"
echo "[OK] RTMPose 三张公开图片已在 Genio 720 完成测试: ${OUTPUT_DIR}"
