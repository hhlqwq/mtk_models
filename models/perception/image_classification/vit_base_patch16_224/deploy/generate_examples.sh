#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT="$(cd "${MODEL_ROOT}/../../../.." && pwd)"
readonly CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_DIR="${MTK_BOARD_ROOT:-/root/hailong.he}/vit_public_examples"
readonly INPUT_DIR="${MODEL_ROOT}/examples/input/public"
readonly WORK_DIR="${MODEL_ROOT}/examples/input/generated"
readonly RAW_DIR="${MODEL_ROOT}/examples/output/board_raw"
readonly OUTPUT_DIR="${MODEL_ROOT}/examples/output/public"
readonly -a SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

mapfile -t images < <(find "${INPUT_DIR}" -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort)
test "${#images[@]}" -eq 3
docker exec "${CONTAINER}" sh -c \
    "rm -rf \
      '/workspace/models/perception/image_classification/vit_base_patch16_224/examples/input/generated' \
      '/workspace/models/perception/image_classification/vit_base_patch16_224/examples/output/board_raw' \
      '/workspace/models/perception/image_classification/vit_base_patch16_224/examples/output/public' && \
     mkdir -p \
      '/workspace/models/perception/image_classification/vit_base_patch16_224/examples/input/generated' \
      '/workspace/models/perception/image_classification/vit_base_patch16_224/examples/output/board_raw' \
      '/workspace/models/perception/image_classification/vit_base_patch16_224/examples/output/public' && \
     chmod 0777 \
      '/workspace/models/perception/image_classification/vit_base_patch16_224/examples/input/generated' \
      '/workspace/models/perception/image_classification/vit_base_patch16_224/examples/output/board_raw' \
      '/workspace/models/perception/image_classification/vit_base_patch16_224/examples/output/public'"

echo "[1/5] 为三张公开图片生成 INT8 输入."
index=0
for image_path in "${images[@]}"; do
    index=$((index + 1))
    relative_image="${image_path#${REPO_ROOT}/}"
    docker exec "${CONTAINER}" python3 \
        "/workspace/models/perception/image_classification/vit_base_patch16_224/deploy/inference_demo/prepare_input.py" \
        --image "/workspace/${relative_image}" \
        --tflite "/workspace/models/perception/image_classification/vit_base_patch16_224/models/model_int8.tflite" \
        --output "/workspace/models/perception/image_classification/vit_base_patch16_224/examples/input/generated/sample_${index}.bin" \
        --metadata "/workspace/models/perception/image_classification/vit_base_patch16_224/examples/input/generated/sample_${index}.json"
    echo "[PREPARE] ${index}/3 $(basename "${image_path}")"
done

echo "[2/5] 创建干净的板端三图运行目录."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "rm -rf '${BOARD_DIR}' && mkdir -p '${BOARD_DIR}/inputs' '${BOARD_DIR}/output'"
echo "[3/5] 部署 DLA、输入和逐图运行脚本."
scp "${SSH_OPTIONS[@]}" "${MODEL_ROOT}/models/model_int8.dla" \
    "${MODEL_ROOT}/deploy/inference_demo/board_eval_loop.sh" \
    "${BOARD_HOST}:${BOARD_DIR}/"
scp "${SSH_OPTIONS[@]}" "${WORK_DIR}"/*.bin \
    "${BOARD_HOST}:${BOARD_DIR}/inputs/"

echo "[4/5] 在 Genio 720 NPU 逐张推理."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "chmod +x '${BOARD_DIR}/board_eval_loop.sh' && \
     '${BOARD_DIR}/board_eval_loop.sh' '${BOARD_DIR}/model_int8.dla' \
     '${BOARD_DIR}/inputs' '${BOARD_DIR}/output' 1"
scp "${SSH_OPTIONS[@]}" "${BOARD_HOST}:${BOARD_DIR}/output/*.bin" "${RAW_DIR}/"

echo "[5/5] 生成 Top-5 图片和 JSON."
docker exec "${CONTAINER}" python3 \
    "/workspace/models/perception/image_classification/vit_base_patch16_224/deploy/generate_examples.py" \
    --input-dir "/workspace/models/perception/image_classification/vit_base_patch16_224/examples/input/public" \
    --metadata-dir "/workspace/models/perception/image_classification/vit_base_patch16_224/examples/input/generated" \
    --npu-dir "/workspace/models/perception/image_classification/vit_base_patch16_224/examples/output/board_raw" \
    --labels "/workspace/models/perception/image_classification/vit_base_patch16_224/original/labels.txt" \
    --output-dir "/workspace/models/perception/image_classification/vit_base_patch16_224/examples/output/public" \
    --count 3
echo "[OK] ViT 三张公开图片已在 Genio 720 完成测试: ${OUTPUT_DIR}"
