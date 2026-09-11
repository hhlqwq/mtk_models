#!/usr/bin/env bash

set -euo pipefail

readonly REPO_ROOT="/data/users/hailong.he/github/mtk_models"
readonly MODEL_ROOT="${REPO_ROOT}/models/perception/image_classification/vit_base_patch16_224"
readonly RUN_ID="${VIT_EXAMPLE_RUN_ID:-20260910_vit_torchvision_v2}"
readonly RUN_ROOT="${REPO_ROOT}/.eval/vit_base_patch16_224/runs/${RUN_ID}"
readonly CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
readonly DATASET="/data/users/hailong.he/nas_smb/Datasets/open_source/raw/ILSVRC2012/val"

test -f "${RUN_ROOT}/board.done"
echo "[1/2] 从正式板端运行 ${RUN_ID} 提取五份 NPU 输出."
docker exec "${CONTAINER}" python3 \
    "/workspace/models/perception/image_classification/vit_base_patch16_224/deploy/generate_examples.py" \
    --manifest "/workspace/.eval/vit_base_patch16_224/runs/${RUN_ID}/manifest.jsonl" \
    --npu-dir "/workspace/.eval/vit_base_patch16_224/runs/${RUN_ID}/npu_outputs" \
    --images-dir "${DATASET}" \
    --metadata "/workspace/models/perception/image_classification/vit_base_patch16_224/examples/input/input_metadata.json" \
    --labels "/workspace/models/perception/image_classification/vit_base_patch16_224/original/labels.txt" \
    --input-dir "/workspace/models/perception/image_classification/vit_base_patch16_224/examples/input/generated" \
    --output-dir "/workspace/models/perception/image_classification/vit_base_patch16_224/examples/output/generated" \
    --count 5
echo "[2/2] ViT 五图示例生成完成: ${MODEL_ROOT}/examples/output/generated"
