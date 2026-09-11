#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT="$(git -C "${MODEL_ROOT}" rev-parse --show-toplevel)"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_ROOT="${MTK_BOARD_ROOT:-/root/hailong.he}"
readonly BOARD_MODEL_DIR="${BOARD_ROOT}/rtmpose_body2d_cpp"
readonly BOARD_DATASET="${BOARD_ROOT}/datasets/coco/val2017"
readonly COCO_ROOT="${COCO_ROOT:-/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017}"
readonly CONTAINER_COCO_ROOT="${CONTAINER_COCO_ROOT:-/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017}"
readonly RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
readonly BOARD_OUTPUT="${BOARD_MODEL_DIR}/runs/${RUN_ID}"
readonly BOARD_SMOKE="${BOARD_MODEL_DIR}/runs/${RUN_ID}_smoke"
readonly LOCAL_OUTPUT="${MODEL_ROOT}/examples/output/board_cpp_accuracy/${RUN_ID}"
readonly WORK_DIR="${MODEL_ROOT}/.eval/rtmpose_body2d"
readonly MANIFEST="${WORK_DIR}/person_detections.tsv"
readonly BINARY="${SCRIPT_DIR}/inference_demo/rtmpose_board_eval"
readonly ANNOTATIONS="${COCO_ROOT}/annotations/coco_wholebody_val_v1.0.json"
readonly CONTAINER_ANNOTATIONS="${CONTAINER_COCO_ROOT}/annotations/coco_wholebody_val_v1.0.json"
readonly DETECTIONS="${COCO_ROOT}/person_detection_results/COCO_val2017_detections_AP_H_56_person.json"
readonly EVALUATOR="${SCRIPT_DIR}/inference_demo/evaluate_coco_wholebody.py"
readonly CONTAINER_NAME="${MTK_G720_CONTAINER:-hhl_g720_8011}"
readonly CONTAINER_MODEL_ROOT="${MODEL_ROOT}"
readonly -a SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "[ERROR] EVAL_RUN_ID 只能包含字母、数字、点、下划线和连字符." >&2
    exit 1
fi
if [[ -e "${LOCAL_OUTPUT}" ]]; then
    echo "[ERROR] 本地运行目录已存在: ${LOCAL_OUTPUT}" >&2
    exit 1
fi

test -f "${MODEL_ROOT}/models/model_int8.dla"
test -f "${ANNOTATIONS}"
test -f "${DETECTIONS}"

echo "[1/9] 生成并校验 104125 个人体检测框清单."
mkdir -p "${WORK_DIR}"
python3 "${SCRIPT_DIR}/inference_demo/prepare_eval_manifest.py" \
    --detections "${DETECTIONS}" --output "${MANIFEST}"
test "$(($(wc -l < "${MANIFEST}") - 1))" -eq 104125

echo "[2/9] 交叉编译板端常驻 C++ 推理程序."
bash "${SCRIPT_DIR}/build_board_cpp.sh"

echo "[3/9] 核对板端数据集和全新运行目录."
image_count="$(ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "find '${BOARD_DATASET}/images' -maxdepth 1 -type f -name '*.jpg' | wc -l")"
test "${image_count}" -eq 5000
if ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
        "test -e '${BOARD_OUTPUT}' -o -e '${BOARD_SMOKE}'"; then
    echo "[ERROR] 板端运行目录已存在,请使用新 EVAL_RUN_ID." >&2
    exit 1
fi

echo "[4/9] 部署程序、DLA 和检测框清单."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "mkdir -p '${BOARD_MODEL_DIR}' '${BOARD_OUTPUT}' '${BOARD_SMOKE}'"
scp "${SSH_OPTIONS[@]}" "${BINARY}" "${MODEL_ROOT}/models/model_int8.dla" \
    "${MANIFEST}" "${BOARD_HOST}:${BOARD_MODEL_DIR}/"

echo "[5/9] 先执行两个检测框的板端冒烟."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "'${BOARD_MODEL_DIR}/rtmpose_board_eval' \
        --model '${BOARD_MODEL_DIR}/model_int8.dla' \
        --images '${BOARD_DATASET}/images' \
        --manifest '${BOARD_MODEL_DIR}/person_detections.tsv' \
        --output-dir '${BOARD_SMOKE}' --limit 2 --warmup 2 \
        --progress-interval 1 2>&1 | tee '${BOARD_SMOKE}/board_eval.log'"
test "$(ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "wc -l < '${BOARD_SMOKE}/predictions.jsonl'")" -eq 2

echo "[6/9] 执行 104125 个人体框的完整板端评测."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "'${BOARD_MODEL_DIR}/rtmpose_board_eval' \
        --model '${BOARD_MODEL_DIR}/model_int8.dla' \
        --images '${BOARD_DATASET}/images' \
        --manifest '${BOARD_MODEL_DIR}/person_detections.tsv' \
        --output-dir '${BOARD_OUTPUT}' --warmup 20 \
        --progress-interval 500 2>&1 | tee '${BOARD_OUTPUT}/board_eval.log'"
test "$(ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "wc -l < '${BOARD_OUTPUT}/processed_ids.txt'")" -eq 104125

echo "[7/9] 回传原始预测和板端运行证据."
mkdir -p "${LOCAL_OUTPUT}"
scp "${SSH_OPTIONS[@]}" \
    "${BOARD_HOST}:${BOARD_OUTPUT}/predictions.jsonl" \
    "${BOARD_HOST}:${BOARD_OUTPUT}/processed_ids.txt" \
    "${BOARD_HOST}:${BOARD_OUTPUT}/timing_summary.json" \
    "${BOARD_HOST}:${BOARD_OUTPUT}/board_eval.log" \
    "${LOCAL_OUTPUT}/"

echo "[8/9] 在固定 G720 容器中计算 COCO-WholeBody AP/AR."
docker exec "${CONTAINER_NAME}" python3 -c "import xtcocotools"
docker exec -w "${REPO_ROOT}" "${CONTAINER_NAME}" python3 \
    "${CONTAINER_MODEL_ROOT}/deploy/inference_demo/evaluate_coco_wholebody.py" \
    --annotations "${CONTAINER_ANNOTATIONS}" \
    --predictions "${CONTAINER_MODEL_ROOT}/examples/output/board_cpp_accuracy/${RUN_ID}/predictions.jsonl" \
    --formatted "${CONTAINER_MODEL_ROOT}/examples/output/board_cpp_accuracy/${RUN_ID}/wholebody_predictions.json" \
    --metrics "${CONTAINER_MODEL_ROOT}/examples/output/board_cpp_accuracy/${RUN_ID}/coco_wholebody_metrics.json" \
    --summary-log "${CONTAINER_MODEL_ROOT}/examples/output/board_cpp_accuracy/${RUN_ID}/coco_wholebody_summary.log"

echo "[9/9] 固化输入、输出哈希与 Git 版本."
{
    echo "schema_version=1"
    echo "run_id=${RUN_ID}"
    echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    echo "annotations_sha256=$(sha256sum "${ANNOTATIONS}" | awk '{print $1}')"
    echo "detections_sha256=$(sha256sum "${DETECTIONS}" | awk '{print $1}')"
    echo "dla_sha256=$(sha256sum "${MODEL_ROOT}/models/model_int8.dla" | awk '{print $1}')"
    echo "cpp_source_sha256=$(sha256sum "${SCRIPT_DIR}/inference_demo/rtmpose_board_eval.cpp" | awk '{print $1}')"
    echo "input_shape=1,3,256,192"
    echo "keypoints=133"
    echo "simcc_split_ratio=2.0"
    echo "score_mode=bbox_keypoint"
    echo "keypoint_score_threshold=0.2"
    echo "nms_mode=oks_nms"
    echo "nms_threshold=0.9"
} > "${LOCAL_OUTPUT}/run_inputs_manifest.txt"
cd "${LOCAL_OUTPUT}"
sha256sum predictions.jsonl processed_ids.txt timing_summary.json \
    board_eval.log wholebody_predictions.json coco_wholebody_metrics.json \
    coco_wholebody_summary.log run_inputs_manifest.txt \
    > run_outputs_sha256.txt
echo "[OK] RTMPose 板端正式精度评测完成: ${LOCAL_OUTPUT}"
