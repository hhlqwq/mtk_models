#!/usr/bin/env bash
# RTMPose 单脚本流程: 89 编译上传,92 板端执行 WholeBody 全量测试.

set -euo pipefail
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 用户配置: 留空的可选项使用仓库内默认路径.
if [[ ! -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    # 待量化 ONNX: 留空则使用 models/model_mtk_compatible.onnx.
    MODEL_ONNX=""
    # INT8 校准图片目录.
    CALIBRATION_IMAGES=""
    # INT8 校准标注文件.
    CALIBRATION_ANNOTATIONS=""
    # 模型输出目录: 留空则使用 models/.
    MODEL_OUTPUT_DIR=""
    # 板端 COCO-WholeBody 数据集目录.
    BOARD_DATASET_DIR=""
    # 板端部署目录.
    BOARD_DEPLOY_DIR=""
    # 板端结果目录: 留空则位于部署目录下.
    BOARD_RESULTS_DIR=""
    # 板端 SSH 用户和地址.
    BOARD_HOST="root@192.168.0.92"
fi

# 板端阶段: 完整处理 5000 张图片和 104125 个人体检测框.
if [[ -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    source "${SCRIPT_DIR}/board_paths.conf"
    readonly RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    readonly RUN_DIR="${BOARD_RESULTS_DIR}/${RUN_ID}"
    readonly DETECTIONS="${BOARD_DATASET_DIR}/person_detection_results/COCO_val2017_detections_AP_H_56_person.json"
    readonly ANNOTATIONS="${BOARD_DATASET_DIR}/annotations/coco_wholebody_val_v1.0.json"
    test -s "${SCRIPT_DIR}/model_int8.dla"
    test -s "${DETECTIONS}"
    test -s "${ANNOTATIONS}"
    test "$(find "${BOARD_DATASET_DIR}/images" -maxdepth 1 -type f -name '*.jpg' | wc -l)" -eq 5000
    python3 -c 'import numpy, xtcocotools'
    test ! -e "${RUN_DIR}"
    mkdir -p "${RUN_DIR}/report"

    echo "[1/4] 生成并核对 104125 个检测框清单."
    "${SCRIPT_DIR}/prepare_eval_manifest" --detections "${DETECTIONS}" \
        --output "${RUN_DIR}/person_detections.tsv"
    test "$(($(wc -l < "${RUN_DIR}/person_detections.tsv") - 1))" -eq 104125

    echo "[2/4] 在板端执行 C++ NPU 推理."
    "${SCRIPT_DIR}/rtmpose_board_eval" \
        --model "${SCRIPT_DIR}/model_int8.dla" \
        --images "${BOARD_DATASET_DIR}/images" \
        --manifest "${RUN_DIR}/person_detections.tsv" \
        --output-dir "${RUN_DIR}" --warmup 20 --progress-interval 500 \
        2>&1 | tee "${RUN_DIR}/board_eval.log"
    test "$(wc -l < "${RUN_DIR}/processed_ids.txt")" -eq 104125

    echo "[3/4] 计算 WholeBody AP/AR."
    python3 "${SCRIPT_DIR}/evaluate_coco_wholebody.py" \
        --annotations "${ANNOTATIONS}" \
        --predictions "${RUN_DIR}/predictions.jsonl" \
        --formatted "${RUN_DIR}/wholebody_predictions.json" \
        --metrics "${RUN_DIR}/coco_wholebody_metrics.json" \
        --summary-log "${RUN_DIR}/coco_wholebody_summary.log" \
        2>&1 | tee "${RUN_DIR}/cocoeval.log"

    echo "[4/4] 整理报告."
    cp "${RUN_DIR}/timing_summary.json" \
        "${RUN_DIR}/coco_wholebody_metrics.json" \
        "${RUN_DIR}/coco_wholebody_summary.log" \
        "${RUN_DIR}/board_eval.log" "${RUN_DIR}/cocoeval.log" \
        "${RUN_DIR}/report/"
    python3 - "${RUN_DIR}/coco_wholebody_metrics.json" \
        "${RUN_DIR}/report/summary.json" "${RUN_ID}" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if metrics.get("protocol", {}).get("detections_before_nms") != 104125:
    raise SystemExit("[ERROR] WholeBody 检测框覆盖不完整.")
summary = {
    "status": "complete", "model": "rtmpose_body2d",
    "run_id": sys.argv[3], "dataset": "coco_wholebody_val2017",
    "detections": 104125, "accuracy": metrics["metrics"],
}
Path(sys.argv[2]).write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
    uname -a > "${RUN_DIR}/report/system.txt"
    echo "[OK] 报告: ${RUN_DIR}/report"
    exit 0
fi

if (( $# != 0 )); then
    echo "[ERROR] 在 89 直接运行 bash deploy/run.sh,无需参数." >&2
    exit 2
fi

# 路径配置: 模型、校准集、生成目录、板端数据和工具链均可覆盖.
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly MODEL_ONNX="${MODEL_ONNX:-${MODEL_ROOT}/models/model_mtk_compatible.onnx}"
readonly CALIBRATION_IMAGES="${CALIBRATION_IMAGES:?请指定校准图片目录}"
readonly CALIBRATION_ANNOTATIONS="${CALIBRATION_ANNOTATIONS:?请指定 COCO 校准标注文件}"
readonly MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-${MODEL_ROOT}/models}"
readonly BOARD_DATASET_DIR="${BOARD_DATASET_DIR:?请指定板端 COCO WholeBody 数据集目录}"
readonly BOARD_DEPLOY_DIR="${BOARD_DEPLOY_DIR:?请指定 BOARD_DEPLOY_DIR}"
readonly BOARD_RESULTS_DIR="${BOARD_RESULTS_DIR:-${BOARD_DEPLOY_DIR}/results}"
readonly BOARD_HOST="${BOARD_HOST:-root@192.168.0.92}"
readonly CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
readonly MTK_SETUP_SCRIPT="${MTK_SETUP_SCRIPT:-/opt/mtk-build/setup_container.sh}"
readonly NCC_ROOT="${NCC_ROOT:-/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host}"
readonly TOOLCHAIN_ROOT="${MTK_G720_CPP_TOOLCHAIN_ROOT:-/data/users/hailong.he/data/MTKG720/cpp_toolchain}"
readonly NEURON_INCLUDE="${MTK_NEURON_INCLUDE:-/data/users/hailong.he/data/MTKG720/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/include}"
readonly CXX="${CROSS_CXX:-aarch64-linux-gnu-g++}"
readonly SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

test -s "${MODEL_ONNX}"
test -d "${CALIBRATION_IMAGES}"
test -s "${CALIBRATION_ANNOTATIONS}"
if [[ ! "${BOARD_DEPLOY_DIR}" =~ ^/[A-Za-z0-9_./-]+$ ]]; then
    echo "[ERROR] BOARD_DEPLOY_DIR 必须是无空格的板端绝对路径." >&2
    exit 2
fi

echo "[1/3] 在 Docker 中量化并编译 DLA."
docker exec -i -e MODEL_ROOT="${MODEL_ROOT}" -e MODEL_ONNX="${MODEL_ONNX}" \
    -e CALIBRATION_IMAGES="${CALIBRATION_IMAGES}" \
    -e CALIBRATION_ANNOTATIONS="${CALIBRATION_ANNOTATIONS}" \
    -e MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR}" \
    -e MTK_SETUP_SCRIPT="${MTK_SETUP_SCRIPT}" -e NCC_ROOT="${NCC_ROOT}" \
    "${CONTAINER}" bash -s <<'DOCKER_BUILD'
set -euo pipefail
mkdir -p "${MODEL_OUTPUT_DIR}"
bash "${MTK_SETUP_SCRIPT}"
python "${MODEL_ROOT}/deploy/convert_int8.py" \
    --onnx "${MODEL_ONNX}" --image-dir "${CALIBRATION_IMAGES}" \
    --annotations "${CALIBRATION_ANNOTATIONS}" \
    --output "${MODEL_OUTPUT_DIR}/model_int8.tflite"
export LD_LIBRARY_PATH="${NCC_ROOT}/lib:${LD_LIBRARY_PATH:-}"
"${NCC_ROOT}/bin/ncc-tflite" --arch=mdla5.3 \
    --suppress-output --disallow-bridge \
    "${MODEL_OUTPUT_DIR}/model_int8.tflite" -o "${MODEL_OUTPUT_DIR}/model_int8.dla"
DOCKER_BUILD
test -s "${MODEL_OUTPUT_DIR}/model_int8.dla"

echo "[2/3] 在 89 交叉编译推理和框清单 C++ 程序."
readonly OPENCV_SOURCE="${TOOLCHAIN_ROOT}/opencv-4.9.0/opencv-4.9.0"
readonly OPENCV_BUILD="${TOOLCHAIN_ROOT}/opencv-4.9.0/build-aarch64-headers"
readonly TARGET_LIBS="${TOOLCHAIN_ROOT}/genio720-libs"
command -v "${CXX}" >/dev/null 2>&1
"${CXX}" -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic \
    -I"${OPENCV_BUILD}" -I"${OPENCV_SOURCE}/modules/core/include" \
    -I"${OPENCV_SOURCE}/modules/imgproc/include" \
    -I"${OPENCV_SOURCE}/modules/imgcodecs/include" -I"${NEURON_INCLUDE}" \
    "${SCRIPT_DIR}/inference_demo/rtmpose_board_eval.cpp" \
    "${TARGET_LIBS}/libneuronusdk_runtime.mtk.so.8" \
    "${TARGET_LIBS}/libopencv_imgcodecs.so.409" \
    "${TARGET_LIBS}/libopencv_imgproc.so.409" \
    "${TARGET_LIBS}/libopencv_core.so.409" \
    -Wl,--allow-shlib-undefined -pthread -ldl \
    -o "${SCRIPT_DIR}/inference_demo/rtmpose_board_eval"
"${CXX}" -std=c++20 -O2 -DNDEBUG -Wall -Wextra -Wpedantic \
    -I"${OPENCV_BUILD}" -I"${OPENCV_SOURCE}/modules/core/include" \
    "${SCRIPT_DIR}/inference_demo/prepare_eval_manifest.cpp" \
    "${TARGET_LIBS}/libopencv_core.so.409" \
    -Wl,--allow-shlib-undefined -pthread -ldl \
    -o "${SCRIPT_DIR}/inference_demo/prepare_eval_manifest"
file "${SCRIPT_DIR}/inference_demo/rtmpose_board_eval" \
    "${SCRIPT_DIR}/inference_demo/prepare_eval_manifest"

echo "[3/3] 上传模型、程序、评测代码和路径配置到板端."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" "mkdir -p '${BOARD_DEPLOY_DIR}'"
scp "${SSH_OPTIONS[@]}" "${MODEL_OUTPUT_DIR}/model_int8.dla" \
    "${SCRIPT_DIR}/inference_demo/rtmpose_board_eval" \
    "${SCRIPT_DIR}/inference_demo/prepare_eval_manifest" \
    "${SCRIPT_DIR}/inference_demo/evaluate_coco_wholebody.py" \
    "${SCRIPT_DIR}/run.sh" "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "chmod 755 '${BOARD_DEPLOY_DIR}/rtmpose_board_eval' '${BOARD_DEPLOY_DIR}/prepare_eval_manifest'"
printf 'BOARD_DATASET_DIR=%q\nBOARD_RESULTS_DIR=%q\n' \
    "${BOARD_DATASET_DIR}" "${BOARD_RESULTS_DIR}" |
    ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
        "cat > '${BOARD_DEPLOY_DIR}/board_paths.conf'"
echo "[OK] 在板端运行: bash '${BOARD_DEPLOY_DIR}/run.sh'"
