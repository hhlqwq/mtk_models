#!/usr/bin/env bash
# 两步交付: 编译主机转换并上传,开发板执行全量测试.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    source "${SCRIPT_DIR}/board_paths.conf"
    RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    [[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
    RESULT_DIR="${BOARD_RESULTS_DIR}/${RUN_ID}"
    RUN_DIR="${RESULT_DIR}/work"
    test ! -e "${RESULT_DIR}"
    test -s "${BOARD_DATASET_DIR}/annotations/instances_val2017.json"
    test "$(find "${BOARD_DATASET_DIR}/images" -maxdepth 1 -name '*.jpg' | wc -l)" -eq 5000
    python3 -c 'import cv2, numpy, pycocotools'
    mkdir -p "${RESULT_DIR}"
    echo "[开发板 1/3] 预热并执行 COCO val2017 全量 C++ NPU 推理."
    "${SCRIPT_DIR}/board/yolov8n_board_eval" \
        --model "${SCRIPT_DIR}/models/model_int8.dla" \
        --config "${SCRIPT_DIR}/models/runtime_config.csv" \
        --images "${BOARD_DATASET_DIR}/images" --output-dir "${RUN_DIR}" \
        --warmup 20 --confidence 0.001 --iou 0.6 --max-det 300 \
        2>&1 | tee "${RESULT_DIR}/board_eval.log"
    echo "[开发板 2/3] 校验 5000 张覆盖并计算精度."
    python3 "${SCRIPT_DIR}/board/evaluate_coco.py" \
        --annotations "${BOARD_DATASET_DIR}/annotations/instances_val2017.json" \
        --predictions "${RUN_DIR}/predictions.json" \
        --processed-ids "${RUN_DIR}/processed_ids.txt" \
        --timings "${RUN_DIR}/timing_summary_current_run.json" \
        --fp32-map "${FP32_MAP}" --fp32-source "${FP32_SOURCE}" \
        --metrics "${RESULT_DIR}/summary.json" --run-id "${RUN_ID}" \
        2>&1 | tee "${RESULT_DIR}/cocoeval.log"
    echo "[开发板 3/3] 绘制实际检测结果,保留本次运行证据."
    python3 "${SCRIPT_DIR}/board/render_examples.py" \
        --input-dir "${SCRIPT_DIR}/examples/input" \
        --output-dir "${RESULT_DIR}/examples/output" --work-dir "${RUN_DIR}"
    sha256sum "${SCRIPT_DIR}/models/"* "${SCRIPT_DIR}/board/yolov8n_board_eval" \
        > "${RESULT_DIR}/SHA256SUMS"
    uname -a > "${RESULT_DIR}/system.txt"
    echo "[OK] 板端汇总: ${RESULT_DIR}/summary.json"
    exit 0
fi

# 编译主机配置区: 默认复用现有 Genio 720 工具链和 COCO 数据.
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_WEIGHTS="${YOLOV8_WEIGHTS:-${MODEL_ROOT}/models/yolov8n.pt}"
MODEL_OUTPUT_DIR="${MODEL_ROOT}/models"
CALIBRATION_DIR="${YOLOV8_CALIBRATION_DIR:-/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017/images}"
DATASET_DIR="${CALIBRATION_DIR%/images}"
BUILD_WORK_DIR="/tmp/hailongcodex/$(date +%F)/yolov8n"
CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
NCC_ROOT="/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host"
TOOLCHAIN_ROOT="/data/users/hailong.he/data/MTKG720/cpp_toolchain"
NEURON_INCLUDE="/data/users/hailong.he/data/MTKG720/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/include"
BOARD_HOST="root@192.168.0.92"
BOARD_DEPLOY_DIR="/root/hailong.he/open_models/yolov8n"
BOARD_DATASET_DIR="/root/hailong.he/datasets/coco/val2017"
SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)
test -s "${MODEL_WEIGHTS}"
test -s "${DATASET_DIR}/annotations/instances_val2017.json"
echo '31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95 '"${MODEL_WEIGHTS}" | sha256sum -c -
mkdir -p "${BUILD_WORK_DIR}"

echo "[编译主机 1/4] 离线导出、量化并编译 MDLA 5.3 模型."
docker exec -i -e MODEL_ROOT="${MODEL_ROOT}" -e MODEL_WEIGHTS="${MODEL_WEIGHTS}" \
    -e MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR}" -e CALIBRATION_DIR="${CALIBRATION_DIR}" \
    -e BUILD_WORK_DIR="${BUILD_WORK_DIR}" -e NCC_ROOT="${NCC_ROOT}" \
    -e PYTHONDONTWRITEBYTECODE=1 "${CONTAINER}" bash -s <<'BUILD'
set -euo pipefail
mkdir -p "${BUILD_WORK_DIR}/tmp" "${BUILD_WORK_DIR}/cache"
export TMPDIR="${BUILD_WORK_DIR}/tmp" XDG_CACHE_HOME="${BUILD_WORK_DIR}/cache"
cd "${BUILD_WORK_DIR}"
python "${MODEL_ROOT}/deploy/host/export_model.py" --weights "${MODEL_WEIGHTS}" \
    --image "${CALIBRATION_DIR}/000000000139.jpg" --output-dir "${MODEL_OUTPUT_DIR}"
python "${MODEL_ROOT}/deploy/host/convert_int8.py" \
    --onnx "${MODEL_OUTPUT_DIR}/model_fp32.onnx" --calibration-dir "${CALIBRATION_DIR}" \
    --samples 100 --output "${MODEL_OUTPUT_DIR}/model_int8.tflite"
export LD_LIBRARY_PATH="${NCC_ROOT}/lib:${LD_LIBRARY_PATH:-}"
"${NCC_ROOT}/bin/ncc-tflite" --arch=mdla5.3 --suppress-output --disallow-bridge \
    "${MODEL_OUTPUT_DIR}/model_int8.tflite" -o "${MODEL_OUTPUT_DIR}/model_int8.dla"
sha256sum "${MODEL_OUTPUT_DIR}/yolov8n.pt" "${MODEL_OUTPUT_DIR}/model_fp32.onnx" \
    "${MODEL_OUTPUT_DIR}/model_int8.tflite" "${MODEL_OUTPUT_DIR}/model_int8.dla" \
    "${MODEL_OUTPUT_DIR}/runtime_config.csv" > "${MODEL_OUTPUT_DIR}/SHA256SUMS"
BUILD

echo "[编译主机 2/4] 交叉编译 C++ 板端评测程序."
OPENCV_SOURCE="${TOOLCHAIN_ROOT}/opencv-4.9.0/opencv-4.9.0"
OPENCV_BUILD="${TOOLCHAIN_ROOT}/opencv-4.9.0/build-aarch64-headers"
TARGET_LIBS="${TOOLCHAIN_ROOT}/genio720-libs"
aarch64-linux-gnu-g++ -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic \
    -I"${OPENCV_BUILD}" -I"${OPENCV_SOURCE}/modules/core/include" \
    -I"${OPENCV_SOURCE}/modules/imgproc/include" \
    -I"${OPENCV_SOURCE}/modules/imgcodecs/include" -I"${NEURON_INCLUDE}" \
    "${SCRIPT_DIR}/board/yolov8n_board_eval.cpp" \
    "${TARGET_LIBS}/libneuronusdk_runtime.mtk.so.8" \
    "${TARGET_LIBS}/libopencv_imgcodecs.so.409" \
    "${TARGET_LIBS}/libopencv_imgproc.so.409" "${TARGET_LIBS}/libopencv_core.so.409" \
    -Wl,--allow-shlib-undefined -pthread -ldl -o "${BUILD_WORK_DIR}/yolov8n_board_eval"

echo "[编译主机 3/4] 评测同协议 PyTorch 和 ONNX 浮点基准."
for backend in pytorch onnx; do
    if [[ "${backend}" == pytorch ]]; then
        INPUT_ARGS=(--weights "${MODEL_WEIGHTS}")
    else
        INPUT_ARGS=(--onnx "${MODEL_OUTPUT_DIR}/model_fp32.onnx")
    fi
    docker exec -e PYTHONDONTWRITEBYTECODE=1 "${CONTAINER}" \
        python "${SCRIPT_DIR}/host/evaluate_fp32.py" "${INPUT_ARGS[@]}" \
        --dataset-root "${DATASET_DIR}" \
        --output "${MODEL_ROOT}/results/${backend}_summary.json"
done
FP32_MAP="$(docker exec "${CONTAINER}" python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["map_50_95"])' \
    "${MODEL_ROOT}/results/onnx_summary.json")"
FP32_SOURCE="本次 ONNX FP32 全量实测,COCO val2017,conf=0.001,IoU=0.6,max_det=300,单最佳类别"

echo "[编译主机 4/4] 上传模型、程序、示例和板端配置."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "mkdir -p '${BOARD_DEPLOY_DIR}/models' '${BOARD_DEPLOY_DIR}/board' '${BOARD_DEPLOY_DIR}/examples/input'"
scp "${SSH_OPTIONS[@]}" "${MODEL_OUTPUT_DIR}/model_int8.dla" \
    "${MODEL_OUTPUT_DIR}/runtime_config.csv" "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/models/"
scp "${SSH_OPTIONS[@]}" "${BUILD_WORK_DIR}/yolov8n_board_eval" \
    "${SCRIPT_DIR}/board/evaluate_coco.py" "${SCRIPT_DIR}/board/render_examples.py" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/board/"
scp "${SSH_OPTIONS[@]}" "${SCRIPT_DIR}/run.sh" "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/"
scp "${SSH_OPTIONS[@]}" "${MODEL_ROOT}/examples/input/"* \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/examples/input/"
printf 'BOARD_DATASET_DIR=%q\nBOARD_RESULTS_DIR=%q\nFP32_MAP=%q\nFP32_SOURCE=%q\n' \
    "${BOARD_DATASET_DIR}" "${BOARD_DEPLOY_DIR}/results" "${FP32_MAP}" "${FP32_SOURCE}" |
    ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" "cat > '${BOARD_DEPLOY_DIR}/board_paths.conf'"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" "chmod 755 '${BOARD_DEPLOY_DIR}/board/yolov8n_board_eval'"
echo "[NEXT] 开发板执行: bash '${BOARD_DEPLOY_DIR}/run.sh'"
