#!/usr/bin/env bash
# 单脚本两步流程: 在编译主机编译并上传,在开发板执行测试.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 编译主机配置区: 只修改等号右侧的路径或名称.板端使用上传的 board_paths.conf.
if [[ ! -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    # 模型目录: 根据本脚本的位置自动确定.
    MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

    # 1. 模型与校准数据.
    # 模型文件: 原始 PyTorch 权重.
    MODEL_WEIGHTS="${MODEL_ROOT}/models/yolov5s.pt"
    # 源码文件: Ultralytics YOLOv5 压缩包.
    SOURCE_ARCHIVE="${MODEL_ROOT}/models/yolov5-485da42.zip"
    # 补丁文件: MTK 模型转换补丁压缩包.
    PATCH_ARCHIVE="${MODEL_ROOT}/models/model_conversion_YOLOv5s_example_20240916.zip"
    # 校准数据: 编译主机和 Docker 都能访问的图片目录,必须填写.
    CALIBRATION_DIR="/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017/images"

    # 2. 产物与临时目录.
    # 模型输出目录: 转换与编译产物保存在这里.
    MODEL_OUTPUT_DIR="${MODEL_ROOT}/models"
    # 临时构建目录: 缓存和中间文件保存在仓库外.
    BUILD_WORK_DIR="/tmp/hailongcodex/$(date +%F)/yolov5s"
    # DLA 文件: 编译后的完整路径.
    OUTPUT_DLA="${MODEL_OUTPUT_DIR}/model_int8.dla"
    # C++ 输出: 编译主机上的临时程序,该目录与 Docker 临时目录各自独立.
    BOARD_BINARY="${BUILD_WORK_DIR}/yolov5s_board_eval"

    # 3. 板端地址与数据.
    # 板端地址: SSH 用户和地址.
    BOARD_HOST="root@192.168.0.92"
    # 板端部署目录: 上传模型、程序和本脚本的目录,必须填写.
    BOARD_DEPLOY_DIR="/root/hailong.he/open_models/yolov5s"
    # 板端结果目录: 保存本次测试汇总.
    BOARD_RESULTS_DIR="${BOARD_DEPLOY_DIR}/results"
    # 板端全量数据: COCO val2017 根目录,包含 images/ 和 annotations/.
    BOARD_DATASET_DIR="/root/hailong.he/datasets/coco/val2017"

    # 4. ONNX 精度数据.
    # 浮点精度图片: 编译主机和 Docker 均可访问的 COCO val2017 全量图片目录.
    FP32_IMAGES_DIR="${CALIBRATION_DIR}"
    # 浮点精度标注: 与上述 5000 张图片对应的 COCO 标注文件.
    FP32_ANNOTATIONS="${FP32_IMAGES_DIR%/images}/annotations/instances_val2017.json"

    # 5. 编译环境: 通常无需修改.
    # Docker 容器: 编译主机上的 Genio 720 编译环境.
    MTK_G720_CONTAINER="hhl_g720_8011"
    # 环境脚本: Docker 内 MTK SDK 初始化脚本.
    MTK_SETUP_SCRIPT="/opt/mtk-build/setup_container.sh"
    # 模型编译器: Docker 内 ncc-tflite 可执行文件.
    NCC_BIN="/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/bin/ncc-tflite"
    # 编译器库: Docker 内 ncc-tflite 依赖库目录.
    NCC_LIB="/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/lib"
    # C++ 工具链: 编译主机上的 AArch64 编译工具和 OpenCV 库目录.
    MTK_G720_CPP_TOOLCHAIN_ROOT="/data/users/hailong.he/data/MTKG720/cpp_toolchain"
    # Runtime 头文件: 编译主机上的 Neuron Runtime include 目录.
    MTK_NEURON_INCLUDE="/data/users/hailong.he/data/MTKG720/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/include"
    # C++ 编译器: 编译主机上的 AArch64 交叉编译命令.
    CROSS_CXX="aarch64-linux-gnu-g++"
    # SSH 选项: 首次连接接受主机密钥,之后验证保存的密钥.
    SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)
fi

if [[ -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    # 板端阶段: 使用已上传的模型、程序和路径配置执行推理.
    readonly DEPLOY_DIR="${SCRIPT_DIR}"
    source "${DEPLOY_DIR}/board_paths.conf"
    readonly MODEL="${DEPLOY_DIR}/models/model_int8.dla"
    readonly BINARY="${DEPLOY_DIR}/board/yolov5s_board_eval"
    readonly RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    readonly RESULT_DIR="${BOARD_RESULTS_DIR}/${RUN_ID}"
    readonly RUN_DIR="${RESULT_DIR}/work"
    # 在全量推理前检查少量示例的工具、清单和图片依赖.
    test -s "${SCRIPT_DIR}/examples/input/samples.json"
    test -s "${SCRIPT_DIR}/board/render_examples.py"
    python3 -c 'import cv2, numpy'

    if (( $# != 0 )); then
        echo "[ERROR] 在板端直接运行 bash run.sh,无需参数,默认执行全量测试." >&2
        exit 2
    fi
    if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "[ERROR] EVAL_RUN_ID 含有不支持的字符." >&2
        exit 2
    fi
    test -s "${MODEL}"
    test -x "${BINARY}"
    if [[ -e "${RESULT_DIR}" ]]; then
        echo "[ERROR] 结果目录已存在: ${RESULT_DIR}" >&2
        exit 2
    fi

    : "${BOARD_DATASET_DIR:?全量测试需要在第一步设置 BOARD_DATASET_DIR}"
    readonly IMAGES_DIR="${BOARD_DATASET_DIR}/images"
    readonly ANNOTATIONS="${BOARD_DATASET_DIR}/annotations/instances_val2017.json"
    test -f "${ANNOTATIONS}"
    test "$(find "${IMAGES_DIR}" -maxdepth 1 -type f -name '*.jpg' | wc -l)" -eq 5000
    python3 -c 'import pycocotools'
    mkdir -p "${RUN_DIR}"

    echo "[开发板 1/3] 在板端运行 5000 张图片的 C++ NPU 推理."
    "${BINARY}" --model "${MODEL}" --images "${IMAGES_DIR}" \
        --output-dir "${RUN_DIR}" --warmup 20 --progress-interval 50 \
        2>&1 | tee "${RUN_DIR}/board_eval.log"

    echo "[开发板 2/3] 在板端计算 mAP 并汇总 NPU 平均耗时和精度变化."
    FP32_ARGS=()
    if [[ -n "${FP32_MAP:-}" ]]; then
        FP32_ARGS=(--fp32-map "${FP32_MAP}"
            --fp32-source "${FP32_BASELINE_SOURCE:-用户提供的同协议 FP32 基准}")
    fi
    python3 "${DEPLOY_DIR}/board/evaluate_coco.py" \
        --annotations "${ANNOTATIONS}" \
        --predictions "${RUN_DIR}/predictions.json" \
        --processed-ids "${RUN_DIR}/processed_ids.txt" \
        --timings "${RUN_DIR}/timing_summary_current_run.json" \
        "${FP32_ARGS[@]}" \
        --metrics "${RESULT_DIR}/summary.json" \
        --run-id "${RUN_ID}" \
        2>&1 | tee "${RUN_DIR}/cocoeval.log"

    echo "[示例] 保存少量板端效果示例."
    python3 "${SCRIPT_DIR}/board/render_examples.py" \
        --model "yolov5s" --input-dir "${SCRIPT_DIR}/examples/input" \
        --output-dir "${RESULT_DIR}/examples/output" --work-dir "${RUN_DIR}" \
        --dataset-root "${BOARD_DATASET_DIR}" --models-dir "${SCRIPT_DIR}/models"

    echo "[开发板 3/3] 保存汇总结果."
    # 仅在评测和汇总成功后清理本次中间文件; 失败时由 set -e 保留现场.
    test -s "${RESULT_DIR}/summary.json"
    rm -rf -- "${RUN_DIR}"
    echo "[OK] 全量测试结果: ${RESULT_DIR}/summary.json"
    exit 0
fi

if (( $# != 0 )); then
    echo "[ERROR] 在编译主机直接运行 bash deploy/run.sh,无需参数." >&2
    exit 2
fi

# 编译主机阶段: 检查配置,转换模型,交叉编译并上传.
: "${CALIBRATION_DIR:?请在脚本顶部配置 CALIBRATION_DIR}"
: "${BOARD_DEPLOY_DIR:?请在脚本顶部配置 BOARD_DEPLOY_DIR}"
: "${BOARD_DATASET_DIR:?请在脚本顶部配置板端 COCO 全量数据集目录}"

if [[ ! "${BOARD_DEPLOY_DIR}" =~ ^/[A-Za-z0-9_./-]+$ ]]; then
    echo "[ERROR] BOARD_DEPLOY_DIR 必须是无空格的板端绝对路径." >&2
    exit 2
fi
test -f "${MODEL_WEIGHTS}"
test -f "${SOURCE_ARCHIVE}"
test -f "${PATCH_ARCHIVE}"
test -d "${CALIBRATION_DIR}"
test -d "${FP32_IMAGES_DIR}"
test -f "${FP32_ANNOTATIONS}"
if [[ "${MODEL_OUTPUT_DIR}" != /* || "${OUTPUT_DLA}" != /* ]]; then
    echo "[ERROR] MODEL_OUTPUT_DIR 和 OUTPUT_DLA 必须是宿主机与容器共用的绝对路径." >&2
    exit 2
fi
if [[ "${BUILD_WORK_DIR}" != /* ]]; then
    echo "[ERROR] BUILD_WORK_DIR 必须是 Docker 内仓库外的绝对路径." >&2
    exit 2
fi
if [[ "${BOARD_BINARY}" != /* || "${BOARD_RESULTS_DIR}" != /* ]] ||
        [[ "${BOARD_DATASET_DIR}" != /* ]]; then
    echo "[ERROR] C++ 程序、板端结果和板端数据集路径必须是绝对路径." >&2
    exit 2
fi

# 从配置的数据集生成示例清单,仅写入仓库外的构建目录.
mkdir -p "${BUILD_WORK_DIR}/examples"
docker exec -e PYTHONDONTWRITEBYTECODE=1 "${MTK_G720_CONTAINER}" \
    python "${MODEL_ROOT}/../../../../tools/prepare_examples.py" \
    --model "yolov5s" --dataset-root "${FP32_IMAGES_DIR%/images}" \
    --output "${BUILD_WORK_DIR}/examples/samples.json"
docker cp "${MTK_G720_CONTAINER}:${BUILD_WORK_DIR}/examples/samples.json" \
    "${BUILD_WORK_DIR}/examples/samples.json"

echo "[编译主机 1] 在 Docker 中转换模型并编译 DLA: ${OUTPUT_DLA}"
docker exec -i \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e MODEL_ROOT="${MODEL_ROOT}" \
    -e MODEL_WEIGHTS="${MODEL_WEIGHTS}" \
    -e SOURCE_ARCHIVE="${SOURCE_ARCHIVE}" \
    -e PATCH_ARCHIVE="${PATCH_ARCHIVE}" \
    -e CALIBRATION_DIR="${CALIBRATION_DIR}" \
    -e MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR}" \
    -e BUILD_WORK_DIR="${BUILD_WORK_DIR}" \
    -e OUTPUT_DLA="${OUTPUT_DLA}" \
    -e FP32_IMAGES_DIR="${FP32_IMAGES_DIR}" \
    -e FP32_ANNOTATIONS="${FP32_ANNOTATIONS}" \
    -e MTK_SETUP_SCRIPT="${MTK_SETUP_SCRIPT}" \
    -e NCC_BIN="${NCC_BIN}" \
    -e NCC_LIB="${NCC_LIB}" \
    "${MTK_G720_CONTAINER}" bash -s <<'DOCKER_BUILD'
set -euo pipefail

readonly SOURCE_DIR="${BUILD_WORK_DIR}/yolov5"
readonly SOURCE_ARCHIVE_ROOT="yolov5-485da42273839d20ea6bdaf142fd02c1027aba61"
readonly PATCH_DIR="${BUILD_WORK_DIR}/mtk_patch"
readonly PATCH_FILE="${PATCH_DIR}/Fix_yolov5_mtk_tflite_issue.patch"

test -f "${MODEL_WEIGHTS}"
test -f "${SOURCE_ARCHIVE}"
test -f "${PATCH_ARCHIVE}"
test -d "${CALIBRATION_DIR}"
test -d "${FP32_IMAGES_DIR}"
test -f "${FP32_ANNOTATIONS}"
test -f "${MTK_SETUP_SCRIPT}"
test -x "${NCC_BIN}"
test -d "${NCC_LIB}"
mkdir -p "${MODEL_OUTPUT_DIR}" "${BUILD_WORK_DIR}" "$(dirname "${OUTPUT_DLA}")"
mkdir -p "${BUILD_WORK_DIR}/tmp" "${BUILD_WORK_DIR}/cache"
export TMPDIR="${BUILD_WORK_DIR}/tmp"
export XDG_CACHE_HOME="${BUILD_WORK_DIR}/cache"
export TORCH_HOME="${BUILD_WORK_DIR}/cache/torch"
if [[ "${MODEL_WEIGHTS}" != "${MODEL_OUTPUT_DIR}/yolov5s.pt" ]]; then
    cp "${MODEL_WEIGHTS}" "${MODEL_OUTPUT_DIR}/yolov5s.pt"
fi

echo "[Docker 1/3] 展开源码并应用 MTK 补丁."
if [[ ! -d "${SOURCE_DIR}" ]]; then
    unzip -q "${SOURCE_ARCHIVE}" -d "$(dirname "${SOURCE_DIR}")"
    mv "$(dirname "${SOURCE_DIR}")/${SOURCE_ARCHIVE_ROOT}" "${SOURCE_DIR}"
fi
test -f "${SOURCE_DIR}/export.py"
mkdir -p "${PATCH_DIR}"
unzip -q -j -o "${PATCH_ARCHIVE}" -d "${PATCH_DIR}"
command -v patch >/dev/null 2>&1 || {
    echo "[ERROR] Docker 内缺少 patch 命令." >&2
    exit 1
}
if patch -d "${SOURCE_DIR}" -p1 --dry-run -R < "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "[INFO] MTK 补丁已应用."
else
    patch -d "${SOURCE_DIR}" -p1 --forward < "${PATCH_FILE}"
fi

echo "[Docker 2/3] 导出 ONNX 并量化为 INT8 TFLite."
bash "${MTK_SETUP_SCRIPT}"
cd "${SOURCE_DIR}"
python export.py --weights "${MODEL_OUTPUT_DIR}/yolov5s.pt" \
    --img-size 640 640 --batch-size 1 --device 0 --include torchscript onnx
mv -f "${MODEL_OUTPUT_DIR}/yolov5s.onnx" "${MODEL_OUTPUT_DIR}/model_fp32.onnx"
python "${MODEL_ROOT}/deploy/host/convert_int8.py" \
    --torchscript "${MODEL_OUTPUT_DIR}/yolov5s.torchscript" \
    --calibration-dir "${CALIBRATION_DIR}" \
    --output "${MODEL_OUTPUT_DIR}/model_int8.tflite"

echo "[Docker 3/3] 使用 MDLA 5.3 编译 DLA."
export LD_LIBRARY_PATH="${NCC_LIB}:${LD_LIBRARY_PATH:-}"
"${NCC_BIN}" --arch=mdla5.3 --suppress-output --disallow-bridge \
    "${MODEL_OUTPUT_DIR}/model_int8.tflite" -o "${OUTPUT_DLA}"
DOCKER_BUILD
test -s "${OUTPUT_DLA}"

echo "[ONNX] 在编译主机 Docker 中评测 FP32 全量精度."
docker exec -i -e PYTHONDONTWRITEBYTECODE=1 \
    -e MODEL_ROOT="${MODEL_ROOT}" -e MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR}" \
    -e BUILD_WORK_DIR="${BUILD_WORK_DIR}" -e FP32_IMAGES_DIR="${FP32_IMAGES_DIR}" \
    -e FP32_ANNOTATIONS="${FP32_ANNOTATIONS}" \
    "${MTK_G720_CONTAINER}" bash -s <<'ONNX_ACCURACY'
set -euo pipefail
test -d "${FP32_IMAGES_DIR}"
test -f "${FP32_ANNOTATIONS}"
export TMPDIR="${BUILD_WORK_DIR}/tmp"
export XDG_CACHE_HOME="${BUILD_WORK_DIR}/cache"
REPO_ROOT="$(cd "${MODEL_ROOT}/../../../.." && pwd)"
python "${REPO_ROOT}/tools/accuracy/yolov5s_val_coco.py" \
    --onnx "${MODEL_OUTPUT_DIR}/model_fp32.onnx" \
    --images-dir "${FP32_IMAGES_DIR}" --ann "${FP32_ANNOTATIONS}" \
    --summary "${BUILD_WORK_DIR}/fp32_accuracy/summary.json" \
    --image-size 640 --confidence 0.001 --iou 0.6 --max-det 300
ONNX_ACCURACY
# 传回实测数值; Docker 与编译主机的临时目录各自独立.
FP32_MAP="$(docker exec "${MTK_G720_CONTAINER}" python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["map_50_95"])' \
    "${BUILD_WORK_DIR}/fp32_accuracy/summary.json")"
FP32_BASELINE_SOURCE="本次 ONNX FP32 全量实测,COCO val2017,conf=0.001,IoU=0.6,max_det=300"

echo "[编译主机 2] 在编译主机 交叉编译板端 C++ 测试程序."
readonly OPENCV_SOURCE="${MTK_G720_CPP_TOOLCHAIN_ROOT}/opencv-4.9.0/opencv-4.9.0"
readonly OPENCV_BUILD="${MTK_G720_CPP_TOOLCHAIN_ROOT}/opencv-4.9.0/build-aarch64-headers"
readonly TARGET_LIBS="${MTK_G720_CPP_TOOLCHAIN_ROOT}/genio720-libs"
command -v "${CROSS_CXX}" >/dev/null 2>&1
test -f "${OPENCV_SOURCE}/modules/core/include/opencv2/core.hpp"
test -f "${OPENCV_BUILD}/opencv2/cvconfig.h"
test -f "${MTK_NEURON_INCLUDE}/neuron/api/RuntimeAPI.h"
test -f "${TARGET_LIBS}/libneuronusdk_runtime.mtk.so.8"
mkdir -p "$(dirname "${BOARD_BINARY}")"
"${CROSS_CXX}" -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic \
    -I"${OPENCV_BUILD}" \
    -I"${OPENCV_SOURCE}/modules/core/include" \
    -I"${OPENCV_SOURCE}/modules/imgproc/include" \
    -I"${OPENCV_SOURCE}/modules/imgcodecs/include" \
    -I"${MTK_NEURON_INCLUDE}" \
    "${SCRIPT_DIR}/board/yolov5s_board_eval.cpp" \
    "${TARGET_LIBS}/libneuronusdk_runtime.mtk.so.8" \
    "${TARGET_LIBS}/libopencv_imgcodecs.so.409" \
    "${TARGET_LIBS}/libopencv_imgproc.so.409" \
    "${TARGET_LIBS}/libopencv_core.so.409" \
    -Wl,--allow-shlib-undefined -pthread -ldl -o "${BOARD_BINARY}"
file "${BOARD_BINARY}"
test -s "${BOARD_BINARY}"

echo "[编译主机 3] 创建板端目录并上传 DLA、程序和评测代码."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "mkdir -p '${BOARD_DEPLOY_DIR}/models' '${BOARD_DEPLOY_DIR}/board'"
scp "${SSH_OPTIONS[@]}" "${OUTPUT_DLA}" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/models/model_int8.dla"
scp "${SSH_OPTIONS[@]}" "${BOARD_BINARY}" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/board/yolov5s_board_eval"
scp "${SSH_OPTIONS[@]}" "${SCRIPT_DIR}/run.sh" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/run.sh"
scp "${SSH_OPTIONS[@]}" "${SCRIPT_DIR}/board/evaluate_coco.py" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/board/evaluate_coco.py"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "chmod 755 '${BOARD_DEPLOY_DIR}/board/yolov5s_board_eval'"

echo "[编译主机 4] 写入板端数据集和结果路径."
# 上传统一可视化工具和少量示例输入,不增加 Shell 入口.
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "mkdir -p '${BOARD_DEPLOY_DIR}/examples/input'"
scp "${SSH_OPTIONS[@]}" "${MODEL_ROOT}/../../../../tools/render_examples.py" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/board/render_examples.py"
scp "${SSH_OPTIONS[@]}" "${BUILD_WORK_DIR}/examples/samples.json" \
    "${MODEL_ROOT}/examples/input"/sample_* \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/examples/input/"
printf 'BOARD_DATASET_DIR=%q\nBOARD_RESULTS_DIR=%q\nFP32_MAP=%q\nFP32_BASELINE_SOURCE=%q\n' \
    "${BOARD_DATASET_DIR}" "${BOARD_RESULTS_DIR}" "${FP32_MAP}" "${FP32_BASELINE_SOURCE}" |
    ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
        "cat > '${BOARD_DEPLOY_DIR}/board_paths.conf'"
echo "[OK] DLA: ${OUTPUT_DLA}"
echo "[OK] 板端文件: ${BOARD_HOST}:${BOARD_DEPLOY_DIR}"
echo "[NEXT] 在板端运行: bash '${BOARD_DEPLOY_DIR}/run.sh'"
