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
    # 官方 ONNX.
    SOURCE_ONNX="${MODEL_ROOT}/models/model_fp32.onnx"

    # 2. 产物与临时目录.
    # 模型输出目录: 转换与编译产物保存在这里.
    MODEL_OUTPUT_DIR="${MODEL_ROOT}/models"
    # 临时构建目录: 缓存和中间文件保存在仓库外.
    BUILD_WORK_DIR="/tmp/hailongcodex/$(date +%F)/yoloworld_xl"
    # C++ 输出: 编译主机交叉编译生成的临时程序完整路径.
    BOARD_BINARY_OUTPUT="${BUILD_WORK_DIR}/yoloworld_board_eval"

    # 3. 板端地址与数据.
    # 板端地址: SSH 用户和地址.
    BOARD_HOST="root@192.168.0.92"
    # 板端部署目录: 上传模型、程序和本脚本的目录,必须填写.
    BOARD_DEPLOY_DIR="/root/hailong.he/open_models/yoloworld_xl"
    # 板端结果目录: 保存本次测试汇总.
    BOARD_RESULTS_DIR="${BOARD_DEPLOY_DIR}/results"
    # 板端 COCO val2017 数据集目录.
    BOARD_DATASET_DIR="/root/hailong.he/datasets/coco/val2017"
    # 板端 ONNX Runtime 动态库路径.
    ORT_RUNTIME_LIB="/usr/lib/libonnxruntime.so.1.20.2"

    # 4. ONNX 精度数据.
    # 浮点精度数据: 编译主机与 Docker 可访问的数据集根目录,必须填写.
    ONNX_DATASET_DIR="/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017"

    # 5. 编译环境: 通常无需修改.
    # Docker 容器: 编译主机上的 Genio 720 编译环境.
    MTK_G720_CONTAINER="hhl_g720_8011"
    # C++ 工具链: 编译主机上的 AArch64 编译工具和 OpenCV 库目录.
    MTK_G720_CPP_TOOLCHAIN_ROOT="/data/users/hailong.he/data/MTKG720/cpp_toolchain"
    # C++ 编译器: 编译主机上的 AArch64 交叉编译命令.
    CROSS_CXX="aarch64-linux-gnu-g++"
    # Runtime 头文件: 编译主机上的 ONNX Runtime C API 头文件.
    ORT_HEADER="${MODEL_ROOT}/../../../../tools/third_party/onnxruntime/onnxruntime_c_api.h"
    # SSH 选项: 首次连接接受主机密钥,之后验证保存的密钥.
    SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)
fi

# 板端阶段: 此模型由 ONNX Runtime Neuron EP 在线编译,没有离线 DLA.
if [[ -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    source "${SCRIPT_DIR}/board_paths.conf"
    readonly RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    readonly RESULT_DIR="${BOARD_RESULTS_DIR}/${RUN_ID}"
    readonly RUN_DIR="${RESULT_DIR}/work"
    readonly ANNOTATIONS="${BOARD_DATASET_DIR}/annotations/instances_val2017.json"
    test -s "${SCRIPT_DIR}/models/model_fp32_pure_npu.onnx"
    test -s "${ANNOTATIONS}"
    test "$(find "${BOARD_DATASET_DIR}/images" -maxdepth 1 -type f -name '*.jpg' | wc -l)" -eq 5000
    test -s "${ORT_RUNTIME_LIB}"
    python3 -c 'import pycocotools'
    test ! -e "${RESULT_DIR}"
    mkdir -p "${RUN_DIR}/raw" "${RUN_DIR}/report"

    echo "[开发板 1/3] 在板端执行 5000 张 Neuron EP 推理."
    "${SCRIPT_DIR}/board/yoloworld_board_eval" \
        --model "${SCRIPT_DIR}/models/model_fp32_pure_npu.onnx" \
        --images "${BOARD_DATASET_DIR}/images" \
        --output-dir "${RUN_DIR}/raw" \
        2>&1 | tee "${RUN_DIR}/board_eval.log"
    echo "[开发板 2/3] 计算 COCO bbox AP."
    python3 "${SCRIPT_DIR}/board/evaluate_full_coco.py" \
        --annotations "${ANNOTATIONS}" \
        --results "${RUN_DIR}/raw/results.jsonl" \
        --profile "$(cat "${RUN_DIR}/raw/profile_path.txt")" \
        --output-dir "${RUN_DIR}/report" --run-id "${RUN_ID}" \
        2>&1 | tee "${RUN_DIR}/cocoeval.log"
    # 所有原始数据仅在本次 work 下生成; 汇总成功后由工具清理.
    python3 "${SCRIPT_DIR}/board/summarize_board_result.py" \
        --model "yoloworld_xl" --work-dir "${RUN_DIR}" \
        --output "${RESULT_DIR}/summary.json" --run-id "${RUN_ID}" \
        --reference "${REFERENCE_ACCURACY:-}" \
        --reference-source "${REFERENCE_SOURCE:-用户提供的同协议参考基准}"
    exit 0
fi

if (( $# != 0 )); then
    echo "[ERROR] 在编译主机直接运行 bash deploy/run.sh,无需参数." >&2
    exit 2
fi

# 编译主机阶段: 检查配置,转换模型,交叉编译并上传.
: "${ONNX_DATASET_DIR:?请在脚本顶部指定编译主机的全量精度数据集}"
test -d "${ONNX_DATASET_DIR}"
if [[ "${ONNX_DATASET_DIR}" != /* ]]; then
    echo "[ERROR] ONNX_DATASET_DIR 必须是编译主机与 Docker 共用的绝对路径." >&2
    exit 2
fi
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
: "${BOARD_DATASET_DIR:?请指定板端 COCO 数据集目录}"
: "${BOARD_DEPLOY_DIR:?请指定 BOARD_DEPLOY_DIR}"
readonly CONTAINER="${MTK_G720_CONTAINER}"
readonly TOOLCHAIN_ROOT="${MTK_G720_CPP_TOOLCHAIN_ROOT}"
readonly CXX="${CROSS_CXX}"
readonly BINARY_OUTPUT="${BOARD_BINARY_OUTPUT}"

mkdir -p "${BUILD_WORK_DIR}/tmp"
export TMPDIR="${BUILD_WORK_DIR}/tmp"

test -s "${SOURCE_ONNX}"
test -s "${ORT_HEADER}"
if [[ ! "${BOARD_DEPLOY_DIR}" =~ ^/[A-Za-z0-9_./-]+$ ]]; then
    echo "[ERROR] BOARD_DEPLOY_DIR 必须是无空格的板端绝对路径." >&2
    exit 2
fi

echo "[编译主机 1] 在 Docker 中准备兼容 ONNX 并验证等价性."
docker exec -i -e BUILD_WORK_DIR="${BUILD_WORK_DIR}" -e PYTHONDONTWRITEBYTECODE=1 -e MODEL_ROOT="${MODEL_ROOT}" \
    -e SOURCE_ONNX="${SOURCE_ONNX}" \
    -e MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR}" \
    "${CONTAINER}" bash -s <<'DOCKER_BUILD'
set -euo pipefail
mkdir -p "${MODEL_OUTPUT_DIR}" "${BUILD_WORK_DIR}/tmp" "${BUILD_WORK_DIR}/cache"
export TMPDIR="${BUILD_WORK_DIR}/tmp"
export XDG_CACHE_HOME="${BUILD_WORK_DIR}/cache"
export TORCH_HOME="${BUILD_WORK_DIR}/cache/torch"
cd "${BUILD_WORK_DIR}"
python3 "${MODEL_ROOT}/deploy/host/prepare_onnx.py" \
    --input "${SOURCE_ONNX}" \
    --output "${MODEL_OUTPUT_DIR}/model_fp32_opset13.onnx" \
    --raw-output "${MODEL_OUTPUT_DIR}/model_fp32_raw.onnx"
python3 "${MODEL_ROOT}/deploy/host/prepare_pure_npu_onnx.py" \
    --input "${MODEL_OUTPUT_DIR}/model_fp32_raw.onnx" \
    --output "${MODEL_OUTPUT_DIR}/model_fp32_pure_npu.onnx"
python3 "${MODEL_ROOT}/deploy/host/verify_onnx_equivalence.py" \
    --source "${SOURCE_ONNX}" \
    --converted "${MODEL_OUTPUT_DIR}/model_fp32_opset13.onnx" \
    --raw "${MODEL_OUTPUT_DIR}/model_fp32_raw.onnx" \
    --report "${BUILD_WORK_DIR}/onnx_equivalence.json"
DOCKER_BUILD

echo "[ONNX] 在编译主机 Docker 中评测全量浮点精度."
docker exec -i -e PYTHONDONTWRITEBYTECODE=1 \
    -e MODEL_ROOT="${MODEL_ROOT}" -e ONNX_DATASET_DIR="${ONNX_DATASET_DIR}" \
    -e ONNX_REFERENCE="${MODEL_OUTPUT_DIR}/model_fp32_raw.onnx" -e BUILD_WORK_DIR="${BUILD_WORK_DIR}" \
    "${CONTAINER}" bash -s <<'ONNX_ACCURACY'
set -euo pipefail
test -d "${ONNX_DATASET_DIR}"
export TMPDIR="${BUILD_WORK_DIR}/tmp"
export XDG_CACHE_HOME="${BUILD_WORK_DIR}/cache"
export TORCH_HOME="${BUILD_WORK_DIR}/cache/torch"
REPO_ROOT="$(cd "${MODEL_ROOT}/../../../.." && pwd)"
python "${REPO_ROOT}/tools/accuracy/evaluate_onnx.py" \
    --model "yoloworld_xl" --onnx "${ONNX_REFERENCE}" \
    --dataset-root "${ONNX_DATASET_DIR}" \
    --output "${BUILD_WORK_DIR}/onnx_accuracy/summary.json"
ONNX_ACCURACY
# 将本次实测精度写入板端配置,不传递主机预测、耗时或内存.
REFERENCE_ACCURACY="$(docker exec "${CONTAINER}" python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["accuracy"])' \
    "${BUILD_WORK_DIR}/onnx_accuracy/summary.json")"
REFERENCE_SOURCE="本次 ONNX 浮点全量实测,使用与板端相同的评测协议"
test -s "${MODEL_OUTPUT_DIR}/model_fp32_opset13.onnx"
test -s "${MODEL_OUTPUT_DIR}/model_fp32_pure_npu.onnx"

echo "[编译主机 2] 在编译主机交叉编译板端 C++ 测试程序."
readonly OPENCV_SOURCE="${TOOLCHAIN_ROOT}/opencv-4.9.0/opencv-4.9.0"
readonly OPENCV_BUILD="${TOOLCHAIN_ROOT}/opencv-4.9.0/build-aarch64-headers"
readonly TARGET_LIBS="${TOOLCHAIN_ROOT}/genio720-libs"
command -v "${CXX}" >/dev/null 2>&1
mkdir -p "$(dirname "${BINARY_OUTPUT}")"
"${CXX}" -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic \
    -Wno-deprecated-enum-enum-conversion \
    -I"$(dirname "${ORT_HEADER}")" -I"${OPENCV_BUILD}" \
    -I"${OPENCV_SOURCE}/modules/core/include" \
    -I"${OPENCV_SOURCE}/modules/imgproc/include" \
    -I"${OPENCV_SOURCE}/modules/imgcodecs/include" \
    "${SCRIPT_DIR}/board/yoloworld_board_eval.cpp" \
    "${TARGET_LIBS}/libopencv_imgcodecs.so.409" \
    "${TARGET_LIBS}/libopencv_imgproc.so.409" \
    "${TARGET_LIBS}/libopencv_core.so.409" \
    -Wl,--allow-shlib-undefined -pthread -ldl -o "${BINARY_OUTPUT}"
file "${BINARY_OUTPUT}"

echo "[编译主机 上传] 上传模型、程序、评测代码和路径配置到板端."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "mkdir -p '${BOARD_DEPLOY_DIR}/models' '${BOARD_DEPLOY_DIR}/board'"
scp "${SSH_OPTIONS[@]}" \
    "${MODEL_OUTPUT_DIR}/model_fp32_pure_npu.onnx" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/models/"
scp "${SSH_OPTIONS[@]}" \
    "${SCRIPT_DIR}/board/evaluate_full_coco.py" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/board/"
scp "${SSH_OPTIONS[@]}" "${MODEL_ROOT}/../../../../tools/summarize_board_result.py" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/board/"
scp "${SSH_OPTIONS[@]}" "${BINARY_OUTPUT}" "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/board/yoloworld_board_eval"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" "chmod 755 '${BOARD_DEPLOY_DIR}/board/yoloworld_board_eval'"
scp "${SSH_OPTIONS[@]}" "${SCRIPT_DIR}/run.sh" "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/"
printf 'BOARD_DATASET_DIR=%q\nBOARD_RESULTS_DIR=%q\nREFERENCE_ACCURACY=%q\nREFERENCE_SOURCE=%q\nORT_RUNTIME_LIB=%q\n' \
    "${BOARD_DATASET_DIR}" "${BOARD_RESULTS_DIR}" "${REFERENCE_ACCURACY}" "${REFERENCE_SOURCE}" "${ORT_RUNTIME_LIB}" |
    ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
        "cat > '${BOARD_DEPLOY_DIR}/board_paths.conf'"
echo "[NEXT] 在板端运行: bash '${BOARD_DEPLOY_DIR}/run.sh'"
