#!/usr/bin/env bash
# Whisper-Tiny 单脚本流程: 89 编译双 DLA 和 C++,开发板测 LibriSpeech.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 用户配置: 留空的可选项使用仓库内默认路径.
if [[ ! -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    # Encoder ONNX: 留空则使用 models/encoder_fp32.onnx.
    ENCODER_ONNX=""
    # Decoder ONNX: 留空则使用 models/decoder_step_fp32.onnx.
    DECODER_ONNX=""
    # 临时构建目录: 辅助输入、缓存和程序放在仓库外.
    BUILD_WORK_DIR="/tmp/hailongcodex/$(date +%F)/whisper_tiny"
    # 模型输出目录: 留空则使用 models/.
    MODEL_OUTPUT_DIR=""
    # 板端 LibriSpeech test-clean 数据集目录.
    LIBRISPEECH_ROOT=""
    # 板端部署目录.
    BOARD_DEPLOY_DIR=""
    # 板端结果目录: 留空则位于部署目录下.
    BOARD_RESULTS_DIR=""
    # ONNX 精度数据: 编译主机与 Docker 都可访问的全量数据集根目录,必须填写.
    ONNX_DATASET_DIR=""
    # 部署精度方案: 按实际编译信息填写,例如 w8a8、w8a16、fp16、fp32、mixed; 未确认用 unknown.
    DEPLOYMENT_PRECISION="unknown"
    # 实际权重类型: 例如 int8、fp16、mixed; 不根据输入输出或文件名推断.
    WEIGHT_DTYPE="unknown"
    # 实际激活类型: 例如 int16、fp16、mixed; 未确认用 unknown.
    ACTIVATION_DTYPE="unknown"
    # 精度依据: 编译配置/报告或混合层说明; 填写精度时同步填写,这些配置不改变编译策略.
    PRECISION_SOURCE=""
    # 板端 SSH 用户和地址.
    BOARD_HOST="root@192.168.0.92"
fi

# 板端阶段: C++ 准备音频和推理,Python 仅计算 WER.
if [[ -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    source "${SCRIPT_DIR}/board_paths.conf"
    readonly RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    readonly RESULT_DIR="${BOARD_RESULTS_DIR}/${RUN_ID}"
    readonly RUN_DIR="${RESULT_DIR}/work"
    readonly CACHE_DIR="${RUN_DIR}/cache/librispeech"
    readonly TOOL_DIR="${SCRIPT_DIR}/tools"
    readonly REPORT_DIR="${RUN_DIR}/report"
    export PYTHONPATH="${TOOL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    test -d "${LIBRISPEECH_ROOT}"
    test -s "${SCRIPT_DIR}/encoder_fp32.dla"
    test -s "${SCRIPT_DIR}/decoder_step_fp32.dla"
    test -s "${SCRIPT_DIR}/mel_filters_f32.bin"
    test -s "${SCRIPT_DIR}/decode_config.txt"
    test ! -e "${RESULT_DIR}"
    command -v ffmpeg
    python3 -c 'from whisper.normalizers import EnglishTextNormalizer; from whisper.tokenizer import get_tokenizer'

    echo "[1/4] 在板端整理 2620 条音频并生成 Mel 输入."
    "${SCRIPT_DIR}/prepare_board_audio" \
        --dataset-root "${LIBRISPEECH_ROOT}" \
        --filters "${SCRIPT_DIR}/mel_filters_f32.bin" \
        --output-dir "${CACHE_DIR}"
    test "$(wc -l < "${CACHE_DIR}/source_manifest.jsonl")" -eq 2620
    mkdir -p "${RUN_DIR}/logs" "${REPORT_DIR}"

    echo "[2/4] 在板端运行双 DLA 常驻 C++ 推理."
    "${SCRIPT_DIR}/whisper_board_eval" \
        "${SCRIPT_DIR}/encoder_fp32.dla" \
        "${SCRIPT_DIR}/decoder_step_fp32.dla" \
        "${CACHE_DIR}/board_manifest.tsv" \
        "${SCRIPT_DIR}/decode_config.txt" \
        "${RUN_DIR}/board_predictions.jsonl" \
        2>&1 | tee "${RUN_DIR}/logs/board.log"

    echo "[3/4] 在板端计算 LibriSpeech WER."
    python3 "${TOOL_DIR}/evaluate_accuracy.py" \
        --dataset librispeech \
        --source-manifest "${CACHE_DIR}/source_manifest.jsonl" \
        --board-predictions "${RUN_DIR}/board_predictions.jsonl" \
        --preprocess-metrics "${CACHE_DIR}/preprocess_metrics.jsonl" \
        --output-dir "${REPORT_DIR}"
    echo "[4/4] 汇总核心指标并清理临时结果."
    # 所有原始数据仅在本次 work 下生成; 汇总成功后由工具清理.
    python3 "${SCRIPT_DIR}/summarize_board_result.py" \
        --model "whisper_tiny" --work-dir "${RUN_DIR}" \
        --output "${RESULT_DIR}/summary.json" --run-id "${RUN_ID}" \
        --reference "${REFERENCE_ACCURACY:-}" \
        --reference-source "${REFERENCE_SOURCE:-用户提供的同协议参考基准}" \
        --precision "${DEPLOYMENT_PRECISION:-unknown}" \
        --weight-dtype "${WEIGHT_DTYPE:-unknown}" \
        --activation-dtype "${ACTIVATION_DTYPE:-unknown}" \
        --precision-source "${PRECISION_SOURCE:-}"
    exit 0
fi

if (( $# != 0 )); then
    echo "[ERROR] 在编译主机直接运行 bash deploy/run.sh,无需参数." >&2
    exit 2
fi

# 路径配置: 双 ONNX、产物、LibriSpeech、工具链和板端位置均可覆盖.
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${ONNX_DATASET_DIR:?请在脚本顶部指定编译主机的全量精度数据集}"
test -d "${ONNX_DATASET_DIR}"
if [[ "${ONNX_DATASET_DIR}" != /* ]]; then
    echo "[ERROR] ONNX_DATASET_DIR 必须是编译主机与 Docker 共用的绝对路径." >&2
    exit 2
fi
readonly ENCODER_ONNX="${ENCODER_ONNX:-${MODEL_ROOT}/models/encoder_fp32.onnx}"
readonly DECODER_ONNX="${DECODER_ONNX:-${MODEL_ROOT}/models/decoder_step_fp32.onnx}"
readonly MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-${MODEL_ROOT}/models}"
readonly LIBRISPEECH_ROOT="${LIBRISPEECH_ROOT:?请指定板端 LibriSpeech test-clean 目录}"
readonly BOARD_DEPLOY_DIR="${BOARD_DEPLOY_DIR:?请指定 BOARD_DEPLOY_DIR}"
readonly BOARD_RESULTS_DIR="${BOARD_RESULTS_DIR:-${BOARD_DEPLOY_DIR}/results}"
readonly BOARD_HOST="${BOARD_HOST:-root@192.168.0.92}"
readonly CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
readonly MTK_SETUP_SCRIPT="${MTK_SETUP_SCRIPT:-/opt/mtk-build/setup_container.sh}"
readonly NCC_ROOT="${NCC_ROOT:-/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host}"
readonly NCC_MODE="${NCC_MODE:-check}"
readonly TOOLCHAIN_ROOT="${MTK_G720_CPP_TOOLCHAIN_ROOT:-/data/users/hailong.he/data/MTKG720/cpp_toolchain}"
readonly NEURON_INCLUDE="${MTK_NEURON_INCLUDE:-/data/users/hailong.he/data/MTKG720/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/include}"
readonly CXX="${CROSS_CXX:-aarch64-linux-gnu-g++}"
readonly SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

mkdir -p "${BUILD_WORK_DIR}/tmp"
export TMPDIR="${BUILD_WORK_DIR}/tmp"

test -s "${ENCODER_ONNX}"
test -s "${DECODER_ONNX}"
if [[ ! "${BOARD_DEPLOY_DIR}" =~ ^/[A-Za-z0-9_./-]+$ ]]; then
    echo "[ERROR] BOARD_DEPLOY_DIR 必须是无空格的板端绝对路径." >&2
    exit 2
fi
if [[ "${NCC_MODE}" != "check" && "${NCC_MODE}" != "strict" ]]; then
    echo "[ERROR] NCC_MODE 只能为 check 或 strict." >&2
    exit 2
fi

echo "[1/4] 在 Docker 中转换并编译双 DLA."
docker exec -i -e BUILD_WORK_DIR="${BUILD_WORK_DIR}" -e PYTHONDONTWRITEBYTECODE=1 -e MODEL_ROOT="${MODEL_ROOT}" \
    -e ENCODER_ONNX="${ENCODER_ONNX}" -e DECODER_ONNX="${DECODER_ONNX}" \
    -e MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR}" \
    -e MTK_SETUP_SCRIPT="${MTK_SETUP_SCRIPT}" \
    -e NCC_ROOT="${NCC_ROOT}" -e NCC_MODE="${NCC_MODE}" \
    "${CONTAINER}" bash -s <<'DOCKER_BUILD'
set -euo pipefail
mkdir -p "${MODEL_OUTPUT_DIR}" "${BUILD_WORK_DIR}/tmp" "${BUILD_WORK_DIR}/cache"
export TMPDIR="${BUILD_WORK_DIR}/tmp"
export XDG_CACHE_HOME="${BUILD_WORK_DIR}/cache"
export TORCH_HOME="${BUILD_WORK_DIR}/cache/torch"
cd "${BUILD_WORK_DIR}"
bash "${MTK_SETUP_SCRIPT}"
python "${MODEL_ROOT}/deploy/convert_fp32.py" \
    --onnx "${ENCODER_ONNX}" \
    --output "${MODEL_OUTPUT_DIR}/encoder_fp32.tflite"
python "${MODEL_ROOT}/deploy/convert_fp32.py" \
    --onnx "${DECODER_ONNX}" \
    --output "${MODEL_OUTPUT_DIR}/decoder_step_fp32.tflite"
export LD_LIBRARY_PATH="${NCC_ROOT}/lib:${LD_LIBRARY_PATH:-}"
for model_name in encoder decoder_step; do
    input="${MODEL_OUTPUT_DIR}/${model_name}_fp32.tflite"
    output="${MODEL_OUTPUT_DIR}/${model_name}_fp32.dla"
    if [[ "${NCC_MODE}" == "strict" ]]; then
        "${NCC_ROOT}/bin/ncc-tflite" --arch=mdla5.3 \
            --suppress-input --suppress-output --disallow-bridge \
            "${input}" -o "${output}"
    else
        "${NCC_ROOT}/bin/ncc-tflite" --arch=mdla5.3 \
            --show-exec-plan "${input}" -o "${output}"
    fi
done
DOCKER_BUILD

echo "[ONNX] 在编译主机 Docker 中评测全量浮点精度."
docker exec -i -e PYTHONDONTWRITEBYTECODE=1 \
    -e MODEL_ROOT="${MODEL_ROOT}" -e ONNX_DATASET_DIR="${ONNX_DATASET_DIR}" \
    -e ONNX_REFERENCE="${ENCODER_ONNX}" -e BUILD_WORK_DIR="${BUILD_WORK_DIR}" \
    -e DECODER_ONNX="${DECODER_ONNX}" \
    "${CONTAINER}" bash -s <<'ONNX_ACCURACY'
set -euo pipefail
test -d "${ONNX_DATASET_DIR}"
export TMPDIR="${BUILD_WORK_DIR}/tmp"
export XDG_CACHE_HOME="${BUILD_WORK_DIR}/cache"
export TORCH_HOME="${BUILD_WORK_DIR}/cache/torch"
REPO_ROOT="$(cd "${MODEL_ROOT}/../../../.." && pwd)"
python "${REPO_ROOT}/tools/accuracy/evaluate_onnx.py" \
    --model "whisper_tiny" --onnx "${ONNX_REFERENCE}" \
    --decoder-onnx "${DECODER_ONNX}" \
    --dataset-root "${ONNX_DATASET_DIR}" \
    --output "${BUILD_WORK_DIR}/onnx_accuracy/summary.json"
ONNX_ACCURACY
# 将本次实测精度写入板端配置,不传递主机预测、耗时或内存.
REFERENCE_ACCURACY="$(docker exec "${CONTAINER}" python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["accuracy"])' \
    "${BUILD_WORK_DIR}/onnx_accuracy/summary.json")"
REFERENCE_SOURCE="本次 ONNX 浮点全量实测,使用与板端相同的评测协议"
test -s "${MODEL_OUTPUT_DIR}/encoder_fp32.dla"
test -s "${MODEL_OUTPUT_DIR}/decoder_step_fp32.dla"

echo "[2/4] 在编译主机交叉编译板端音频准备和推理程序."
readonly OPENCV_SOURCE="${TOOLCHAIN_ROOT}/opencv-4.9.0/opencv-4.9.0"
readonly OPENCV_BUILD="${TOOLCHAIN_ROOT}/opencv-4.9.0/build-aarch64-headers"
readonly TARGET_LIBS="${TOOLCHAIN_ROOT}/genio720-libs"
command -v "${CXX}" >/dev/null 2>&1
"${CXX}" -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic \
    -I"${OPENCV_BUILD}" -I"${OPENCV_SOURCE}/modules/core/include" \
    "${SCRIPT_DIR}/prepare_board_audio.cpp" \
    "${TARGET_LIBS}/libopencv_core.so.409" \
    -Wl,--allow-shlib-undefined -pthread -ldl \
    -o "${BUILD_WORK_DIR}/prepare_board_audio"
"${CXX}" -std=c++20 -O2 -DNDEBUG -Wall -Wextra -Wpedantic \
    -I"${NEURON_INCLUDE}" \
    "${SCRIPT_DIR}/inference_demo/whisper_board_eval.cpp" \
    "${TARGET_LIBS}/libneuronusdk_runtime.mtk.so.8" \
    -Wl,--allow-shlib-undefined -pthread -ldl \
    -o "${BUILD_WORK_DIR}/whisper_board_eval"
file "${BUILD_WORK_DIR}/prepare_board_audio" \
    "${BUILD_WORK_DIR}/whisper_board_eval"

echo "[3/4] 在 Docker 中导出滤波器、解码规则和指标依赖."
readonly ASSETS_DIR="${BUILD_WORK_DIR}/board_assets"
docker exec -e PYTHONDONTWRITEBYTECODE=1 -e TMPDIR="${BUILD_WORK_DIR}/tmp" -e MODEL_ROOT="${MODEL_ROOT}" \
    "${CONTAINER}" python3 "${SCRIPT_DIR}/export_board_assets.py" \
    --output-dir "${ASSETS_DIR}"
docker exec -e PYTHONDONTWRITEBYTECODE=1 -e TMPDIR="${BUILD_WORK_DIR}/tmp" -e MODEL_ROOT="${MODEL_ROOT}" \
    "${CONTAINER}" python3 "${SCRIPT_DIR}/export_eval_vendor.py" \
    --output-dir "${ASSETS_DIR}/whisper_eval_vendor"
mkdir -p "${ASSETS_DIR}"
# Docker 与主机的临时目录独立,显式取回上传所需的文件.
docker cp "${CONTAINER}:${ASSETS_DIR}/." "${ASSETS_DIR}/"
test -s "${ASSETS_DIR}/mel_filters_f32.bin"
test -s "${ASSETS_DIR}/decode_config.txt"

echo "[4/4] 上传双 DLA、程序、指标代码和路径配置到板端."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "mkdir -p '${BOARD_DEPLOY_DIR}/tools'"
scp "${SSH_OPTIONS[@]}" \
    "${MODEL_OUTPUT_DIR}/encoder_fp32.dla" \
    "${MODEL_OUTPUT_DIR}/decoder_step_fp32.dla" \
    "${BUILD_WORK_DIR}/whisper_board_eval" \
    "${BUILD_WORK_DIR}/prepare_board_audio" \
    "${ASSETS_DIR}/mel_filters_f32.bin" \
    "${ASSETS_DIR}/decode_config.txt" "${SCRIPT_DIR}/run.sh" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/"
scp "${SSH_OPTIONS[@]}" \
    "${SCRIPT_DIR}/evaluate_accuracy.py" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/tools/"
scp -r "${SSH_OPTIONS[@]}" \
    "${ASSETS_DIR}/whisper_eval_vendor/whisper" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/tools/"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "chmod 755 '${BOARD_DEPLOY_DIR}/prepare_board_audio' '${BOARD_DEPLOY_DIR}/whisper_board_eval'"
scp "${SSH_OPTIONS[@]}" \
    "${MODEL_ROOT}/../../../../tools/summarize_board_result.py" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/summarize_board_result.py"
printf 'LIBRISPEECH_ROOT=%q\nBOARD_RESULTS_DIR=%q\nREFERENCE_ACCURACY=%q\nREFERENCE_SOURCE=%q\nDEPLOYMENT_PRECISION=%q\nWEIGHT_DTYPE=%q\nACTIVATION_DTYPE=%q\nPRECISION_SOURCE=%q\n' \
    "${LIBRISPEECH_ROOT}" "${BOARD_RESULTS_DIR}" "${REFERENCE_ACCURACY}" "${REFERENCE_SOURCE}" "${DEPLOYMENT_PRECISION}" "${WEIGHT_DTYPE}" "${ACTIVATION_DTYPE}" "${PRECISION_SOURCE}" |
    ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
        "cat > '${BOARD_DEPLOY_DIR}/board_paths.conf'"
echo "[OK] 在板端运行: bash '${BOARD_DEPLOY_DIR}/run.sh'"
