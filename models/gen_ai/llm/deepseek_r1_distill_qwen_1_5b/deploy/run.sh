#!/usr/bin/env bash

# 两步流程: Ubuntu89 Docker 导出编译并上传,92 开发板单独运行测试.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    # 1. 官方资源与评测语料.资源必须离线准备,脚本不下载文件.
    MODEL_SOURCE="${MODEL_SOURCE:-/tmp/hailongcodex/2026-10-10/deepseek_source}"
    # 可选 JSONL: 每行包含 id 和 text; 不填写时仅运行自编双语样例.
    EVAL_CORPUS="${EVAL_CORPUS:-}"
    # 2. 模型产物与临时目录.
    MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-${MODEL_ROOT}/models/generated}"
    BUILD_WORK_DIR="${BUILD_WORK_DIR:-/tmp/hailongcodex/$(date +%F)/deepseek_r1_distill_qwen_1_5b}"
    # 3. 板端地址与结果目录.
    BOARD_HOST="${BOARD_HOST:-root@192.168.0.92}"
    BOARD_DEPLOY_DIR="${BOARD_DEPLOY_DIR:-/root/hailong.he/open_models/deepseek_r1_distill_qwen_1_5b}"
    BOARD_RESULTS_DIR="${BOARD_RESULTS_DIR:-${BOARD_DEPLOY_DIR}/results}"
    # 4. 固定输入范围与解码上限.当前仅实现单 Token Prefill/Decode.
    CONTEXT_SIZE="${CONTEXT_SIZE:-1024}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
    # 5. 既有编译环境,依赖在隔离环境手动准备.
    MTK_G720_CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
    LLM_PYTHON="${LLM_PYTHON:-/tmp/hailongcodex/2026-10-10/deepseek_env/bin/python}"
    NCC_ROOT="${NCC_ROOT:-/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host}"
    SDK_HOST="${SDK_HOST:-/data/users/hailong.he/data/MTKG720/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host}"
    TARGET_LIBS="${TARGET_LIBS:-/data/users/hailong.he/data/MTKG720/cpp_toolchain/genio720-libs}"
    CROSS_CXX="${CROSS_CXX:-aarch64-linux-gnu-g++}"
    SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=yes)
fi

if (( $# != 0 )); then
    echo '[错误] 直接运行 bash run.sh,参数在配置区或环境变量中设置.' >&2
    exit 2
fi

if [[ -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    source "${SCRIPT_DIR}/board_paths.conf"
    RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
        echo '[错误] EVAL_RUN_ID 只能包含字母、数字、下划线和连字符.' >&2
        exit 2
    fi
    echo "[板端] 运行编号 ${RUN_ID},只使用 Neuron 硬件 Runtime."
    python3 "${SCRIPT_DIR}/board/evaluate.py" \
        --models "${SCRIPT_DIR}/models" \
        --library "${SCRIPT_DIR}/board/libneuron_bridge.so" \
        --samples "${SCRIPT_DIR}/models/samples.json" \
        --output "${BOARD_RESULTS_DIR}/${RUN_ID}" --run-id "${RUN_ID}"
    echo "[回收] ${BOARD_RESULTS_DIR}/${RUN_ID}; 使用编译主机 summarize.py 汇总."
    exit 0
fi

for path in "${MODEL_SOURCE}" "${MODEL_OUTPUT_DIR}" "${BUILD_WORK_DIR}" "${BOARD_DEPLOY_DIR}"; do
    if [[ "${path}" != /* || "${path}" == *"'"* || "${path}" == *$'\n'* ]]; then
        echo '[错误] 目录必须是无单引号或换行的绝对路径.' >&2
        exit 2
    fi
done
test -s "${MODEL_SOURCE}/source_manifest.json"
command -v "${CROSS_CXX}" >/dev/null
docker exec "${MTK_G720_CONTAINER}" test -x "${LLM_PYTHON}"
mkdir -p "${BUILD_WORK_DIR}/logs"

echo '[编译主机 1/5] 在 Docker 校验官方资源并导出静态 ONNX 分片.'
docker exec -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
    "${MTK_G720_CONTAINER}" "${LLM_PYTHON}" "${SCRIPT_DIR}/host/export_model.py" \
    --source "${MODEL_SOURCE}" --output "${MODEL_OUTPUT_DIR}" --context "${CONTEXT_SIZE}" \
    2>&1 | tee "${BUILD_WORK_DIR}/logs/export.log"

echo '[编译主机 2/5] 计算官方 PyTorch 与 ONNX 同协议参考.'
CORPUS_ARGUMENTS=()
if [[ -n "${EVAL_CORPUS}" ]]; then CORPUS_ARGUMENTS=(--corpus "${EVAL_CORPUS}"); fi
docker exec -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
    "${MTK_G720_CONTAINER}" "${LLM_PYTHON}" "${SCRIPT_DIR}/host/evaluate_reference.py" \
    --source "${MODEL_SOURCE}" --models "${MODEL_OUTPUT_DIR}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" "${CORPUS_ARGUMENTS[@]}" \
    2>&1 | tee "${BUILD_WORK_DIR}/logs/reference.log"

echo '[编译主机 3/5] 转换 TFLite 并编译 MDLA 5.3 DLA,禁止桥接.'
docker exec -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
    "${MTK_G720_CONTAINER}" "${LLM_PYTHON}" "${SCRIPT_DIR}/host/convert_model.py" \
    --models "${MODEL_OUTPUT_DIR}" --ncc-root "${NCC_ROOT}" \
    2>&1 | tee "${BUILD_WORK_DIR}/logs/compile.log"

echo '[编译主机 4/5] 交叉编译 Neuron Runtime 桥接库.'
"${CROSS_CXX}" -std=c++17 -O2 -shared -fPIC -Wall -Wextra -Wpedantic \
    -I"${SDK_HOST}/include" "${SCRIPT_DIR}/board/neuron_bridge.cpp" \
    "${TARGET_LIBS}/libneuronusdk_runtime.mtk.so.8" -Wl,--allow-shlib-undefined \
    -o "${BUILD_WORK_DIR}/libneuron_bridge.so"

echo '[编译主机 5/5] 取回 Docker 产物并上传板端.'
mkdir -p "${BUILD_WORK_DIR}/models"
for name in export_manifest.json source_manifest.json artifact_manifest.json samples.json reference.json embedding_fp16.bin; do
    docker cp "${MTK_G720_CONTAINER}:${MODEL_OUTPUT_DIR}/${name}" "${BUILD_WORK_DIR}/models/${name}"
done
docker exec "${MTK_G720_CONTAINER}" sh -c \
    "cd '${MODEL_OUTPUT_DIR}' && tar -cf - ./*.dla" > "${BUILD_WORK_DIR}/dla.tar"
tar -xf "${BUILD_WORK_DIR}/dla.tar" -C "${BUILD_WORK_DIR}/models"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "mkdir -p '${BOARD_DEPLOY_DIR}/models' '${BOARD_DEPLOY_DIR}/board'"
scp "${SSH_OPTIONS[@]}" "${BUILD_WORK_DIR}/models/"*.dla \
    "${BUILD_WORK_DIR}/models/embedding_fp16.bin" \
    "${BUILD_WORK_DIR}/models/export_manifest.json" \
    "${BUILD_WORK_DIR}/models/source_manifest.json" \
    "${BUILD_WORK_DIR}/models/artifact_manifest.json" \
    "${BUILD_WORK_DIR}/models/samples.json" "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/models/"
scp "${SSH_OPTIONS[@]}" "${SCRIPT_DIR}/board/evaluate.py" \
    "${BUILD_WORK_DIR}/libneuron_bridge.so" "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/board/"
scp "${SSH_OPTIONS[@]}" "${SCRIPT_DIR}/run.sh" "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/run.sh"
printf 'BOARD_RESULTS_DIR=%q\n' "${BOARD_RESULTS_DIR}" | \
    ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" "cat > '${BOARD_DEPLOY_DIR}/board_paths.conf'"
echo "[下一步] 在板端执行: bash '${BOARD_DEPLOY_DIR}/run.sh'"
echo "[汇总] 取回结果后在 Docker 执行: ${LLM_PYTHON} '${SCRIPT_DIR}/host/summarize.py' --source '${MODEL_SOURCE}' --models '${MODEL_OUTPUT_DIR}' --results '<已回收结果目录>'"
