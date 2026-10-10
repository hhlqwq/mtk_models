#!/usr/bin/env bash

# 原生 GAI 流程: 在 Ubuntu89 Docker 运行,保留各阶段日志和失败状态.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
TASK_ROOT="${TASK_ROOT:-/tmp/hailongcodex/2026-10-10}"
WORK="${NATIVE_WORK:-${TASK_ROOT}/minicpm5_native}"
SOURCE="${MODEL_SOURCE:-${TASK_ROOT}/minicpm5_source}"
PYTHON="${GAI_PYTHON:-${TASK_ROOT}/deepseek_gai_env/bin/python}"
NCC_ROOT="${NCC_ROOT:-/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host}"
CONTEXT="${CONTEXT_SIZE:-1024}"
PREFILL="${PREFILL_TOKENS:-128}"
CALIBRATION_BATCHES="${CALIBRATION_BATCHES:-32}"
CALIBRATION_TRAIN="${CALIBRATION_TRAIN:-}"
WEIGHT_OPT_CONFIG="${WEIGHT_OPT_CONFIG:-}"
MODEL="${WORK}/MiniCPM5-2B"
QUANTIZED="${WORK}/tflite/MiniCPM5-2B_asym4W_sym16A_Overall_hessian"
if [[ -n "${WEIGHT_OPT_CONFIG}" ]]; then
    opt_name="${WEIGHT_OPT_CONFIG##*/}"
    QUANTIZED="${QUANTIZED}_${opt_name%.json}"
fi
STAGE="${1:-all}"

if (( $# > 1 )) || [[ ! "${STAGE}" =~ ^(all|prepare|calibrate|quantize|shape|compile)$ ]]; then
    echo '[错误] 用法: bash run_native.sh [prepare|calibrate|quantize|shape|compile|all].' >&2
    exit 2
fi
if [[ ! "${CONTEXT}" =~ ^[0-9]+$ || ! "${PREFILL}" =~ ^[0-9]+$ ]] ||
    (( PREFILL < 1 || CONTEXT < PREFILL )); then
    echo '[错误] Prefill 和上下文必须为正整数且 Prefill 不超过上下文.' >&2
    exit 2
fi
for path in "${WORK}" "${SOURCE}" "${PYTHON}" "${NCC_ROOT}"; do
    if [[ "${path}" != /* || "${path}" == *$'\n'* ]]; then
        echo '[错误] 环境路径必须为无换行的绝对路径.' >&2
        exit 2
    fi
done
mkdir -p "${WORK}/logs"
docker exec "${CONTAINER}" mkdir -p "${WORK}"

run_python() {
    # 在隔离 Python 中执行已准备的命令,离线读取官方模型.
    docker exec -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        -w "${WORK}" "${CONTAINER}" "${PYTHON}" "$@"
}

if [[ "${STAGE}" == all || "${STAGE}" == prepare ]]; then
    echo '[1/5] 校验官方资源并生成 MiniCPM5 双语校准 Token.'
    prepare_options=()
    if [[ -n "${CALIBRATION_TRAIN}" ]]; then
        prepare_options=(--calibration-train "${CALIBRATION_TRAIN}")
    fi
    run_python "${SCRIPT_DIR}/host/prepare_native.py" --source "${SOURCE}" \
        --output "${WORK}" --context "${CONTEXT}" "${prepare_options[@]}" \
        2>&1 | tee "${WORK}/logs/prepare.log"
fi
if [[ "${STAGE}" == all || "${STAGE}" == calibrate ]]; then
    echo '[2/5] 生成真实权重校准数据,包含 Prompt 和生成阶段.'
    run_python -c 'from pathlib import Path; Path("calibration_complete.txt").unlink(missing_ok=True)'
    run_python "${PYTHON%/python}/mtk_make_llm_ptq_calib_dataset" converter \
        "${MODEL}/config.json" "${WORK}/calibration.jsonl" \
        -b "${CALIBRATION_BATCHES}" -m 128 2>&1 | tee "${WORK}/logs/calibration.log"
    run_python -c 'from pathlib import Path; import sys; Path("calibration_complete.txt").write_text(sys.argv[1]+"\n")' "${CALIBRATION_BATCHES}"
fi
if [[ "${STAGE}" == all || "${STAGE}" == quantize ]]; then
    run_python -c 'from pathlib import Path; import sys; assert Path("calibration_complete.txt").read_text().strip()==sys.argv[1], "校准完成标记与批次数不符"' "${CALIBRATION_BATCHES}"
    echo '[3/5] 原生 W4A16 量化与 Hessian 权重优化,逐层记录进度.'
    quantize_options=()
    if [[ -n "${WEIGHT_OPT_CONFIG}" ]]; then
        quantize_options=(--extra_converter_options "${WEIGHT_OPT_CONFIG}")
    fi
    run_python "${PYTHON%/python}/mtk_ptq_llm" converter "${MODEL}/config.json" \
        -p asym4W_sym16A -d "${WORK}/calibration_datasets/MiniCPM5-2B" \
        -m Overall -w hessian "${quantize_options[@]}" \
        2>&1 | tee "${WORK}/logs/quantization.log"
fi
if [[ "${STAGE}" == all || "${STAGE}" == shape ]]; then
    echo "[4/5] 导出 ${PREFILL}t${CONTEXT}c 和 1t${CONTEXT}c,各一个分片."
    run_python "${PYTHON%/python}/mtk_fix_llm_shape" "${QUANTIZED}" \
        "${PREFILL}t${CONTEXT}c" "1t${CONTEXT}c" -n 1 \
        2>&1 | tee "${WORK}/logs/shape.log"
fi
if [[ "${STAGE}" == all || "${STAGE}" == compile ]]; then
    echo '[5/5] 编译 Genio720 MDLA5.3 单核模型,禁止 CPU 桥接.'
    docker exec -i -e LD_LIBRARY_PATH="${NCC_ROOT}/lib" \
        "${CONTAINER}" bash -s -- "${QUANTIZED}" "${NCC_ROOT}" "${PREFILL}" "${CONTEXT}" \
        2>&1 <<'COMPILE' | tee "${WORK}/logs/compile.log"
set -euo pipefail
for shape in "${3}t${4}c" "1t${4}c"; do
    folder="${1}_${shape}"
    mapfile -t files < <(find "${folder}" -maxdepth 1 -name '*.tflite' -type f)
    if (( ${#files[@]} != 1 )); then
        echo "[错误] ${folder} 必须只包含一个静态模型." >&2
        exit 1
    fi
    model="${files[0]}"
    extra=()
    if [[ "${shape}" == 1t* ]]; then
        extra=(--split-16a4w-conv-oc)
    else
        extra=(--broadcast-act-wgt --broadcast-flow-distance=63 --split-large-conv-ic=1536)
    fi
    "${2}/bin/ncc-tflite" --arch=mdla5.3 --l1-size-kb=256 --num-mdla=1 \
        --opt=3 --opt-footprint --opt-aggressive --stable-linearize \
        --gno=LTS,Inception --gno-exp --gno-non-4d-tiling --mlo \
        --disable-apusys --fc-to-conv --mdla-int16-lut \
        --suppress-input --suppress-output --show-memory-summary \
        --mdla-conv-exp --intval-color-legacy --disallow-bridge "${extra[@]}" \
        "${model}" --dla-file "${model%.tflite}.dla"
    test -s "${model%.tflite}.dla"
done
COMPILE
fi
echo "[阶段完成] ${STAGE}; 编译成功后仍需板端验证,不能标记为交付完成."
