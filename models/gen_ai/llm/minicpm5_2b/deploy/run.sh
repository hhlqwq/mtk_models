#!/usr/bin/env bash
# 容器内一键转换并上传; 板端运行同一入口,默认全量评测.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 打印可复制的实际命令,外部编译器不再经过 Python 转发.
run_command() {
    printf '[执行] '
    printf '%q ' "$@"
    printf '\n'
    "$@"
}

# 编译一张静态图,公共参数与 Prefill/Decode 差异在此完整展示.
compile_dla() {
    local source="$1"
    local phase="$2"
    local target="${source%.tflite}.dla"
    local extra=()
    if [[ "${phase}" == prefill ]]; then
        extra=(--broadcast-act-wgt --broadcast-flow-distance=63 --split-large-conv-ic=1536)
    else
        extra=(--split-16a4w-conv-oc)
    fi
    [[ -s "${source}" ]] || { echo "[错误] 缺少静态图: ${source}" >&2; return 2; }
    run_command env LD_LIBRARY_PATH="${NCC_ROOT}/lib" \
        "${NCC_ROOT}/bin/ncc-tflite" \
        --arch=mdla5.3 --l1-size-kb=256 --num-mdla=1 \
        --opt=3 --opt-footprint --opt-accuracy --stable-linearize \
        --gno=LTS,Inception --gno-exp --gno-non-4d-tiling --mlo \
        --disable-apusys --fc-to-conv --mdla-int16-lut \
        --suppress-input --suppress-output --show-memory-summary \
        --mdla-conv-exp --intval-color-legacy --disallow-bridge \
        "${extra[@]}" "${source}" --dla-file "${target}"
    [[ -s "${target}" ]] || { echo "[错误] DLA 未生成: ${target}" >&2; return 2; }
    echo "[DLA] ${target}"
}

export MTK_EVAL_TEST=0
if (( $# == 1 )) && [[ "$1" == "-test" ]]; then
    export MTK_EVAL_TEST=1
elif (( $# != 0 )); then
    echo '[错误] 用法: bash run.sh [-test]' >&2
    exit 2
fi
export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
if [[ "${MTK_EVAL_TEST}" == 1 && "${EVAL_RUN_ID}" != *_test ]]; then
    export EVAL_RUN_ID="${EVAL_RUN_ID}_test"
fi
if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    exec 9>"${SCRIPT_DIR}/.run.lock"
    flock -n 9 || { echo '[错误] 本模型已有运行任务.' >&2; exit 2; }
    export PYTHONPATH="${SCRIPT_DIR}/board${PYTHONPATH:+:${PYTHONPATH}}"
    echo "[板端 1/3] 生成、输入一致性及性能检查: ${EVAL_RUN_ID}."
    run_command python3 "${SCRIPT_DIR}/board/infer.py"
    echo '[板端 2/3] MMLU Accuracy 和 WikiText2 PPL.'
    run_command python3 "${SCRIPT_DIR}/board/evaluate.py"
    echo '[板端 3/3] 汇总同协议指标及部署文件大小.'
    run_command python3 "${SCRIPT_DIR}/board/summarize.py"
else
    : "${MTK_GAI_PYTHON:?请先通过 docker/enter_container.sh 进入容器}"
    : "${MTK_MODELS_ROOT:?缺少公共项目环境}"
    : "${NCC_ROOT:?缺少 NCC 编译器路径,请检查本地平台配置}"
    export PYTHONPATH="${MTK_MODELS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    echo '[资源预检] 检查 NAS 数据集和模型路径,不读取模型权重.'
    run_command "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/evaluate.py" --check-data
    MODEL_BUILD_DIR="$(python3 "${MTK_MODELS_ROOT}/tools/storage.py" --print-build-dir "${SCRIPT_DIR}/../config.yaml")"
    mkdir -p "${MODEL_BUILD_DIR}"
    exec 9>"${MODEL_BUILD_DIR}/.run.lock"
    flock -n 9 || { echo '[错误] 本模型已有转换任务.' >&2; exit 2; }
    echo '[容器 1/8] 校验官方来源并准备转换副本.'
    run_command "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/prepare.py"
    echo '[容器 2/8] 检查评测资源与完整题集上下文需求.'
    run_command "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/evaluate.py" --prepare-only
    echo '[容器 3/8] 校准、量化并导出双静态图.'
    run_command "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/convert.py"
    echo '[容器 4/8] 直接调用 ncc-tflite 编译双 DLA.'
    source "${MODEL_BUILD_DIR}/native/compile.env"
    compile_dla "${PREFILL_TFLITE}" prefill
    compile_dla "${DECODE_TFLITE}" decode
    echo '[容器 5/8] 同协议 FP32 参考评价及板端输入准备.'
    run_command "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/evaluate.py"
    echo '[容器 6/8] 交叉编译公共 Neuron 桥接库.'
    run_command make -C "${SCRIPT_DIR}/board" BUILD_DIR="${MODEL_BUILD_DIR}"
    echo '[容器 7/8] 打包模型、评价输入及板端入口.'
    run_command "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/package.py"
    echo '[容器 8/8] 上传开发板.'
    run_command "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/upload.py"
fi
