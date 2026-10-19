#!/usr/bin/env bash
# 编译步骤在容器内执行,请先从仓库根目录运行 bash docker/enter_container.sh.
# 默认全量,-test 为快速验证.先用 docker/enter_container.sh 进入容器,完成后在开发板执行同一个脚本.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认全量,-test 只缩小评测数据,不改变校准或模型转换配置.
export MTK_EVAL_TEST=0
EVAL_PATH_SUFFIX=""
if (( $# == 1 )) && [[ "$1" == "-test" ]]; then
    export MTK_EVAL_TEST=1
    EVAL_PATH_SUFFIX="_test"
    shift
    EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    [[ "${EVAL_RUN_ID}" == test_* ]] || EVAL_RUN_ID="test_${EVAL_RUN_ID}"
    export EVAL_RUN_ID
    echo "[模式] 固定小子集流程验证,不代表全量精度."
fi
if (( $# != 0 )); then
    echo "[错误] 用法: bash run.sh [-test]" >&2
    exit 2
fi

if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    # 开发板: 四个步骤使用同一个运行编号.
    if [[ "${EVAL_RESUME:-0}" == "1" && -z "${EVAL_RUN_ID:-}" ]]; then
        echo "[错误] 续跑时请设置原来的 EVAL_RUN_ID." >&2
        exit 2
    fi
    RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    CONFIG="${SCRIPT_DIR}/config.json"
    echo "[板端 1/4] NPU 推理,运行编号: ${RUN_ID}."
    python3 "${SCRIPT_DIR}/board/infer.py" --config "${CONFIG}" --run-id "${RUN_ID}"
    echo "[板端 2/4] COCO 精度评测."
    python3 "${SCRIPT_DIR}/board/full_accuracy_board.py" --mode evaluate --config "${CONFIG}" --run-id "${RUN_ID}"
    echo "[板端 3/4] 生成效果示例."
    python3 "${SCRIPT_DIR}/board/render.py" --config "${CONFIG}" --run-id "${RUN_ID}"
    echo "[板端 4/4] 汇总指标与模型大小."
    python3 "${SCRIPT_DIR}/board/summarize.py" --config "${CONFIG}" --run-id "${RUN_ID}"
else
    # 编译容器: 各 Python 文件读取 YAML 并直接执行当前步骤.
    CONFIG="${SCRIPT_DIR}/../config.yaml"
    MODEL_BUILD_DIR="$(python3 "${MTK_MODELS_ROOT:?请先通过 docker/enter_container.sh 进入容器}/tools/storage.py" --print-build-dir "${CONFIG}")"
    echo "[容器 1/6] 导出 FP32 ONNX."
    python3 "${SCRIPT_DIR}/host/export_model.py" --config "${CONFIG}"
    echo "[容器 2/6] INT8 校准与量化."
    python3 "${SCRIPT_DIR}/host/convert_int8.py" --config "${CONFIG}"
    echo "[容器 3/6] 编译 DLA."
    python3 "${SCRIPT_DIR}/host/compile_model.py" --config "${CONFIG}"
    echo "[容器 4/6] ONNX 精度评测."
    python3 "${SCRIPT_DIR}/host/evaluate_onnx.py" --config "${CONFIG}"
    echo "[容器 5/6] 交叉编译板端 C++ 程序."
    make -C "${SCRIPT_DIR}/board" BUILD_DIR="${MODEL_BUILD_DIR}"
    echo "[容器 6/6] 上传到开发板."
    python3 "${SCRIPT_DIR}/host/upload.py" --config "${CONFIG}"
fi
