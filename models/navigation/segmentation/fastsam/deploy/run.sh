#!/usr/bin/env bash
# 无参数入口.主机完成后,登录开发板直接执行同一个脚本.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 0 )); then
    echo "[错误] run.sh 无需参数." >&2
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
    # 89 主机: 各 Python 文件自行读取 YAML,需要 SDK 的步骤自行进入 Docker.
    CONFIG="${SCRIPT_DIR}/../config.yaml"
    export FASTSAM_RUN_DATE="$(date +%F)"
    echo "[主机 1/6] 导出 FP32 ONNX."
    python3 "${SCRIPT_DIR}/host/export_model.py" --config "${CONFIG}"
    echo "[主机 2/6] INT8 校准与量化."
    python3 "${SCRIPT_DIR}/host/convert_int8.py" --config "${CONFIG}"
    echo "[主机 3/6] 编译 DLA."
    python3 "${SCRIPT_DIR}/host/compile_model.py" --config "${CONFIG}"
    echo "[主机 4/6] 全量 ONNX 精度评测."
    python3 "${SCRIPT_DIR}/host/evaluate_onnx.py" --config "${CONFIG}"
    echo "[主机 5/6] 交叉编译板端 C++ 程序."
    python3 "${SCRIPT_DIR}/host/build_cpp.py" --config "${CONFIG}"
    echo "[主机 6/6] 上传到开发板."
    python3 "${SCRIPT_DIR}/host/upload.py" --config "${CONFIG}"
fi
