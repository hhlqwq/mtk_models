#!/usr/bin/env bash
# 独立全量精度: 只读取已有模型和评价资源,不量化、编译或部署.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 0 )); then
    echo "[错误] 用法: bash evaluate.sh,不接受参数." >&2
    exit 2
fi
export MTK_EVAL_SCOPE=full
export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
if [[ "${EVAL_RUN_ID}" != *_full ]]; then
    export EVAL_RUN_ID="${EVAL_RUN_ID}_full"
fi
echo "[未实现] 当前 DeepSeek 只有生成样例,没有正式全量数据集及评分协议." >&2
echo "请先实现并确认精度协议,不能把 Demo 作为全量精度结果." >&2
exit 2
