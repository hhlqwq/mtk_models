#!/usr/bin/env bash
# 原生部署包的通路检查只保存少量文本示例,不生成评价报告.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 0 )); then
    echo "[错误] 用法: bash compile.sh,不接受参数." >&2
    exit 2
fi
cd "${ROOT}"
exec 9>"${ROOT}/.run.lock"
flock -n 9 || { echo "[错误] 本模型已有运行任务." >&2; exit 2; }
WORK="/tmp/hailongcodex/$(date +%F)/native_compile_$(date +%H%M%S)_$$"
mkdir -p "${WORK}" "${ROOT}/examples/output"
MAX_TOKENS="${MAX_NEW_TOKENS:-32}"
[[ "${MAX_TOKENS}" =~ ^[0-9]+$ ]] && (( MAX_TOKENS >= 2 )) || exit 2
for language in zh en; do
    echo "[示例] ${language} 原生推理."
    llm_cmdline_tool scripts/config-yocto.yaml --read-tokens \
        -i "scripts/prompts/${language}_demo.tokens.txt" -m "$((MAX_TOKENS - 1))" \
        >"${WORK}/${language}.log" 2>&1
    python3 - "${WORK}/${language}.log" "${ROOT}/examples/output/${language}.txt" <<'PY'
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text(errors="replace")
answer = re.search(r"\[Full\s*Response\]\s*\n(.*?)\n(?:Generated Tokens:|\[Latency\])", text, re.S)
if answer is None:
    raise ValueError("原生输出没有完整回答.")
Path(sys.argv[2]).write_text(answer.group(1).strip() + "\n", encoding="utf-8")
PY
    echo "[示例完成] ${language}."
done
echo "[通路完成] 示例已保存,正式精度、性能与 RSS 请执行 evaluate.sh."
