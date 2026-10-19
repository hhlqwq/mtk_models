#!/usr/bin/env bash
# 运行原生双语 Demo,保留实际输出与运行日志.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
MAX_TOKENS="${MAX_NEW_TOKENS:-768}"
[[ "${MAX_TOKENS}" =~ ^[0-9]+$ ]] && (( MAX_TOKENS >= 2 )) || exit 2
# 原生 CLI 的 -m 统计首 Token 后的调用次数,总生成上限需减 1.
case "${INPUT_MODE:-tokens}" in
    tokens) TOKEN_OPTION=(--read-tokens); SUFFIX=.tokens.txt ;;
    text) TOKEN_OPTION=(); SUFFIX=.txt ;;
    *) echo "[错误] INPUT_MODE 必须为 tokens 或 text." >&2; exit 2 ;;
esac
mkdir -p "results/${RUN_ID}"
for language in zh en; do
    echo "[板端] ${language} Demo"
    llm_cmdline_tool scripts/config-yocto.yaml "${TOKEN_OPTION[@]}" -i "scripts/prompts/${language}_demo${SUFFIX}" -m "$((MAX_TOKENS - 1))" 2>&1 | tee "results/${RUN_ID}/${language}_demo.log"
done
