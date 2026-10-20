#!/usr/bin/env bash
# 运行原生双语 Demo,仅保留最新一次输出与日志.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
RUN_ID=latest_demo
exec 9>"${ROOT}/.run.lock"
flock -n 9 || { echo '[错误] 本模型已有运行任务.' >&2; exit 2; }
MAX_TOKENS="${MAX_NEW_TOKENS:-768}"
[[ "${MAX_TOKENS}" =~ ^[0-9]+$ ]] && (( MAX_TOKENS >= 2 )) || exit 2
# 原生 CLI 的 -m 统计首 Token 后的调用次数,总生成上限需减 1.
case "${INPUT_MODE:-tokens}" in
    tokens) TOKEN_OPTION=(--read-tokens); SUFFIX=.tokens.txt ;;
    text) TOKEN_OPTION=(); SUFFIX=.txt ;;
    *) echo "[错误] INPUT_MODE 必须为 tokens 或 text." >&2; exit 2 ;;
esac
[[ ! -L results && ! -L "results/${RUN_ID}" ]] || exit 2
echo '[清理] 替换上一次 Demo 输出.'
rm -rf -- "results/${RUN_ID}"
mkdir -p "results/${RUN_ID}"
for language in zh en; do
    echo "[板端] ${language} Demo"
    python3 - "results/${RUN_ID}/${language}.memory.json" \
        llm_cmdline_tool scripts/config-yocto.yaml "${TOKEN_OPTION[@]}" \
        -i "scripts/prompts/${language}_demo${SUFFIX}" -m "$((MAX_TOKENS - 1))" \
        <<'PY' 2>&1 | tee "results/${RUN_ID}/${language}_demo.log"
import json
from pathlib import Path
import resource
import subprocess
import sys

subprocess.run(sys.argv[2:], check=True)
peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
if peak <= 0:
    raise RuntimeError("没有采集到有效的推理进程峰值 RSS.")
Path(sys.argv[1]).write_text(json.dumps({"peak_process_rss_mib": peak}) + "\n")
print(f"[推理进程峰值 RSS] {peak:.2f} MiB.", flush=True)
PY
done
python3 - "results/${RUN_ID}" <<'PY'
import json
from pathlib import Path
import sys

folder = Path(sys.argv[1])
peaks = {name: json.loads((folder / f"{name}.memory.json").read_text())["peak_process_rss_mib"]
         for name in ("zh", "en")}
report = {"scope": "bilingual_generation_demo", "memory_summary": {
    "peak_process_rss_mib": max(peaks.values()), "by_language_mib": peaks,
    "scope": "native_cli_process"}}
(folder / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
for name in peaks:
    (folder / f"{name}.memory.json").unlink()
print(f"[Demo 汇总] 峰值 RSS={max(peaks.values()):.2f} MiB; {folder / 'summary.json'}.")
PY
