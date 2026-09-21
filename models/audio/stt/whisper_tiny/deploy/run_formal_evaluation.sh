#!/usr/bin/env bash

set -euo pipefail

readonly DATASET="${DATASET:?请设置 DATASET=librispeech 或 aishell1}"
readonly DATASET_ROOT="${DATASET_ROOT:?请设置已解压数据集根目录}"
readonly RUN_ID="${RUN_ID:?请设置唯一 RUN_ID}"
readonly ARCHIVE="${ARCHIVE:?请设置原始数据压缩包路径}"
readonly REPO_ROOT="${REPO_ROOT:-/data/users/hailong.he/github/mtk_models}"
readonly MODEL_ROOT="${REPO_ROOT}/models/audio/stt/whisper_tiny"
readonly DEPLOY_ROOT="${MODEL_ROOT}/deploy"
readonly EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/.eval/whisper_tiny/${RUN_ID}}"
readonly CONTAINER="${MTK_CONTAINER:-hhl_g720_8011}"
readonly SKIP_BOARD_BUILD="${SKIP_BOARD_BUILD:-0}"

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] 未找到 docker,请退出容器并在 Ubuntu89 宿主机运行本脚本." >&2
    exit 127
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null)" != "true" ]]; then
    echo "[ERROR] MTK 容器未运行或不存在: ${CONTAINER}." >&2
    exit 2
fi
if [[ "${REFERENCE_DEVICE:-cuda}" == "none" ]]; then
    echo "[ERROR] 一键正式评测必须生成 OpenAI 框架基线,REFERENCE_DEVICE 不能为 none." >&2
    exit 2
fi

if [[ "${SKIP_BOARD_BUILD}" == "1" ]]; then
    echo "[PIPELINE 1/4] 复用已由总入口编译的板端批量评测程序."
else
    echo "[PIPELINE 1/4] 交叉编译板端批量评测程序."
    bash "${DEPLOY_ROOT}/build_board_cpp.sh"
fi

echo "[PIPELINE 2/4] 构建数据清单、Mel 和 OpenAI 框架基线."
bash "${DEPLOY_ROOT}/prepare_accuracy.sh"

echo "[PIPELINE 3/4] 部署到 Genio 720 并执行完整板端评测."
bash "${DEPLOY_ROOT}/run_accuracy_board.sh"

echo "[PIPELINE 4/4] 汇总精度、耗时、一致性和资源指标."
bash "${DEPLOY_ROOT}/summarize_accuracy.sh"

echo "[OK] ${DATASET} 单数据集正式评测完成."
echo "[OK] 报告: ${EVAL_ROOT}/report/report.md"
echo "[OK] 汇总: ${EVAL_ROOT}/report/summary.json"
