#!/usr/bin/env bash
# 模型单脚本入口: 编译主机完成构建与上传,开发板完成测试.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 模型输入文件的绝对路径.
MODEL_SOURCE=""
# 校准数据目录的绝对路径.
CALIBRATION_DIR=""
# 临时目录: 缓存、辅助输入和编译程序必须放在仓库外.
BUILD_WORK_DIR="/tmp/hailongcodex/$(date +%F)/model"
# 生成产物的目录.
MODEL_OUTPUT_DIR=""
# 同协议核心参考精度: 留空时不计算损失.
REFERENCE_ACCURACY=""
# 参考基准来源: 记录后端、数据集和评测协议.
REFERENCE_SOURCE=""
# 板端测试数据目录.
BOARD_DATASET_DIR=""
# 板端部署目录.
BOARD_DEPLOY_DIR=""
# 开发板 SSH 用户和地址.
BOARD_HOST=""

if [[ -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    source "${SCRIPT_DIR}/board_paths.conf"
    # 完整测试成功后只保留 summary.json,共用 tools/summarize_board_result.py 的汇总规则.
    echo "[TODO] 在开发板执行模型测试并汇总核心指标."
else
    echo "[TODO] 在编译主机转换模型、交叉编译 C++ 程序并上传."
fi
exit 2
