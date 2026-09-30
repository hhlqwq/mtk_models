#!/usr/bin/env bash
# 模型单脚本入口: 编译主机完成构建与上传,开发板完成测试.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 模型输入文件的绝对路径.
MODEL_SOURCE=""
# 校准数据目录的绝对路径.
CALIBRATION_DIR=""
# 生成产物的目录.
MODEL_OUTPUT_DIR=""
# 板端测试数据目录.
BOARD_DATASET_DIR=""
# 板端部署目录.
BOARD_DEPLOY_DIR=""
# 开发板 SSH 用户和地址.
BOARD_HOST=""

if [[ -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    source "${SCRIPT_DIR}/board_paths.conf"
    echo "[TODO] 在开发板执行模型测试."
else
    echo "[TODO] 在编译主机转换模型、交叉编译 C++ 程序并上传."
fi
exit 2
