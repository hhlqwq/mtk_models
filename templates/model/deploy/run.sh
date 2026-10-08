#!/usr/bin/env bash
# 单脚本两步流程: 在编译主机编译并上传,在开发板执行测试.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 编译主机配置区: 只修改等号右侧的路径或名称.板端使用上传的 board_paths.conf.
if [[ ! -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    # 模型目录: 根据本脚本的位置自动确定.
    MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

    # 1. 模型与校准数据.
    # 模型文件: 原始权重或待编译的 ONNX 完整路径.
    MODEL_SOURCE=""
    # 校准数据: 编译主机与 Docker 可访问的图片目录.
    CALIBRATION_DIR=""

    # 2. 产物与临时目录.
    # 模型输出目录: 转换与编译产物保存在这里.
    MODEL_OUTPUT_DIR="${MODEL_ROOT}/models"
    # 临时构建目录: 缓存和中间文件保存在仓库外.
    BUILD_WORK_DIR="/tmp/hailongcodex/$(date +%F)/model_id"

    # 3. 板端地址与数据.
    # 板端地址: SSH 用户和地址.
    BOARD_HOST=""
    # 板端部署目录: 上传模型、程序和本脚本的目录.
    BOARD_DEPLOY_DIR=""
    # 板端结果目录: 保存本次测试汇总.
    BOARD_RESULTS_DIR="${BOARD_DEPLOY_DIR}/results"
    # 板端数据: 与 ONNX 评测使用相同数据及标签.
    BOARD_DATASET_DIR=""

    # 4. ONNX 精度数据.
    # 浮点精度数据: 编译主机与 Docker 可访问的数据集根目录.
    ONNX_DATASET_DIR=""

    # 5. 编译环境: 通常无需修改.
    # Docker 容器: 编译主机上的 Genio 720 编译环境.
    MTK_G720_CONTAINER=""
    # C++ 编译器: 编译主机上的 AArch64 交叉编译命令.
    CROSS_CXX="aarch64-linux-gnu-g++"
    # SSH 选项: 首次连接接受主机密钥,之后验证保存的密钥.
    SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)
fi

# 板端阶段: 使用上传的配置执行测试并汇总核心指标.
if [[ -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    source "${SCRIPT_DIR}/board_paths.conf"
    # 模型与参数位于 models/,板端程序与评测代码位于 board/.
    echo "[TODO] 在开发板执行模型测试,成功后仅保留 summary.json."
    exit 2
fi

# 编译主机阶段: 检查配置,转换模型,交叉编译并上传.
if (( $# != 0 )); then
    echo "[ERROR] 在编译主机直接运行 bash deploy/run.sh,无需参数." >&2
    exit 2
fi
echo "[TODO] 在编译主机转换模型、评测 ONNX、交叉编译 C++ 程序并上传."
exit 2
