#!/usr/bin/env bash

# MTK Genio 720 项目公共环境变量.
export MTK_MODELS_ROOT="/data/users/hailong.he/github/mtk_models"
export MTK_SDK_ROOT="/data/users/hailong.he/data/MTKG720"
export MTK_DATASET_ROOT="/data/users/hailong.he/nas_smb/Datasets/open_source/raw"
export MTK_NP_ROOT="${MTK_SDK_ROOT}/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211"
export MTK_NCC_BIN="${MTK_NP_ROOT}/neuron_sdk/host/bin"
export MTK_NCC_LIB="${MTK_NP_ROOT}/neuron_sdk/host/lib"
export MTK_BOARD_HOST="root@192.168.0.92"
export MTK_BOARD_ROOT="/root/hailong.he"
export MTK_CONTAINER_NAME="hhl_g720_311"

export PATH="${MTK_NCC_BIN}:${PATH}"
export LD_LIBRARY_PATH="${MTK_NCC_LIB}:${LD_LIBRARY_PATH:-}"

echo "[MTK] 项目目录: ${MTK_MODELS_ROOT}"
echo "[MTK] NeuroPilot SDK: ${MTK_NP_ROOT}"
echo "[MTK] 数据集目录: ${MTK_DATASET_ROOT}"
echo "[MTK] 目标开发板: ${MTK_BOARD_HOST}"
