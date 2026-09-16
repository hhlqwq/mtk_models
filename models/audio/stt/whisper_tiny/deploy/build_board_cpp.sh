#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SOURCE="${SCRIPT_DIR}/inference_demo/inspect_whisper_io.cpp"
readonly OUTPUT="${SCRIPT_DIR}/inference_demo/inspect_whisper_io"
readonly TOOLCHAIN_ROOT="${MTK_G720_CPP_TOOLCHAIN_ROOT:-/data/users/hailong.he/data/MTKG720/cpp_toolchain}"
readonly TARGET_LIBS="${TOOLCHAIN_ROOT}/genio720-libs"
readonly NEURON_INCLUDE="${MTK_NEURON_INCLUDE:-/data/users/hailong.he/data/MTKG720/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/include}"
readonly CXX="${CROSS_CXX:-/usr/bin/aarch64-linux-gnu-g++}"

test -x "${CXX}"
test -f "${SOURCE}"
test -f "${NEURON_INCLUDE}/neuron/api/RuntimeAPI.h"
test -f "${TARGET_LIBS}/libneuronusdk_runtime.mtk.so.8"

echo "[1/2] 交叉编译 Whisper-Tiny 板端 I/O 检查程序."
"${CXX}" \
    -std=c++20 -O2 -DNDEBUG -Wall -Wextra -Wpedantic \
    -I"${NEURON_INCLUDE}" \
    "${SOURCE}" \
    "${TARGET_LIBS}/libneuronusdk_runtime.mtk.so.8" \
    -Wl,--allow-shlib-undefined -pthread -ldl \
    -o "${OUTPUT}"

echo "[2/2] 检查目标架构和文件哈希."
file "${OUTPUT}"
sha256sum "${OUTPUT}"
