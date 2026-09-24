#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TOOLCHAIN_ROOT="${MTK_G720_CPP_TOOLCHAIN_ROOT:-/data/users/hailong.he/data/MTKG720/cpp_toolchain}"
readonly TARGET_LIBS="${TOOLCHAIN_ROOT}/genio720-libs"
readonly OPENCV_SOURCE="${TOOLCHAIN_ROOT}/opencv-4.9.0/opencv-4.9.0"
readonly OPENCV_BUILD="${TOOLCHAIN_ROOT}/opencv-4.9.0/build-aarch64-headers"
readonly NEURON_INCLUDE="${MTK_NEURON_INCLUDE:-/data/users/hailong.he/data/MTKG720/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/include}"
readonly CXX="${CROSS_CXX:-/usr/bin/aarch64-linux-gnu-g++}"

test -x "${CXX}"
test -f "${OPENCV_SOURCE}/modules/core/include/opencv2/core.hpp"
test -f "${OPENCV_BUILD}/opencv2/cvconfig.h"
test -f "${TARGET_LIBS}/libopencv_core.so.409"
test -f "${NEURON_INCLUDE}/neuron/api/RuntimeAPI.h"
test -f "${TARGET_LIBS}/libneuronusdk_runtime.mtk.so.8"

build_binary() {
    # 交叉编译一个仅依赖 Neuron Runtime 的 AArch64 程序.
    local source_name="$1"
    local output_name="$2"
    local source="${SCRIPT_DIR}/inference_demo/${source_name}"
    local output="${SCRIPT_DIR}/inference_demo/${output_name}"
    test -f "${source}"
    "${CXX}" \
        -std=c++20 -O2 -DNDEBUG -Wall -Wextra -Wpedantic \
        -I"${NEURON_INCLUDE}" \
        "${source}" \
        "${TARGET_LIBS}/libneuronusdk_runtime.mtk.so.8" \
        -Wl,--allow-shlib-undefined -pthread -ldl \
        -o "${output}"
    file "${output}"
    sha256sum "${output}"
}

echo "[1/2] 交叉编译 Whisper-Tiny 板端音频准备程序."
"${CXX}" \
    -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic \
    -I"${OPENCV_BUILD}" \
    -I"${OPENCV_SOURCE}/modules/core/include" \
    "${SCRIPT_DIR}/prepare_board_audio.cpp" \
    "${TARGET_LIBS}/libopencv_core.so.409" \
    -Wl,--allow-shlib-undefined -pthread -ldl \
    -o "${SCRIPT_DIR}/prepare_board_audio"
file "${SCRIPT_DIR}/prepare_board_audio"
sha256sum "${SCRIPT_DIR}/prepare_board_audio"

echo "[2/2] 交叉编译 Whisper-Tiny 持久化批量评测程序."
build_binary whisper_board_eval.cpp whisper_board_eval
