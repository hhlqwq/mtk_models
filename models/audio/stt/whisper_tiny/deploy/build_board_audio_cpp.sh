#!/usr/bin/env bash
# 在 89 交叉编译 Whisper LibriSpeech 的 C++ 板端音频准备程序.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SOURCE="${SCRIPT_DIR}/prepare_board_audio.cpp"
readonly OUTPUT="${SCRIPT_DIR}/prepare_board_audio"
readonly TOOLCHAIN_ROOT="${MTK_G720_CPP_TOOLCHAIN_ROOT:-/data/users/hailong.he/data/MTKG720/cpp_toolchain}"
readonly OPENCV_SOURCE="${TOOLCHAIN_ROOT}/opencv-4.9.0/opencv-4.9.0"
readonly OPENCV_BUILD="${TOOLCHAIN_ROOT}/opencv-4.9.0/build-aarch64-headers"
readonly TARGET_LIBS="${TOOLCHAIN_ROOT}/genio720-libs"
readonly CXX="${CROSS_CXX:-/usr/bin/aarch64-linux-gnu-g++}"

test -x "${CXX}"
test -f "${OPENCV_SOURCE}/modules/core/include/opencv2/core.hpp"
test -f "${OPENCV_BUILD}/opencv2/cvconfig.h"
test -f "${TARGET_LIBS}/libopencv_core.so.409"

"${CXX}" \
    -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic \
    -I"${OPENCV_BUILD}" \
    -I"${OPENCV_SOURCE}/modules/core/include" \
    "${SOURCE}" "${TARGET_LIBS}/libopencv_core.so.409" \
    -Wl,--allow-shlib-undefined -pthread -ldl \
    -o "${OUTPUT}"

file "${OUTPUT}"
sha256sum "${OUTPUT}"
