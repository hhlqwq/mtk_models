#!/usr/bin/env bash
# 在 89 交叉编译 YOLO-World XL 的 AArch64 ORT C++ 全量评测程序.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
readonly SOURCE="${SCRIPT_DIR}/inference_demo/yoloworld_board_eval.cpp"
readonly OUTPUT="${SCRIPT_DIR}/inference_demo/yoloworld_board_eval"
readonly ORT_HEADER="${REPO_ROOT}/tools/third_party/onnxruntime/onnxruntime_c_api.h"
readonly TOOLCHAIN_ROOT="${MTK_G720_CPP_TOOLCHAIN_ROOT:-/data/users/hailong.he/data/MTKG720/cpp_toolchain}"
readonly OPENCV_SOURCE="${TOOLCHAIN_ROOT}/opencv-4.9.0/opencv-4.9.0"
readonly OPENCV_BUILD="${TOOLCHAIN_ROOT}/opencv-4.9.0/build-aarch64-headers"
readonly TARGET_LIBS="${TOOLCHAIN_ROOT}/genio720-libs"
readonly CXX="${CROSS_CXX:-/usr/bin/aarch64-linux-gnu-g++}"

test -x "${CXX}"
test -s "${ORT_HEADER}"
test -f "${OPENCV_SOURCE}/modules/core/include/opencv2/core.hpp"
test -f "${OPENCV_BUILD}/opencv2/cvconfig.h"

"${CXX}" \
    -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic \
    -Wno-deprecated-enum-enum-conversion \
    -I"$(dirname "${ORT_HEADER}")" \
    -I"${OPENCV_BUILD}" \
    -I"${OPENCV_SOURCE}/modules/core/include" \
    -I"${OPENCV_SOURCE}/modules/imgproc/include" \
    -I"${OPENCV_SOURCE}/modules/imgcodecs/include" \
    "${SOURCE}" \
    "${TARGET_LIBS}/libopencv_imgcodecs.so.409" \
    "${TARGET_LIBS}/libopencv_imgproc.so.409" \
    "${TARGET_LIBS}/libopencv_core.so.409" \
    -Wl,--allow-shlib-undefined -pthread -ldl \
    -o "${OUTPUT}"

file "${OUTPUT}"
sha256sum "${OUTPUT}"
