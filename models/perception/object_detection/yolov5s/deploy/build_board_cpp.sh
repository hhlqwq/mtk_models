#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SOURCE="${SCRIPT_DIR}/inference_demo/yolov5s_board_eval.cpp"
readonly OUTPUT="${SCRIPT_DIR}/inference_demo/yolov5s_board_eval"
readonly TOOLCHAIN_ROOT="${MTK_G720_CPP_TOOLCHAIN_ROOT:-/data/users/hailong.he/data/MTKG720/cpp_toolchain}"
readonly OPENCV_SOURCE="${TOOLCHAIN_ROOT}/opencv-4.9.0/opencv-4.9.0"
readonly OPENCV_BUILD="${TOOLCHAIN_ROOT}/opencv-4.9.0/build-aarch64-headers"
readonly TARGET_LIBS="${TOOLCHAIN_ROOT}/genio720-libs"
readonly NEURON_INCLUDE="${MTK_NEURON_INCLUDE:-/data/users/hailong.he/data/MTKG720/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/include}"
readonly CXX="${CROSS_CXX:-/usr/bin/aarch64-linux-gnu-g++}"

test -x "${CXX}"
test -f "${OPENCV_SOURCE}/modules/core/include/opencv2/core.hpp"
test -f "${OPENCV_BUILD}/opencv2/cvconfig.h"
test -f "${NEURON_INCLUDE}/neuron/api/RuntimeAPI.h"
test -f "${TARGET_LIBS}/libneuronusdk_runtime.mtk.so.8"

echo "[1/2] 交叉编译 Genio 720 YOLOv5s C++ 评测程序."
"${CXX}" \
    -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic \
    -I"${OPENCV_BUILD}" \
    -I"${OPENCV_SOURCE}/modules/core/include" \
    -I"${OPENCV_SOURCE}/modules/imgproc/include" \
    -I"${OPENCV_SOURCE}/modules/imgcodecs/include" \
    -I"${NEURON_INCLUDE}" \
    "${SOURCE}" \
    "${TARGET_LIBS}/libneuronusdk_runtime.mtk.so.8" \
    "${TARGET_LIBS}/libopencv_imgcodecs.so.409" \
    "${TARGET_LIBS}/libopencv_imgproc.so.409" \
    "${TARGET_LIBS}/libopencv_core.so.409" \
    -Wl,--allow-shlib-undefined -pthread -ldl \
    -o "${OUTPUT}"

echo "[2/2] 检查目标架构."
file "${OUTPUT}"
sha256sum "${OUTPUT}"
