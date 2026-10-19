# 环境在进入 Docker 时加载,模型 Makefile 仅声明源码和程序目标.
PLATFORM ?= g720
ifneq ($(PLATFORM),g720)
$(error 目前仅实现 G720; G5100 需匹配 SDK 与开发板验证后启用)
endif
ifndef CROSS_CXX
$(error 请先通过 docker/enter_container.sh 进入容器)
endif
ifndef MTK_MODELS_ROOT
$(error 缺少项目根目录环境)
endif
BUILD_DIR ?= $(shell python3 "$(MTK_MODELS_ROOT)/tools/storage.py" --print-build-dir "$(MODEL_DIR)/config.yaml")
OPENCV_SOURCE := $(MTK_G720_CPP_TOOLCHAIN_ROOT)/opencv-4.9.0/opencv-4.9.0
OPENCV_BUILD := $(MTK_G720_CPP_TOOLCHAIN_ROOT)/opencv-4.9.0/build-aarch64-headers
LIB_DIR := $(MTK_G720_CPP_TOOLCHAIN_ROOT)/genio720-libs
CPPFLAGS += -I"$(MTK_MODELS_ROOT)" -I"$(MTK_NEURON_INCLUDE)" -I"$(OPENCV_BUILD)" \
    -I"$(OPENCV_SOURCE)/modules/core/include" -I"$(OPENCV_SOURCE)/modules/imgproc/include" \
    -I"$(OPENCV_SOURCE)/modules/imgcodecs/include"
CXXFLAGS += -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic
LDFLAGS += -Wl,--allow-shlib-undefined
LDLIBS += "$(LIB_DIR)/libneuronusdk_runtime.mtk.so.8" \
    "$(LIB_DIR)/libopencv_imgcodecs.so.409" "$(LIB_DIR)/libopencv_imgproc.so.409" \
    "$(LIB_DIR)/libopencv_core.so.409" -pthread -ldl
COMMON_HEADERS := $(wildcard $(MTK_MODELS_ROOT)/common/cpp/*.h)
.DEFAULT_GOAL := all
