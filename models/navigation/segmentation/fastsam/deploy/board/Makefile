# FastSAM 板端程序构建,目前仅实现 G720,G5100 等待 SDK 与硬件适配.
PLATFORM ?= g720
ifeq ($(PLATFORM),g5100)
$(error G5100 尚未适配,请准备对应 SDK、运行库和开发板后再启用)
endif
ifneq ($(PLATFORM),g720)
$(error 目前仅考虑 G720 和 G5100,当前可构建平台为 g720)
endif
ifndef CROSS_CXX
$(error 请先通过 docker/enter_container.sh 进入容器)
endif
ifndef MTK_G720_CPP_TOOLCHAIN_ROOT
$(error 缺少 MTK_G720_CPP_TOOLCHAIN_ROOT,请加载平台环境)
endif
ifndef MTK_NEURON_INCLUDE
$(error 缺少 MTK_NEURON_INCLUDE,请加载平台环境)
endif
ifndef MTK_MODEL_STORAGE_ROOT
$(error 缺少 MTK_MODEL_STORAGE_ROOT,请加载本地资源配置)
endif

BUILD_DIR ?= $(MTK_MODEL_STORAGE_ROOT)/navigation/segmentation/fastsam/build
TARGET := $(BUILD_DIR)/fastsam_board
SOURCES := fastsam_board.cpp runtime_g720.cpp
HEADER := fastsam_runtime.h
OPENCV_SOURCE := $(MTK_G720_CPP_TOOLCHAIN_ROOT)/opencv-4.9.0/opencv-4.9.0
OPENCV_BUILD := $(MTK_G720_CPP_TOOLCHAIN_ROOT)/opencv-4.9.0/build-aarch64-headers
LIB_DIR := $(MTK_G720_CPP_TOOLCHAIN_ROOT)/genio720-libs
RUNTIME_HEADER := $(MTK_NEURON_INCLUDE)/neuron/api/RuntimeAPI.h
LIBRARIES := $(LIB_DIR)/libneuronusdk_runtime.mtk.so.8 \
             $(LIB_DIR)/libopencv_imgcodecs.so.409 \
             $(LIB_DIR)/libopencv_imgproc.so.409 \
             $(LIB_DIR)/libopencv_core.so.409
CPPFLAGS += -I"$(OPENCV_BUILD)" \
            -I"$(OPENCV_SOURCE)/modules/core/include" \
            -I"$(OPENCV_SOURCE)/modules/imgproc/include" \
            -I"$(OPENCV_SOURCE)/modules/imgcodecs/include" \
            -I"$(MTK_NEURON_INCLUDE)"
CXXFLAGS += -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic
LDFLAGS += -Wl,--allow-shlib-undefined
LDLIBS += -pthread -ldl

.PHONY: all
all: $(TARGET)

# 保留头文件依赖,源码、库文件或构建规则变化时重新编译.
$(TARGET): $(SOURCES) $(HEADER) $(RUNTIME_HEADER) $(LIBRARIES) Makefile
	mkdir -p "$(@D)"
	$(CROSS_CXX) $(CPPFLAGS) $(CXXFLAGS) \
	    $(SOURCES) $(LIBRARIES) $(LDFLAGS) $(LDLIBS) -o "$@"
	@printf '[C++] 程序路径: %s\n' "$@"
