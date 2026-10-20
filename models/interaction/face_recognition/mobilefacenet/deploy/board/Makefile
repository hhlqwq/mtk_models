# 在编译容器中直接 make,模型预处理与后处理位于本模型 C++ 源码.
MODEL_DIR := $(abspath ../..)
include $(MTK_MODELS_ROOT)/common/cpp/common.mk

.PHONY: all
all: $(BUILD_DIR)/mobilefacenet_board

$(BUILD_DIR)/mobilefacenet_board: mobilefacenet_board.cpp $(MTK_MODELS_ROOT)/common/cpp/runtime_g720.cpp $(COMMON_HEADERS) Makefile $(MTK_MODELS_ROOT)/common/cpp/common.mk
	mkdir -p "$(@D)"
	$(CROSS_CXX) $(CPPFLAGS) $(CXXFLAGS) mobilefacenet_board.cpp $(MTK_MODELS_ROOT)/common/cpp/runtime_g720.cpp $(LDFLAGS) $(LDLIBS) -o "$@"
	@printf '[C++] 程序路径: %s\n' "$@"
