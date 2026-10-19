# 可在进入 Docker 后直接 make,或用 BUILD_DIR 覆盖输出目录.
MODEL_DIR := $(abspath ../..)
include $(MTK_MODELS_ROOT)/common/cpp/common.mk

.PHONY: all
all: $(BUILD_DIR)/fastsam_board

$(BUILD_DIR)/fastsam_board: fastsam_board.cpp runtime_g720.cpp fastsam_runtime.h $(COMMON_HEADERS) Makefile $(MTK_MODELS_ROOT)/common/cpp/common.mk
	mkdir -p "$(@D)"
	$(CROSS_CXX) $(CPPFLAGS) $(CXXFLAGS) fastsam_board.cpp runtime_g720.cpp $(LDFLAGS) $(LDLIBS) -o "$@"
	@printf '[C++] 程序路径: %s\n' "$@"
