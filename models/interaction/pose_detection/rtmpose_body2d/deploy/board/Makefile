# 可在进入 Docker 后直接 make,或用 BUILD_DIR 覆盖输出目录.
MODEL_DIR := $(abspath ../..)
include $(MTK_MODELS_ROOT)/tools/board/common.mk

.PHONY: all
all: $(BUILD_DIR)/rtmpose_board_eval $(BUILD_DIR)/prepare_eval_manifest

$(BUILD_DIR)/rtmpose_board_eval: rtmpose_board_eval.cpp $(COMMON_HEADERS) Makefile $(MTK_MODELS_ROOT)/tools/board/common.mk
	mkdir -p "$(@D)"
	$(CROSS_CXX) $(CPPFLAGS) $(CXXFLAGS) rtmpose_board_eval.cpp $(LDFLAGS) $(LDLIBS) -o "$@"
	@printf '[C++] 程序路径: %s\n' "$@"

$(BUILD_DIR)/prepare_eval_manifest: prepare_eval_manifest.cpp $(COMMON_HEADERS) Makefile $(MTK_MODELS_ROOT)/tools/board/common.mk
	mkdir -p "$(@D)"
	$(CROSS_CXX) $(CPPFLAGS) $(CXXFLAGS) prepare_eval_manifest.cpp $(LDFLAGS) $(LDLIBS) -o "$@"
	@printf '[C++] 程序路径: %s\n' "$@"
