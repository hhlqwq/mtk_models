# 可在进入 Docker 后直接 make,或用 BUILD_DIR 覆盖输出目录.
MODEL_DIR := $(abspath ../..)
include $(MTK_MODELS_ROOT)/tools/board/common.mk

.PHONY: all
all: $(BUILD_DIR)/yolov8n_board_eval

$(BUILD_DIR)/yolov8n_board_eval: yolov8n_board_eval.cpp $(COMMON_HEADERS) Makefile $(MTK_MODELS_ROOT)/tools/board/common.mk
	mkdir -p "$(@D)"
	$(CROSS_CXX) $(CPPFLAGS) $(CXXFLAGS) yolov8n_board_eval.cpp $(LDFLAGS) $(LDLIBS) -o "$@"
	@printf '[C++] 程序路径: %s\n' "$@"
