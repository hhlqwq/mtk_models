# 可在进入 Docker 后直接 make,或用 BUILD_DIR 覆盖输出目录.
MODEL_DIR := $(abspath ../..)
include $(MTK_MODELS_ROOT)/common/cpp/common.mk

.PHONY: all
all: $(BUILD_DIR)/whisper_board_eval $(BUILD_DIR)/prepare_board_audio

$(BUILD_DIR)/whisper_board_eval: whisper_board_eval.cpp $(COMMON_HEADERS) Makefile $(MTK_MODELS_ROOT)/common/cpp/common.mk
	mkdir -p "$(@D)"
	$(CROSS_CXX) $(CPPFLAGS) $(CXXFLAGS) whisper_board_eval.cpp $(LDFLAGS) $(LDLIBS) -o "$@"
	@printf '[C++] 程序路径: %s\n' "$@"

$(BUILD_DIR)/prepare_board_audio: prepare_board_audio.cpp $(COMMON_HEADERS) Makefile $(MTK_MODELS_ROOT)/common/cpp/common.mk
	mkdir -p "$(@D)"
	$(CROSS_CXX) $(CPPFLAGS) $(CXXFLAGS) prepare_board_audio.cpp $(LDFLAGS) $(LDLIBS) -o "$@"
	@printf '[C++] 程序路径: %s\n' "$@"
