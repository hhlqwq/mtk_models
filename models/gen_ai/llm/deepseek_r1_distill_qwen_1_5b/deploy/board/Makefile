# 在已加载公共工具链的容器中直接 make.
MODEL_DIR := $(abspath ../..)
include $(MTK_MODELS_ROOT)/common/cpp/common.mk

.PHONY: all
all: $(BUILD_DIR)/libneuron_bridge.so

$(BUILD_DIR)/libneuron_bridge.so: neuron_bridge.cpp Makefile $(MTK_MODELS_ROOT)/common/cpp/common.mk
	mkdir -p "$(@D)"
	$(CROSS_CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC -shared "$<" $(LDFLAGS) $(LDLIBS) -o "$@"
	@printf '[C++] 桥接库路径: %s\n' "$@"
