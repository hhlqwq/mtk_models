# 复用公共 Neuron 桥接源码,不依赖其他模型目录.
MODEL_DIR := $(abspath ../..)
include $(MTK_MODELS_ROOT)/common/cpp/common.mk

.PHONY: all
all: $(BUILD_DIR)/libneuron_bridge.so

$(BUILD_DIR)/libneuron_bridge.so: $(MTK_MODELS_ROOT)/common/cpp/neuron_bridge.cpp Makefile $(MTK_MODELS_ROOT)/common/cpp/common.mk
	mkdir -p "$(@D)"
	$(CROSS_CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC -shared "$<" $(LDFLAGS) $(LDLIBS) -o "$@"
	@printf '[C++] 桥接库路径: %s\n' "$@"
