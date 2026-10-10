// 提供 Python 调用所需的 Neuron 硬件 Runtime 接口,模型运算全部交给 MDLA.
#include <cstddef>
#include <cstdint>
#include "neuron/api/RuntimeAPI.h"

extern "C" {

// 创建硬件 Runtime 并加载 DLA,失败时释放资源并返回错误码.
int CreateModel(const char* path, void** runtime) {
  EnvOptions options{};
  options.deviceKind = kEnvOptHardware;
  options.MDLACoreOption = Auto;
  options.CPUThreadNum = 1;
  options.suppressInputConversion = true;
  options.suppressOutputConversion = true;
  int code = NeuronRuntime_create(&options, runtime);
  if (code != 0) return code;
  code = NeuronRuntime_loadNetworkFromFile(*runtime, path);
  if (code != 0) {
    NeuronRuntime_release(*runtime);
    *runtime = nullptr;
  }
  return code;
}

// 返回输入输出数量,供 Python 校验导出契约.
int ModelCounts(void* runtime, size_t* inputs, size_t* outputs) {
  int code = NeuronRuntime_getInputNumber(runtime, inputs);
  if (code != 0) return code;
  return NeuronRuntime_getOutputNumber(runtime, outputs);
}

// 返回指定输入或输出的原生缓冲区大小.
int ModelSize(void* runtime, size_t index, int output, size_t* bytes) {
  if (output) return NeuronRuntime_getOutputPaddedSize(runtime, index, bytes);
  return NeuronRuntime_getInputPaddedSize(runtime, index, bytes);
}

// 绑定固定数量的输入输出,校验缓冲区大小后执行真实硬件推理.
int RunModel(void* runtime, void** inputs, size_t* input_bytes,
             size_t input_count, void** outputs, size_t* output_bytes,
             size_t output_count) {
  size_t actual_inputs = 0;
  size_t actual_outputs = 0;
  int code = ModelCounts(runtime, &actual_inputs, &actual_outputs);
  if (code != 0) return code;
  if (input_count != actual_inputs || output_count != actual_outputs) return -1001;
  const BufferAttribute attribute{NON_ION_FD};
  for (size_t index = 0; index < input_count; ++index) {
    size_t expected = 0;
    code = ModelSize(runtime, index, 0, &expected);
    if (code != 0) return code;
    if (input_bytes[index] != expected) return -1002;
    code = NeuronRuntime_setInput(runtime, index, inputs[index], expected, attribute);
    if (code != 0) return code;
  }
  for (size_t index = 0; index < output_count; ++index) {
    size_t expected = 0;
    code = ModelSize(runtime, index, 1, &expected);
    if (code != 0) return code;
    if (output_bytes[index] != expected) return -1003;
    code = NeuronRuntime_setOutput(runtime, index, outputs[index], expected, attribute);
    if (code != 0) return code;
  }
  return NeuronRuntime_inference(runtime);
}

// 释放本模型 Runtime,不修改板端系统运行库.
void ReleaseModel(void* runtime) {
  if (runtime != nullptr) NeuronRuntime_release(runtime);
}
}
