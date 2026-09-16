// 检查 Whisper-Tiny DLA 在板端 Runtime 中的硬件对齐 I/O.

#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "neuron/api/RuntimeAPI.h"

namespace fs = std::filesystem;

// 检查 Runtime 返回码并转换为异常.
void CheckRuntime(int code, const std::string& operation) {
  if (code != 0) {
    throw std::runtime_error(operation + " 失败,code=" +
                             std::to_string(code));
  }
}

// 输出一个 DLA 的输入输出数量、对齐字节数和四维布局.
void InspectModel(const fs::path& model_path) {
  EnvOptions options{};
  options.deviceKind = kEnvOptHardware;
  options.MDLACoreOption = Auto;
  options.CPUThreadNum = 1;
  options.suppressInputConversion = true;
  options.suppressOutputConversion = true;

  void* runtime = nullptr;
  CheckRuntime(NeuronRuntime_create(&options, &runtime),
               "NeuronRuntime_create");
  try {
    CheckRuntime(NeuronRuntime_loadNetworkFromFile(runtime,
                                                   model_path.c_str()),
                 "NeuronRuntime_loadNetworkFromFile");
    size_t input_count = 0;
    size_t output_count = 0;
    CheckRuntime(NeuronRuntime_getInputNumber(runtime, &input_count),
                 "NeuronRuntime_getInputNumber");
    CheckRuntime(NeuronRuntime_getOutputNumber(runtime, &output_count),
                 "NeuronRuntime_getOutputNumber");
    std::cout << "[MODEL] " << model_path << " inputs=" << input_count
              << " outputs=" << output_count << '\n';

    for (size_t index = 0; index < input_count; ++index) {
      size_t bytes = 0;
      RuntimeAPIDimensions dimensions{};
      CheckRuntime(NeuronRuntime_getInputPaddedSize(runtime, index, &bytes),
                   "NeuronRuntime_getInputPaddedSize");
      CheckRuntime(NeuronRuntime_getInputPaddedDimensions(
                       runtime, index, &dimensions),
                   "NeuronRuntime_getInputPaddedDimensions");
      std::cout << "[INPUT] index=" << index << " bytes=" << bytes
                << " dims=";
      for (uint32_t dimension : dimensions.dimensions) {
        std::cout << dimension << ' ';
      }
      std::cout << '\n';
    }
    for (size_t index = 0; index < output_count; ++index) {
      size_t bytes = 0;
      RuntimeAPIDimensions dimensions{};
      CheckRuntime(NeuronRuntime_getOutputPaddedSize(runtime, index, &bytes),
                   "NeuronRuntime_getOutputPaddedSize");
      CheckRuntime(NeuronRuntime_getOutputPaddedDimensions(
                       runtime, index, &dimensions),
                   "NeuronRuntime_getOutputPaddedDimensions");
      std::cout << "[OUTPUT] index=" << index << " bytes=" << bytes
                << " dims=";
      for (uint32_t dimension : dimensions.dimensions) {
        std::cout << dimension << ' ';
      }
      std::cout << '\n';
    }
  } catch (...) {
    NeuronRuntime_release(runtime);
    throw;
  }
  NeuronRuntime_release(runtime);
}

// 读取命令行中的 DLA 并逐个检查.
int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      throw std::invalid_argument(
          "用法: inspect_whisper_io <model.dla> [model.dla ...]");
    }
    for (int index = 1; index < argc; ++index) {
      InspectModel(argv[index]);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << '\n';
    return 1;
  }
}
