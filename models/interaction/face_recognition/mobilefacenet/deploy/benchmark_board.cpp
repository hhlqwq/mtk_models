// 在板端常驻加载 DLA, 统计预热后的 Neuron Runtime 调用耗时.

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

#include "neuron/api/RuntimeAPI.h"

namespace fs = std::filesystem;

struct Options {
  fs::path model;
  fs::path input;
  fs::path report;
  int warmup = 10;
  int repeats = 100;
};

// 解析性能测试参数, 拒绝缺失项和非正重复次数.
Options ParseArgs(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) throw std::invalid_argument("参数缺少值.");
    const std::string key = argv[index];
    if (key == "--model") options.model = argv[index + 1];
    else if (key == "--input") options.input = argv[index + 1];
    else if (key == "--report") options.report = argv[index + 1];
    else if (key == "--warmup") options.warmup = std::stoi(argv[index + 1]);
    else if (key == "--repeats") options.repeats = std::stoi(argv[index + 1]);
    else throw std::invalid_argument("未知参数: " + key);
  }
  if (options.model.empty() || options.input.empty() || options.report.empty() ||
      options.warmup < 0 || options.repeats <= 0) {
    throw std::invalid_argument("模型、输入、报告或重复次数无效.");
  }
  return options;
}

// 检查 Neuron Runtime 返回值并指出失败步骤.
void CheckRuntime(int result, const std::string& step) {
  if (result != 0) {
    throw std::runtime_error("NeuronRuntime " + step + " 失败: " +
                             std::to_string(result));
  }
}

// 按字节读取正式测试使用的真实量化输入.
std::vector<char> ReadInput(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("无法读取输入: " + path.string());
  std::vector<char> bytes(std::istreambuf_iterator<char>{stream}, {});
  if (bytes.empty()) throw std::runtime_error("输入文件为空.");
  return bytes;
}

// 从排序后的耗时中按线性插值计算分位数.
double Percentile(const std::vector<double>& sorted, double fraction) {
  const double position = (sorted.size() - 1) * fraction;
  const auto lower = static_cast<size_t>(position);
  const auto upper = std::min(lower + 1, sorted.size() - 1);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

// 加载硬件模型并对同一真实输入执行预热与正式重复推理.
int Run(const Options& options) {
  std::vector<char> input = ReadInput(options.input);
  EnvOptions environment{};
  environment.deviceKind = kEnvOptHardware;
  environment.MDLACoreOption = Auto;
  environment.CPUThreadNum = 1;
  environment.suppressInputConversion = true;
  environment.suppressOutputConversion = true;
  void* runtime = nullptr;
  CheckRuntime(NeuronRuntime_create(&environment, &runtime), "create");
  try {
    CheckRuntime(NeuronRuntime_loadNetworkFromFile(runtime, options.model.c_str()),
                 "load");
    size_t input_count = 0;
    size_t output_count = 0;
    size_t input_size = 0;
    CheckRuntime(NeuronRuntime_getInputNumber(runtime, &input_count), "input count");
    CheckRuntime(NeuronRuntime_getOutputNumber(runtime, &output_count), "output count");
    CheckRuntime(NeuronRuntime_getSingleInputPaddedSize(runtime, &input_size),
                 "input size");
    if (input_count != 1 || output_count == 0 || input.size() != input_size) {
      throw std::runtime_error("真实输入与 DLA 输入契约不匹配.");
    }
    std::vector<std::vector<char>> outputs(output_count);
    for (size_t index = 0; index < output_count; ++index) {
      size_t output_size = 0;
      CheckRuntime(NeuronRuntime_getOutputPaddedSize(runtime, index, &output_size),
                   "output size");
      if (output_size == 0) throw std::runtime_error("DLA 输出大小为零.");
      outputs[index].resize(output_size);
    }
    const BufferAttribute attribute{NON_ION_FD};
    CheckRuntime(NeuronRuntime_setSingleInput(runtime, input.data(), input.size(),
                                              attribute), "set input");
    std::vector<double> latency_ms;
    latency_ms.reserve(options.repeats);
    for (int index = 0; index < options.warmup + options.repeats; ++index) {
      for (size_t output = 0; output < outputs.size(); ++output) {
        CheckRuntime(NeuronRuntime_setOutput(runtime, output,
                                             outputs[output].data(),
                                             outputs[output].size(), attribute),
                     "set output");
      }
      const auto start = std::chrono::steady_clock::now();
      CheckRuntime(NeuronRuntime_inference(runtime), "inference");
      const double elapsed = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
      if (index >= options.warmup) latency_ms.push_back(elapsed);
      if ((index + 1) % 25 == 0 || index + 1 == options.warmup + options.repeats) {
        std::cout << "[PROGRESS] " << index + 1 << "/"
                  << options.warmup + options.repeats << std::endl;
      }
    }
    std::sort(latency_ms.begin(), latency_ms.end());
    const double mean = std::accumulate(latency_ms.begin(), latency_ms.end(), 0.0) /
                        latency_ms.size();
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
      throw std::runtime_error("无法读取进程峰值内存.");
    }
    fs::create_directories(options.report.parent_path());
    std::ofstream report(options.report);
    if (!report) throw std::runtime_error("无法写入性能报告.");
    report << "{\n  \"backend\": \"cpp_neuron_runtime_hw\",\n"
           << "  \"timing_scope\": \"NeuronRuntime_inference API call\",\n"
           << "  \"warmup\": " << options.warmup << ",\n"
           << "  \"repeats\": " << options.repeats << ",\n"
           << "  \"mean_ms\": " << mean << ",\n"
           << "  \"p50_ms\": " << Percentile(latency_ms, 0.5) << ",\n"
           << "  \"p95_ms\": " << Percentile(latency_ms, 0.95) << ",\n"
           << "  \"min_ms\": " << latency_ms.front() << ",\n"
           << "  \"max_ms\": " << latency_ms.back() << ",\n"
           << "  \"peak_rss_kib\": " << usage.ru_maxrss << "\n}\n";
    std::cout << "[RESULT] " << options.report << std::endl;
  } catch (...) {
    NeuronRuntime_release(runtime);
    throw;
  }
  NeuronRuntime_release(runtime);
  return 0;
}

// 打印明确的错误信息, 供板端脚本检查退出码.
int main(int argc, char** argv) {
  try {
    return Run(ParseArgs(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << std::endl;
    return 1;
  }
}
