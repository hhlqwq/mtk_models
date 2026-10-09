// 在 Genio 720 上常驻加载 YAMNet,逐窗口执行真实 NPU 推理并汇总音频分数.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

#include "neuron/api/RuntimeAPI.h"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

// 检查每一个 Runtime 操作的返回值.
void Check(int result, const char* operation) {
  if (result != NEURONRUNTIME_NO_ERROR) {
    throw std::runtime_error(std::string(operation) + ": " + std::to_string(result));
  }
}

struct Quantization {
  float scale = 0;
  int zero = 0;
};

// 读取转换器解析出的实际 IO 量化配置.
Quantization ReadQuantization(std::ifstream& stream) {
  std::string line;
  if (!std::getline(stream, line)) throw std::runtime_error("量化配置缺失.");
  std::replace(line.begin(), line.end(), ',', ' ');
  std::istringstream row(line);
  Quantization value;
  if (!(row >> value.scale >> value.zero) || value.scale <= 0 ||
      value.zero < -128 || value.zero > 127) {
    throw std::runtime_error("量化配置无效.");
  }
  return value;
}

class Model {
 public:
  // 创建硬件 Runtime 并严格验证输入输出数量和原生字节布局.
  explicit Model(const fs::path& path) {
    EnvOptions options{};
    options.deviceKind = kEnvOptHardware;
    options.MDLACoreOption = Auto;
    options.CPUThreadNum = 1;
    options.suppressInputConversion = true;
    options.suppressOutputConversion = true;
    Check(NeuronRuntime_create(&options, &runtime_), "create");
    try {
      Check(NeuronRuntime_loadNetworkFromFile(runtime_, path.c_str()), "load");
      size_t inputs = 0, outputs = 0, input_size = 0, output_size = 0;
      Check(NeuronRuntime_getInputNumber(runtime_, &inputs), "input count");
      Check(NeuronRuntime_getOutputNumber(runtime_, &outputs), "output count");
      Check(NeuronRuntime_getSingleInputPaddedSize(runtime_, &input_size), "input size");
      Check(NeuronRuntime_getOutputPaddedSize(runtime_, 0, &output_size), "output size");
      if (inputs != 1 || outputs != 1 || input_size != 6144 ||
          (output_size != 521 && output_size != 528)) {
        throw std::runtime_error("原生 IO 布局不匹配: " + std::to_string(input_size) +
                                 "," + std::to_string(output_size));
      }
      output_.resize(output_size);
      std::cout << "[MODEL] input=" << input_size << " output=" << output_size << '\n';
    } catch (...) {
      NeuronRuntime_release(runtime_);
      runtime_ = nullptr;
      throw;
    }
  }

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  // 释放常驻 Runtime,异常退出同样执行.
  ~Model() {
    if (runtime_) NeuronRuntime_release(runtime_);
  }

  // 每次重新登记 IO,耗时仅覆盖硬件 inference 调用.
  double Infer(std::vector<int8_t>& input) {
    const BufferAttribute attribute{NON_ION_FD};
    Check(NeuronRuntime_setSingleInput(runtime_, input.data(), input.size(), attribute),
          "set input");
    Check(NeuronRuntime_setOutput(runtime_, 0, output_.data(), output_.size(), attribute),
          "set output");
    const auto start = Clock::now();
    Check(NeuronRuntime_inference(runtime_), "inference");
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  }

  // 返回真实 INT8 类别输出,填充字节由调用方排除.
  const std::vector<int8_t>& Output() const { return output_; }

 private:
  void* runtime_ = nullptr;
  std::vector<int8_t> output_;
};

// 执行全量音频缓存推理,按输入清单顺序输出 521 维窗口平均分数.
int Run(int argc, char** argv) {
  if (argc != 4) throw std::runtime_error("用法: yamnet_eval model.dla config.csv work_dir");
  const fs::path work = argv[3];
  std::ifstream configuration(argv[2]);
  const auto input_quant = ReadQuantization(configuration);
  const auto output_quant = ReadQuantization(configuration);
  Model model(argv[1]);
  std::ifstream manifest(work / "patch_counts.txt");
  std::ifstream features(work / "features.bin", std::ios::binary);
  std::ofstream scores(work / "scores.bin", std::ios::binary);
  std::ofstream timings(work / "timings.csv");
  std::ofstream clip_times(work / "clip_times.csv");
  if (!manifest || !features || !scores || !timings || !clip_times) {
    throw std::runtime_error("输入缓存或输出文件打不开.");
  }
  timings << "clip,patch,npu_ms\n";
  clip_times << "clip,patches,inference_stage_ms\n";
  std::vector<float> patch(6144);
  std::vector<int8_t> quantized(6144);
  int count = 0, clip = 0;
  bool warmed = false;
  while (manifest >> count) {
    if (count <= 0) throw std::runtime_error("窗口数量必须为正.");
    std::vector<float> average(521, 0);
    double warmup_ms = 0;
    const auto start = Clock::now();
    for (int index = 0; index < count; ++index) {
      features.read(reinterpret_cast<char*>(patch.data()), patch.size() * sizeof(float));
      if (!features) throw std::runtime_error("特征文件不足.");
      for (size_t value = 0; value < patch.size(); ++value) {
        if (!std::isfinite(patch[value])) throw std::runtime_error("特征含非有限数.");
        const float integer = std::nearbyint(patch[value] / input_quant.scale) + input_quant.zero;
        quantized[value] = static_cast<int8_t>(std::clamp(integer, -128.0F, 127.0F));
      }
      if (!warmed) {
        const auto warmup_start = Clock::now();
        for (int repeat = 0; repeat < 20; ++repeat) model.Infer(quantized);
        warmup_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - warmup_start).count();
        warmed = true;
      }
      const double npu_ms = model.Infer(quantized);
      timings << clip << ',' << index << ',' << npu_ms << '\n';
      for (int category = 0; category < 521; ++category) {
        average[category] += (static_cast<int>(model.Output()[category]) - output_quant.zero) *
                             output_quant.scale / count;
      }
    }
    const double elapsed = std::chrono::duration<double, std::milli>(
        Clock::now() - start).count() - warmup_ms;
    clip_times << clip << ',' << count << ',' << elapsed << '\n';
    scores.write(reinterpret_cast<char*>(average.data()), average.size() * sizeof(float));
    ++clip;
    if (clip % 50 == 0) std::cout << "[NPU] " << clip << "/2000 音频." << std::endl;
  }
  if (clip != 2000 || features.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("音频覆盖或特征长度不匹配.");
  }
  scores.flush(); timings.flush(); clip_times.flush();
  if (!scores || !timings || !clip_times) throw std::runtime_error("结果文件写入失败.");
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("峰值内存读取失败.");
  std::ofstream memory(work / "peak_rss_kib.txt");
  memory << usage.ru_maxrss << '\n';
  if (!memory) throw std::runtime_error("内存记录写入失败.");
  return 0;
}

}  // namespace

// 将异常转换为明确退出状态,禁止失败后继续生成成功汇总.
int main(int argc, char** argv) {
  try {
    return Run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << std::endl;
    return 1;
  }
}
