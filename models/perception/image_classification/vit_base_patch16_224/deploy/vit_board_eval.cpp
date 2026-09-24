// 在 Genio 720 板端以 C++ 完成 ImageNet val 全量预处理和 NPU 推理.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "neuron/api/RuntimeAPI.h"

namespace fs = std::filesystem;

struct Options {
  fs::path model;
  fs::path images;
  fs::path quantization;
  fs::path predictions;
};

// 解析固定的全量测试参数.
Options ParseArgs(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) throw std::invalid_argument("参数缺少值.");
    const std::string key = argv[index];
    if (key == "--model") options.model = argv[index + 1];
    else if (key == "--images") options.images = argv[index + 1];
    else if (key == "--quantization") options.quantization = argv[index + 1];
    else if (key == "--predictions") options.predictions = argv[index + 1];
    else throw std::invalid_argument("未知参数: " + key);
  }
  if (options.model.empty() || options.images.empty() ||
      options.quantization.empty() || options.predictions.empty()) {
    throw std::invalid_argument("缺少模型、图片、量化参数或输出路径.");
  }
  return options;
}

// 从 89 生成的元数据中读取指定数值并校验唯一性.
double ReadQuantValue(const fs::path& path, const std::string& key) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("无法读取量化元数据.");
  const std::string data(std::istreambuf_iterator<char>{input}, {});
  const std::regex pattern("\\\"" + key +
                           "\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)");
  std::sregex_iterator match(data.begin(), data.end(), pattern);
  if (match == std::sregex_iterator() ||
      std::next(match) != std::sregex_iterator()) {
    throw std::runtime_error("量化参数缺失或重复: " + key);
  }
  return std::stod((*match)[1].str());
}

// 严格检查官方编号连续的 50000 张 ImageNet val 图片.
std::vector<fs::path> ListImages(const fs::path& directory) {
  std::vector<fs::path> images;
  for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
    if (entry.is_regular_file() && entry.path().extension() == ".JPEG") {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());
  if (images.size() != 50000) {
    throw std::runtime_error("ImageNet val 图片数量不是 50000.");
  }
  for (size_t index = 0; index < images.size(); ++index) {
    char expected[40];
    std::snprintf(expected, sizeof(expected), "ILSVRC2012_val_%08zu.JPEG", index + 1);
    if (images[index].filename() != expected) {
      throw std::runtime_error("ImageNet val 图片编号不连续: " +
                               images[index].filename().string());
    }
  }
  return images;
}

// 复用项目 OpenCV Resize/Crop、RGB、NCHW INT8 量化口径.
std::vector<int8_t> Preprocess(const fs::path& path, double scale,
                               int zero_point) {
  const cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (image.empty()) throw std::runtime_error("无法读取图片: " + path.string());
  const double ratio = 256.0 / std::min(image.rows, image.cols);
  const int width = static_cast<int>(std::nearbyint(image.cols * ratio));
  const int height = static_cast<int>(std::nearbyint(image.rows * ratio));
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(width, height), 0, 0, cv::INTER_CUBIC);
  const int top = (height - 224) / 2;
  const int left = (width - 224) / 2;
  const cv::Mat crop = resized(cv::Rect(left, top, 224, 224));
  std::vector<int8_t> input(3 * 224 * 224);
  for (int row = 0; row < 224; ++row) {
    const auto* pixels = crop.ptr<cv::Vec3b>(row);
    for (int column = 0; column < 224; ++column) {
      for (int channel = 0; channel < 3; ++channel) {
        const double value = pixels[column][2 - channel] / 255.0;
        const int quantized = static_cast<int>(std::nearbyint(value / scale)) +
                              zero_point;
        input[channel * 224 * 224 + row * 224 + column] =
            static_cast<int8_t>(std::clamp(quantized, -128, 127));
      }
    }
  }
  return input;
}

// 将 DLA 常驻加载,避免每张图重新启动 Runtime.
class NeuronModel {
 public:
  explicit NeuronModel(const fs::path& path) {
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
      Check(NeuronRuntime_getSingleInputPaddedSize(runtime_, &input_size),
            "input size");
      Check(NeuronRuntime_getOutputPaddedSize(runtime_, 0, &output_size),
            "output size");
      if (inputs != 1 || outputs != 1 || input_size != 150528 ||
          (output_size != 1000 && output_size != 1008)) {
        throw std::runtime_error("ViT DLA 输入输出形状不匹配.");
      }
      output_.resize(output_size);
    } catch (...) {
      NeuronRuntime_release(runtime_);
      runtime_ = nullptr;
      throw;
    }
  }

  NeuronModel(const NeuronModel&) = delete;
  NeuronModel& operator=(const NeuronModel&) = delete;

  // 释放板端 Runtime 资源.
  ~NeuronModel() {
    if (runtime_ != nullptr) NeuronRuntime_release(runtime_);
  }

  // 对一张图片执行硬件推理,返回类别排序和纯 NPU 耗时.
  std::pair<std::array<int, 5>, double> Infer(std::vector<int8_t>& input) {
    const BufferAttribute attribute{NON_ION_FD};
    Check(NeuronRuntime_setSingleInput(runtime_, input.data(), input.size(),
                                       attribute), "set input");
    Check(NeuronRuntime_setOutput(runtime_, 0, output_.data(), output_.size(),
                                  attribute), "set output");
    const auto start = std::chrono::steady_clock::now();
    Check(NeuronRuntime_inference(runtime_), "inference");
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    std::array<int, 1000> indices;
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + 5, indices.end(),
                      [this](int left, int right) {
                        if (output_[left] != output_[right]) {
                          return output_[left] > output_[right];
                        }
                        return left < right;
                      });
    return {{{indices[0], indices[1], indices[2], indices[3], indices[4]}},
            elapsed_ms};
  }

 private:
  // 转换 Runtime 状态码为带步骤的错误.
  static void Check(int result, const std::string& step) {
    if (result != 0) {
      throw std::runtime_error("NeuronRuntime " + step + " 失败: " +
                               std::to_string(result));
    }
  }

  void* runtime_ = nullptr;
  std::vector<int8_t> output_;
};

// 执行全量推理并逐张持久化预测供 Python 指标阶段读取.
int main(int argc, char** argv) {
  try {
    const Options options = ParseArgs(argc, argv);
    const double scale = ReadQuantValue(options.quantization, "input_scale");
    const int zero_point = static_cast<int>(ReadQuantValue(
        options.quantization, "input_zero_point"));
    if (scale <= 0) throw std::runtime_error("输入量化 scale 非法.");
    const std::vector<fs::path> images = ListImages(options.images);
    fs::create_directories(options.predictions.parent_path());
    std::ofstream output(options.predictions);
    if (!output) throw std::runtime_error("无法写入预测清单.");
    NeuronModel model(options.model);
    for (size_t index = 0; index < images.size(); ++index) {
      std::vector<int8_t> input = Preprocess(images[index], scale, zero_point);
      const auto [top5, npu_ms] = model.Infer(input);
      output << "{\"index\":" << index << ",\"image\":\""
             << images[index].filename().string() << "\",\"top5\":["
             << top5[0] << ',' << top5[1] << ',' << top5[2] << ','
             << top5[3] << ',' << top5[4] << "],\"npu_ms\":"
             << npu_ms << "}\n";
      output.flush();
      if (!output) throw std::runtime_error("预测清单写入失败.");
      if ((index + 1) % 100 == 0 || index + 1 == images.size()) {
        std::cout << "[PROGRESS] ViT " << index + 1 << '/' << images.size()
                  << " 张.\n";
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << '\n';
    return 1;
  }
}
