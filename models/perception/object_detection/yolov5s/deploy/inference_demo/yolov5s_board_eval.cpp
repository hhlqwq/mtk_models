// 在 Genio 720 板端完成 YOLOv5s 预处理、NPU 推理和后处理.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "neuron/api/RuntimeAPI.h"

namespace fs = std::filesystem;

namespace {

  constexpr int kImageSize = 640;
  constexpr int kInputChannels = 3;
  constexpr int kValuesPerAnchor = 85;
  constexpr int kClassCount = 80;
  constexpr float kInputScale = 0.003921568859368563F;
  constexpr int kInputZeroPoint = -128;

  constexpr std::array<int, 3> kHeadSizes = { 80, 40, 20 };
  constexpr std::array<int, 3> kPaddedWidths = { 80, 48, 32 };
  constexpr std::array<float, 3> kStrides = { 8.0F, 16.0F, 32.0F };
  constexpr std::array<float, 3> kOutputScales = {
      0.08984976261854172F,
      0.07972777634859085F,
      0.08616162091493607F,
  };
  constexpr std::array<int, 3> kOutputZeroPoints = { 75, 61, 55 };
  constexpr std::array<std::array<std::array<float, 2>, 3>, 3> kAnchors = { {
      {{{1.25F, 1.625F}, {2.0F, 3.75F}, {4.125F, 2.875F}}},
      {{{1.875F, 3.8125F}, {3.875F, 2.8125F}, {3.6875F, 7.4375F}}},
      {{{3.625F, 2.8125F}, {4.875F, 6.1875F}, {11.65625F, 10.1875F}}},
  } };
  constexpr std::array<int, 80> kCocoCategoryIds = {
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17,
      18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
      37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
      54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
      74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
  };

  struct Options {
    fs::path model;
    fs::path images;
    fs::path output_dir;
    int limit = 0;
    int warmup = 20;
    int progress_interval = 50;
    int max_detections = 300;
    float confidence = 0.001F;
    float iou = 0.6F;
  };

  struct LetterboxInfo {
    float scale = 1.0F;
    int left = 0;
    int top = 0;
    int original_width = 0;
    int original_height = 0;
  };

  struct Detection {
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
    float score = 0.0F;
    int class_id = 0;
  };

struct Timing {
  double preprocess_ms = 0.0;
  double npu_ms = 0.0;
  double postprocess_ms = 0.0;
  double e2e_ms = 0.0;
};

  // 检查 Runtime 返回码,并在失败时中止.
  void CheckRuntime(int result, const std::string& operation) {
    if (result != NEURONRUNTIME_NO_ERROR) {
      throw std::runtime_error(operation + " 失败,Neuron Runtime 错误码: " +
        std::to_string(result));
    }
  }

  // 计算稳定的 Sigmoid.
  float Sigmoid(float value) {
    const float clipped = std::clamp(value, -30.0F, 30.0F);
    return 1.0F / (1.0F + std::exp(-clipped));
  }

  // 计算两个 xyxy 检测框的 IoU.
  float BoxIou(const Detection& left, const Detection& right) {
    const float x1 = std::max(left.x1, right.x1);
    const float y1 = std::max(left.y1, right.y1);
    const float x2 = std::min(left.x2, right.x2);
    const float y2 = std::min(left.y2, right.y2);
    const float intersection = std::max(0.0F, x2 - x1) *
      std::max(0.0F, y2 - y1);
    const float left_area = std::max(0.0F, left.x2 - left.x1) *
      std::max(0.0F, left.y2 - left.y1);
    const float right_area = std::max(0.0F, right.x2 - right.x1) *
      std::max(0.0F, right.y2 - right.y1);
    return intersection /
      std::max(left_area + right_area - intersection, 1.0e-9F);
  }

  // 解析正整数参数.
  int ParseInteger(const std::string& value, const std::string& name) {
    try {
      return std::stoi(value);
    }
    catch (const std::exception&) {
      throw std::invalid_argument("无效整数参数 " + name + ": " + value);
    }
  }

  // 解析浮点参数.
  float ParseFloat(const std::string& value, const std::string& name) {
    try {
      return std::stof(value);
    }
    catch (const std::exception&) {
      throw std::invalid_argument("无效浮点参数 " + name + ": " + value);
    }
  }

  // 解析命令行参数.
  Options ParseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
      const std::string key = argv[index];
      if (index + 1 >= argc) {
        throw std::invalid_argument("参数缺少值: " + key);
      }
      const std::string value = argv[++index];
      if (key == "--model") {
        options.model = value;
      }
      else if (key == "--images") {
        options.images = value;
      }
      else if (key == "--output-dir") {
        options.output_dir = value;
      }
      else if (key == "--limit") {
        options.limit = ParseInteger(value, key);
      }
      else if (key == "--warmup") {
        options.warmup = ParseInteger(value, key);
      }
      else if (key == "--progress-interval") {
        options.progress_interval = ParseInteger(value, key);
      }
      else if (key == "--max-det") {
        options.max_detections = ParseInteger(value, key);
      }
      else if (key == "--confidence") {
        options.confidence = ParseFloat(value, key);
      }
      else if (key == "--iou") {
        options.iou = ParseFloat(value, key);
      }
      else {
        throw std::invalid_argument("未知参数: " + key);
      }
    }
    if (options.model.empty() || options.images.empty() ||
      options.output_dir.empty()) {
      throw std::invalid_argument(
        "必须提供 --model、--images 和 --output-dir.");
    }
    if (options.limit < 0 || options.warmup < 0 ||
      options.progress_interval <= 0 || options.max_detections <= 0) {
      throw std::invalid_argument("次数参数必须位于有效范围.");
    }
    return options;
  }

  // 返回目录中的 COCO JPEG 文件,并按文件名排序.
  std::vector<fs::path> ListImages(const fs::path& directory, int limit) {
    std::vector<fs::path> images;
    for (const auto& entry : fs::directory_iterator(directory)) {
      if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
        images.push_back(entry.path());
      }
    }
    std::sort(images.begin(), images.end());
    if (limit > 0 && images.size() > static_cast<size_t>(limit)) {
      images.resize(limit);
    }
    if (images.empty()) {
      throw std::runtime_error("没有找到 JPEG 图片: " + directory.string());
    }
    return images;
  }

  // 将 COCO 文件名转换为数值 image_id.
  int64_t ImageIdFromPath(const fs::path& image_path) {
    return std::stoll(image_path.stem().string());
  }

  // 从完成清单中读取已处理 image_id.
  std::unordered_set<int64_t> LoadCompleted(const fs::path& path) {
    std::unordered_set<int64_t> completed;
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line)) {
      if (!line.empty()) {
        completed.insert(std::stoll(line));
      }
    }
    return completed;
  }

  // 将图片转换为模型要求的 NCHW INT8 输入.
  std::pair<std::vector<int8_t>, LetterboxInfo> Preprocess(
    const fs::path& image_path) {
    const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error("无法读取图片: " + image_path.string());
    }
    LetterboxInfo info;
    info.original_width = image.cols;
    info.original_height = image.rows;
    info.scale = std::min(static_cast<float>(kImageSize) / image.cols,
      static_cast<float>(kImageSize) / image.rows);
    const int resized_width = std::lround(image.cols * info.scale);
    const int resized_height = std::lround(image.rows * info.scale);
    info.left = (kImageSize - resized_width) / 2;
    info.top = (kImageSize - resized_height) / 2;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_width, resized_height), 0.0,
      0.0, cv::INTER_LINEAR);
    cv::Mat canvas(kImageSize, kImageSize, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(info.left, info.top, resized_width,
      resized_height)));

    std::vector<int8_t> input(kInputChannels * kImageSize * kImageSize);
    const size_t plane_size = kImageSize * kImageSize;
    for (int row = 0; row < kImageSize; ++row) {
      const auto* pixels = canvas.ptr<cv::Vec3b>(row);
      for (int column = 0; column < kImageSize; ++column) {
        for (int channel = 0; channel < kInputChannels; ++channel) {
          const uint8_t rgb_value = pixels[column][2 - channel];
          const float normalized = static_cast<float>(rgb_value) / 255.0F;
          const int quantized = static_cast<int>(
            std::nearbyint(normalized / kInputScale)) + kInputZeroPoint;
          input[channel * plane_size + row * kImageSize + column] =
            static_cast<int8_t>(std::clamp(quantized, -128, 127));
        }
      }
    }
    return { std::move(input), info };
  }

  class NeuronModel {
  public:
    // 创建硬件 Runtime 并一次性加载 DLA.
    explicit NeuronModel(const fs::path& model_path) {
      EnvOptions options{};
      options.deviceKind = kEnvOptHardware;
      options.MDLACoreOption = Auto;
      options.CPUThreadNum = 1;
      options.suppressInputConversion = true;
      options.suppressOutputConversion = true;
      CheckRuntime(NeuronRuntime_create(&options, &runtime_),
        "NeuronRuntime_create");
      try {
        CheckRuntime(NeuronRuntime_loadNetworkFromFile(
          runtime_, model_path.c_str()),
          "NeuronRuntime_loadNetworkFromFile");
        ValidateIo();
      }
      catch (...) {
        NeuronRuntime_release(runtime_);
        runtime_ = nullptr;
        throw;
      }
    }

    NeuronModel(const NeuronModel&) = delete;
    NeuronModel& operator=(const NeuronModel&) = delete;

    // 释放 Runtime 资源.
    ~NeuronModel() {
      if (runtime_ != nullptr) {
        NeuronRuntime_release(runtime_);
      }
    }

    // 复用已加载模型执行一次硬件推理,并返回耗时.
    double Infer(const std::vector<int8_t>& input) {
      if (input.size() != input_.size()) {
        throw std::invalid_argument("输入字节数不匹配.");
      }
      std::copy(input.begin(), input.end(), input_.begin());
      const BufferAttribute attribute{ NON_ION_FD };
      // Runtime 8.2 会在登记时消费普通缓冲区,每次推理必须重新登记 I/O.
      CheckRuntime(NeuronRuntime_setSingleInput(
        runtime_, input_.data(), input_.size(), attribute),
        "NeuronRuntime_setSingleInput");
      for (size_t index = 0; index < outputs_.size(); ++index) {
        CheckRuntime(NeuronRuntime_setOutput(
          runtime_, index, outputs_[index].data(),
          outputs_[index].size(), attribute),
          "NeuronRuntime_setOutput(" + std::to_string(index) + ")");
      }
      const auto start = std::chrono::steady_clock::now();
      CheckRuntime(NeuronRuntime_inference(runtime_),
        "NeuronRuntime_inference");
      const auto end = std::chrono::steady_clock::now();
      return std::chrono::duration<double, std::milli>(end - start).count();
    }

    // 返回指定检测头的原生对齐输出.
    const std::vector<int8_t>& output(size_t index) const {
      return outputs_.at(index);
    }

  private:
    // 验证模型 I/O 数量和对齐后缓冲区大小.
    void ValidateIo() {
      size_t input_count = 0;
      size_t output_count = 0;
      CheckRuntime(NeuronRuntime_getInputNumber(runtime_, &input_count),
        "NeuronRuntime_getInputNumber");
      CheckRuntime(NeuronRuntime_getOutputNumber(runtime_, &output_count),
        "NeuronRuntime_getOutputNumber");
      if (input_count != 1 || output_count != 3) {
        throw std::runtime_error("模型 I/O 数量不是预期的 1 输入 3 输出.");
      }
      size_t input_size = 0;
      CheckRuntime(NeuronRuntime_getSingleInputPaddedSize(runtime_, &input_size),
        "NeuronRuntime_getSingleInputPaddedSize");
      if (input_size != kInputChannels * kImageSize * kImageSize) {
        throw std::runtime_error("模型输入大小不匹配: " +
          std::to_string(input_size));
      }
      input_.resize(input_size);
      for (size_t index = 0; index < outputs_.size(); ++index) {
        size_t output_size = 0;
        CheckRuntime(NeuronRuntime_getOutputPaddedSize(runtime_, index,
          &output_size),
          "NeuronRuntime_getOutputPaddedSize");
        const size_t expected = kValuesPerAnchor * 3 * kHeadSizes[index] *
          kPaddedWidths[index];
        if (output_size != expected) {
          throw std::runtime_error(
            "模型输出对齐大小不匹配,head=" + std::to_string(index) +
            ",actual=" + std::to_string(output_size) +
            ",expected=" + std::to_string(expected));
        }
        outputs_[index].resize(output_size);
      }
    }

    void* runtime_ = nullptr;
    std::vector<int8_t> input_;
    std::array<std::vector<int8_t>, 3> outputs_;
  };

  // 从 MDLA 原生 NCHW 行对齐输出中读取一个值.
  float ReadOutput(const std::vector<int8_t>& output, int head_index,
    int channel, int row, int column) {
    const int height = kHeadSizes[head_index];
    const int padded_width = kPaddedWidths[head_index];
    const size_t offset =
      (static_cast<size_t>(channel) * height + row) * padded_width + column;
    return (static_cast<int>(output[offset]) - kOutputZeroPoints[head_index]) *
      kOutputScales[head_index];
  }

  // 解码三个 YOLO 检测头并筛选置信度.
  std::vector<Detection> Decode(const NeuronModel& model,
    float confidence) {
    std::vector<Detection> detections;
    detections.reserve(4096);
    for (int head = 0; head < 3; ++head) {
      const auto& output = model.output(head);
      const int size = kHeadSizes[head];
      for (int anchor = 0; anchor < 3; ++anchor) {
        const int base = anchor * kValuesPerAnchor;
        for (int row = 0; row < size; ++row) {
          for (int column = 0; column < size; ++column) {
            const float objectness = Sigmoid(
              ReadOutput(output, head, base + 4, row, column));
            int class_id = 0;
            float class_probability = 0.0F;
            for (int class_index = 0; class_index < kClassCount;
              ++class_index) {
              const float probability = Sigmoid(ReadOutput(
                output, head, base + 5 + class_index, row, column));
              if (probability > class_probability) {
                class_probability = probability;
                class_id = class_index;
              }
            }
            const float score = objectness * class_probability;
            if (score < confidence) {
              continue;
            }
            const float center_x =
              (Sigmoid(ReadOutput(output, head, base, row, column)) * 2.0F +
                column - 0.5F) *
              kStrides[head];
            const float center_y =
              (Sigmoid(ReadOutput(output, head, base + 1, row, column)) *
                2.0F +
                row - 0.5F) *
              kStrides[head];
            const float width_factor =
              Sigmoid(ReadOutput(output, head, base + 2, row, column)) *
              2.0F;
            const float height_factor =
              Sigmoid(ReadOutput(output, head, base + 3, row, column)) *
              2.0F;
            const float width = width_factor * width_factor *
              kAnchors[head][anchor][0] * kStrides[head];
            const float height = height_factor * height_factor *
              kAnchors[head][anchor][1] * kStrides[head];
            detections.push_back({
                center_x - width / 2.0F,
                center_y - height / 2.0F,
                center_x + width / 2.0F,
                center_y + height / 2.0F,
                score,
                class_id,
              });
          }
        }
      }
    }
    return detections;
  }

  // 执行与 torchvision batched_nms 等价的逐类别 NMS.
  std::vector<Detection> ApplyNms(std::vector<Detection> detections,
    float iou_threshold, int max_detections) {
    std::sort(detections.begin(), detections.end(),
      [](const Detection& left, const Detection& right) {
        return left.score > right.score;
      });
    std::vector<Detection> kept;
    kept.reserve(std::min<size_t>(detections.size(), max_detections));
    for (const Detection& candidate : detections) {
      bool suppressed = false;
      for (const Detection& selected : kept) {
        if (candidate.class_id == selected.class_id &&
          BoxIou(candidate, selected) > iou_threshold) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) {
        kept.push_back(candidate);
        if (kept.size() == static_cast<size_t>(max_detections)) {
          break;
        }
      }
    }
    return kept;
  }

  // 将检测框映射回原图并裁剪到图像边界.
  void Rescale(std::vector<Detection>* detections,
    const LetterboxInfo& info) {
    for (Detection& detection : *detections) {
      detection.x1 = std::clamp((detection.x1 - info.left) / info.scale,
        0.0F, static_cast<float>(info.original_width));
      detection.x2 = std::clamp((detection.x2 - info.left) / info.scale,
        0.0F, static_cast<float>(info.original_width));
      detection.y1 = std::clamp((detection.y1 - info.top) / info.scale,
        0.0F, static_cast<float>(info.original_height));
      detection.y2 = std::clamp((detection.y2 - info.top) / info.scale,
        0.0F, static_cast<float>(info.original_height));
    }
  }

  // 转义 JSON 字符串中的特殊字符.
  std::string EscapeJson(const std::string& value) {
    std::ostringstream output;
    for (const char character : value) {
      if (character == '\\' || character == '"') {
        output << '\\';
      }
      output << character;
    }
    return output.str();
  }

  // 将单张图片的检测结果追加为 JSONL.
  void AppendPredictions(std::ofstream* output, int64_t image_id,
    const std::vector<Detection>& detections) {
    *output << std::fixed;
    for (const Detection& detection : detections) {
      *output << "{\"image_id\":" << image_id
        << ",\"category_id\":" << kCocoCategoryIds[detection.class_id]
        << ",\"bbox\":[" << std::setprecision(3) << detection.x1 << ','
        << detection.y1 << ',' << detection.x2 - detection.x1 << ','
        << detection.y2 - detection.y1 << "],\"score\":"
        << std::setprecision(5) << detection.score << "}\n";
    }
    output->flush();
  }

  // 将 JSONL 检测记录转换为 COCOeval 接受的 JSON 数组.
  void FinalizePredictions(const fs::path& jsonl_path,
    const fs::path& json_path) {
    std::ifstream input(jsonl_path);
    std::ofstream output(json_path);
    output << "[\n";
    std::string line;
    bool first = true;
    while (std::getline(input, line)) {
      if (line.empty()) {
        continue;
      }
      if (!first) {
        output << ",\n";
      }
      output << line;
      first = false;
    }
    output << "\n]\n";
  }

  // 计算延迟统计并写入 JSON.
  void WriteSummary(const fs::path& path, const std::vector<Timing>& timings,
    size_t total_images) {
    auto summarize = [&timings](auto accessor) {
      std::vector<double> values;
      values.reserve(timings.size());
      for (const Timing& timing : timings) {
        values.push_back(accessor(timing));
      }
      std::sort(values.begin(), values.end());
      const auto percentile = [&values](double ratio) {
        const double position = ratio * (values.size() - 1);
        const size_t lower = static_cast<size_t>(std::floor(position));
        const size_t upper = static_cast<size_t>(std::ceil(position));
        const double weight = position - lower;
        return values[lower] * (1.0 - weight) + values[upper] * weight;
        };
      const double mean =
        std::accumulate(values.begin(), values.end(), 0.0) / values.size();
      return std::array<double, 6>{mean, percentile(0.5), percentile(0.9),
        percentile(0.95), values.front(),
        values.back()};
      };
    const auto preprocess = summarize(
      [](const Timing& value) { return value.preprocess_ms; });
    const auto npu = summarize([](const Timing& value) { return value.npu_ms; });
    const auto postprocess = summarize(
      [](const Timing& value) { return value.postprocess_ms; });
    const auto e2e = summarize([](const Timing& value) { return value.e2e_ms; });
    const auto write_metric = [](std::ofstream* output, const std::string& name,
      const std::array<double, 6>& values,
      bool trailing_comma) {
        *output << "  \"" << EscapeJson(name) << "\": {\"mean\": "
          << values[0] << ", \"p50\": " << values[1]
          << ", \"p90\": " << values[2] << ", \"p95\": "
          << values[3] << ", \"min\": " << values[4]
          << ", \"max\": " << values[5] << "}"
          << (trailing_comma ? "," : "") << "\n";
      };

  std::ofstream output(path);
  output << std::fixed << std::setprecision(6);
  output << "{\n  \"processed_images\": " << total_images << ",\n";
    write_metric(&output, "preprocess_ms", preprocess, true);
    write_metric(&output, "npu_ms", npu, true);
    write_metric(&output, "postprocess_ms", postprocess, true);
    write_metric(&output, "e2e_ms", e2e, false);
    output << "}\n";
  }

  // 执行完整板端评测流程.
  void Run(const Options& options) {
    fs::create_directories(options.output_dir);
    const auto images = ListImages(options.images, options.limit);
    const fs::path predictions_jsonl = options.output_dir / "predictions.jsonl";
    const fs::path predictions_json = options.output_dir / "predictions.json";
    const fs::path completed_path = options.output_dir / "processed_ids.txt";
    const fs::path timings_path = options.output_dir / "timings.csv";
    auto completed = LoadCompleted(completed_path);

    std::cout << "[1/4] 加载 DLA: " << options.model << std::endl;
    NeuronModel model(options.model);
    const auto warmup_input = Preprocess(images.front()).first;
    std::cout << "[2/4] NPU 预热 " << options.warmup << " 次." << std::endl;
    for (int index = 0; index < options.warmup; ++index) {
      model.Infer(warmup_input);
    }

    std::ofstream predictions(predictions_jsonl, std::ios::app);
    std::ofstream completed_output(completed_path, std::ios::app);
    const bool timings_exist = fs::exists(timings_path) &&
      fs::file_size(timings_path) > 0;
    std::ofstream timings_output(timings_path, std::ios::app);
    if (!timings_exist) {
      timings_output << "image_id,preprocess_ms,npu_ms,postprocess_ms,e2e_ms,"
        "detections\n";
    }

    std::vector<Timing> current_timings;
    size_t processed_count = completed.size();
    std::cout << "[3/4] 板端逐图评测,共 " << images.size()
      << " 张,已完成 " << completed.size() << " 张." << std::endl;
    for (const fs::path& image_path : images) {
      const int64_t image_id = ImageIdFromPath(image_path);
      if (completed.contains(image_id)) {
        continue;
      }
      const auto e2e_start = std::chrono::steady_clock::now();
      const auto preprocess_start = e2e_start;
      auto [input, letterbox_info] = Preprocess(image_path);
      const auto preprocess_end = std::chrono::steady_clock::now();
      const double npu_ms = model.Infer(input);
      const auto postprocess_start = std::chrono::steady_clock::now();
      auto detections = ApplyNms(Decode(model, options.confidence), options.iou,
        options.max_detections);
      Rescale(&detections, letterbox_info);
      const auto end = std::chrono::steady_clock::now();

      const Timing timing{
          std::chrono::duration<double, std::milli>(preprocess_end -
                                                    preprocess_start)
              .count(),
          npu_ms,
          std::chrono::duration<double, std::milli>(end - postprocess_start)
              .count(),
          std::chrono::duration<double, std::milli>(end - e2e_start).count(),
      };
      AppendPredictions(&predictions, image_id, detections);
      timings_output << std::fixed << std::setprecision(6) << image_id << ','
        << timing.preprocess_ms << ',' << timing.npu_ms << ','
        << timing.postprocess_ms << ',' << timing.e2e_ms << ','
        << detections.size() << '\n';
      timings_output.flush();
      completed_output << image_id << '\n';
      completed_output.flush();
      completed.insert(image_id);
      current_timings.push_back(timing);
      ++processed_count;
      if (processed_count % options.progress_interval == 0 ||
        processed_count == images.size()) {
        std::cout << "[board] " << processed_count << '/' << images.size()
          << ",image_id=" << image_id << std::endl;
      }
    }

    predictions.close();
    completed_output.close();
    timings_output.close();
    FinalizePredictions(predictions_jsonl, predictions_json);
    if (!current_timings.empty()) {
      WriteSummary(options.output_dir / "timing_summary_current_run.json",
        current_timings, completed.size());
    }
    std::cout << "[4/4] 评测推理完成,COCO 预测: " << predictions_json
      << std::endl;
  }

}  // namespace

// 解析参数并执行板端评测.
int main(int argc, char** argv) {
  try {
    Run(ParseOptions(argc, argv));
    return 0;
  }
  catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << std::endl;
    return 1;
  }
}
