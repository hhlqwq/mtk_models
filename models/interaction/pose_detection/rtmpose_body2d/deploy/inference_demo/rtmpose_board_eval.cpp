// 在 Genio 720 板端完成 RTMPose WholeBody 常驻 C++ 推理.

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
#include <sys/resource.h>
#include <unordered_set>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "neuron/api/RuntimeAPI.h"

namespace fs = std::filesystem;

namespace {

constexpr int kInputWidth = 192;
constexpr int kInputHeight = 256;
constexpr int kInputChannels = 3;
constexpr int kKeypointCount = 133;
constexpr int kSimccXLength = 384;
constexpr int kSimccYLength = 512;
constexpr float kScalePadding = 1.25F;
constexpr int kInputZeroPoint = -128;
constexpr float kOutputXScale = 0.0020384122617542744F;
constexpr int kOutputXZeroPoint = -24;
constexpr float kOutputYScale = 0.0029528678860515356F;
constexpr int kOutputYZeroPoint = -20;

struct Options {
  fs::path model;
  fs::path images;
  fs::path manifest;
  fs::path output_dir;
  int limit = 0;
  int warmup = 20;
  int progress_interval = 500;
};

struct DetectionInput {
  int64_t image_id = 0;
  int64_t detection_id = 0;
  float bbox_score = 0.0F;
  float x = 0.0F;
  float y = 0.0F;
  float width = 0.0F;
  float height = 0.0F;
};

struct Geometry {
  float center_x = 0.0F;
  float center_y = 0.0F;
  float scale_width = 0.0F;
  float scale_height = 0.0F;
};

struct Timing {
  double preprocess_ms = 0.0;
  double npu_ms = 0.0;
  double postprocess_ms = 0.0;
};

// 检查 Runtime 返回码,失败时立即中止.
void CheckRuntime(int result, const std::string& operation) {
  if (result != NEURONRUNTIME_NO_ERROR) {
    throw std::runtime_error(operation + " 失败,错误码: " +
                             std::to_string(result));
  }
}

// 解析非负整数参数.
int ParseInteger(const std::string& value, const std::string& name) {
  try {
    return std::stoi(value);
  } catch (const std::exception&) {
    throw std::invalid_argument("无效整数参数 " + name + ": " + value);
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
    } else if (key == "--images") {
      options.images = value;
    } else if (key == "--manifest") {
      options.manifest = value;
    } else if (key == "--output-dir") {
      options.output_dir = value;
    } else if (key == "--limit") {
      options.limit = ParseInteger(value, key);
    } else if (key == "--warmup") {
      options.warmup = ParseInteger(value, key);
    } else if (key == "--progress-interval") {
      options.progress_interval = ParseInteger(value, key);
    } else {
      throw std::invalid_argument("未知参数: " + key);
    }
  }
  if (options.model.empty() || options.images.empty() ||
      options.manifest.empty() || options.output_dir.empty()) {
    throw std::invalid_argument(
        "必须提供 --model、--images、--manifest 和 --output-dir.");
  }
  if (options.limit < 0 || options.warmup < 0 ||
      options.progress_interval <= 0) {
    throw std::invalid_argument("次数参数必须位于有效范围.");
  }
  return options;
}

// 使用制表符拆分一行清单.
std::vector<std::string> SplitTabs(const std::string& line) {
  std::vector<std::string> values;
  std::stringstream stream(line);
  std::string value;
  while (std::getline(stream, value, '\t')) {
    values.push_back(value);
  }
  return values;
}

// 读取检测框清单并按 limit 截断.
std::vector<DetectionInput> LoadManifest(const fs::path& path, int limit) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("无法读取检测框清单: " + path.string());
  }
  std::string line;
  std::getline(input, line);
  if (line != "image_id\tdetection_id\tbbox_score\tx\ty\tw\th") {
    throw std::runtime_error("检测框清单表头不匹配.");
  }
  std::vector<DetectionInput> records;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    const auto values = SplitTabs(line);
    if (values.size() != 7) {
      throw std::runtime_error("检测框清单列数错误: " + line);
    }
    records.push_back({std::stoll(values[0]), std::stoll(values[1]),
                       std::stof(values[2]), std::stof(values[3]),
                       std::stof(values[4]), std::stof(values[5]),
                       std::stof(values[6])});
    if (limit > 0 && records.size() == static_cast<size_t>(limit)) {
      break;
    }
  }
  if (records.empty()) {
    throw std::runtime_error("检测框清单为空.");
  }
  return records;
}

// 将 COCO image_id 转换为十二位 JPEG 文件名.
fs::path ImagePath(const fs::path& directory, int64_t image_id) {
  std::ostringstream name;
  name << std::setw(12) << std::setfill('0') << image_id << ".jpg";
  return directory / name.str();
}

// 计算固定输入比例和 1.25 倍边距后的中心与尺度.
Geometry ComputeGeometry(const DetectionInput& detection) {
  Geometry geometry;
  geometry.center_x = detection.x + detection.width * 0.5F;
  geometry.center_y = detection.y + detection.height * 0.5F;
  geometry.scale_width = detection.width;
  geometry.scale_height = detection.height;
  constexpr float aspect_ratio =
      static_cast<float>(kInputWidth) / kInputHeight;
  if (geometry.scale_width > geometry.scale_height * aspect_ratio) {
    geometry.scale_height = geometry.scale_width / aspect_ratio;
  } else {
    geometry.scale_width = geometry.scale_height * aspect_ratio;
  }
  geometry.scale_width *= kScalePadding;
  geometry.scale_height *= kScalePadding;
  return geometry;
}

// 将原图中的人体框仿射裁剪为 NCHW BGR INT8 输入.
std::pair<std::vector<int8_t>, Geometry> Preprocess(
    const cv::Mat& image, const DetectionInput& detection) {
  const Geometry geometry = ComputeGeometry(detection);
  const float half_width = geometry.scale_width * 0.5F;
  const float half_height = geometry.scale_height * 0.5F;
  const std::array<cv::Point2f, 3> source = {
      cv::Point2f(geometry.center_x - half_width,
                  geometry.center_y - half_height),
      cv::Point2f(geometry.center_x + half_width,
                  geometry.center_y - half_height),
      cv::Point2f(geometry.center_x - half_width,
                  geometry.center_y + half_height),
  };
  const std::array<cv::Point2f, 3> target = {
      cv::Point2f(0.0F, 0.0F),
      cv::Point2f(static_cast<float>(kInputWidth), 0.0F),
      cv::Point2f(0.0F, static_cast<float>(kInputHeight)),
  };
  const cv::Mat transform = cv::getAffineTransform(source.data(), target.data());
  cv::Mat crop;
  cv::warpAffine(image, crop, transform, cv::Size(kInputWidth, kInputHeight),
                 cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

  const size_t plane_size = kInputWidth * kInputHeight;
  std::vector<int8_t> output(kInputChannels * plane_size);
  for (int row = 0; row < kInputHeight; ++row) {
    const auto* pixels = crop.ptr<cv::Vec3b>(row);
    for (int column = 0; column < kInputWidth; ++column) {
      for (int channel = 0; channel < kInputChannels; ++channel) {
        output[channel * plane_size + row * kInputWidth + column] =
            static_cast<int8_t>(static_cast<int>(pixels[column][channel]) +
                                kInputZeroPoint);
      }
    }
  }
  return {std::move(output), geometry};
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
      CheckRuntime(
          NeuronRuntime_loadNetworkFromFile(runtime_, model_path.c_str()),
          "NeuronRuntime_loadNetworkFromFile");
      ValidateIo();
    } catch (...) {
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

  // 复用已加载模型执行一次推理并返回 NPU 耗时.
  double Infer(const std::vector<int8_t>& input) {
    if (input.size() != input_.size()) {
      throw std::invalid_argument("输入字节数不匹配.");
    }
    std::copy(input.begin(), input.end(), input_.begin());
    const BufferAttribute attribute{NON_ION_FD};
    CheckRuntime(NeuronRuntime_setSingleInput(
                     runtime_, input_.data(), input_.size(), attribute),
                 "NeuronRuntime_setSingleInput");
    for (size_t index = 0; index < outputs_.size(); ++index) {
      CheckRuntime(NeuronRuntime_setOutput(runtime_, index,
                                           outputs_[index].data(),
                                           outputs_[index].size(), attribute),
                   "NeuronRuntime_setOutput(" + std::to_string(index) + ")");
    }
    const auto start = std::chrono::steady_clock::now();
    CheckRuntime(NeuronRuntime_inference(runtime_), "NeuronRuntime_inference");
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  }

  // 返回指定 SimCC 输出缓冲区.
  const std::vector<int8_t>& output(size_t index) const {
    return outputs_.at(index);
  }

 private:
  // 验证模型 I/O 数量和原生缓冲区大小.
  void ValidateIo() {
    size_t input_count = 0;
    size_t output_count = 0;
    CheckRuntime(NeuronRuntime_getInputNumber(runtime_, &input_count),
                 "NeuronRuntime_getInputNumber");
    CheckRuntime(NeuronRuntime_getOutputNumber(runtime_, &output_count),
                 "NeuronRuntime_getOutputNumber");
    if (input_count != 1 || output_count != 2) {
      throw std::runtime_error("模型 I/O 数量不是预期的 1 输入 2 输出.");
    }
    size_t input_size = 0;
    CheckRuntime(NeuronRuntime_getSingleInputPaddedSize(runtime_, &input_size),
                 "NeuronRuntime_getSingleInputPaddedSize");
    if (input_size != kInputChannels * kInputHeight * kInputWidth) {
      throw std::runtime_error("模型输入大小不匹配: " +
                               std::to_string(input_size));
    }
    input_.resize(input_size);
    const std::array<size_t, 2> expected = {
        kKeypointCount * kSimccXLength,
        kKeypointCount * kSimccYLength,
    };
    for (size_t index = 0; index < outputs_.size(); ++index) {
      size_t output_size = 0;
      CheckRuntime(NeuronRuntime_getOutputPaddedSize(runtime_, index,
                                                     &output_size),
                   "NeuronRuntime_getOutputPaddedSize");
      if (output_size != expected[index]) {
        throw std::runtime_error("模型输出大小不匹配,index=" +
                                 std::to_string(index) + ",actual=" +
                                 std::to_string(output_size));
      }
      outputs_[index].resize(output_size);
    }
  }

  void* runtime_ = nullptr;
  std::vector<int8_t> input_;
  std::array<std::vector<int8_t>, 2> outputs_;
};

// 将一份 133 点 SimCC 输出追加为 JSONL.
void AppendPrediction(std::ofstream* output, const DetectionInput& detection,
                      const Geometry& geometry, const NeuronModel& model) {
  const auto& pred_x = model.output(0);
  const auto& pred_y = model.output(1);
  *output << std::fixed << std::setprecision(6)
          << "{\"detection_id\":" << detection.detection_id
          << ",\"image_id\":" << detection.image_id
          << ",\"bbox_score\":" << detection.bbox_score
          << ",\"area\":" << geometry.scale_width * geometry.scale_height
          << ",\"keypoints\":[";
  for (int keypoint = 0; keypoint < kKeypointCount; ++keypoint) {
    const auto x_begin = pred_x.begin() + keypoint * kSimccXLength;
    const auto y_begin = pred_y.begin() + keypoint * kSimccYLength;
    const int x_index = static_cast<int>(
        std::distance(x_begin, std::max_element(x_begin,
                                                x_begin + kSimccXLength)));
    const int y_index = static_cast<int>(
        std::distance(y_begin, std::max_element(y_begin,
                                                y_begin + kSimccYLength)));
    const float x_score =
        (static_cast<int>(x_begin[x_index]) - kOutputXZeroPoint) * kOutputXScale;
    const float y_score =
        (static_cast<int>(y_begin[y_index]) - kOutputYZeroPoint) * kOutputYScale;
    const float score = std::min(x_score, y_score);
    float source_x = -1.0F;
    float source_y = -1.0F;
    if (score > 0.0F) {
      const float input_x = x_index / 2.0F;
      const float input_y = y_index / 2.0F;
      source_x = input_x / kInputWidth * geometry.scale_width +
                 geometry.center_x - geometry.scale_width * 0.5F;
      source_y = input_y / kInputHeight * geometry.scale_height +
                 geometry.center_y - geometry.scale_height * 0.5F;
    }
    if (keypoint > 0) {
      *output << ',';
    }
    *output << source_x << ',' << source_y << ',' << score;
  }
  *output << "]}\n";
}

// 写出首个样本的原生输入输出,用于跨后端诊断.
void DumpBuffer(const fs::path& path, const void* data, size_t size) {
  std::ofstream output(path, std::ios::binary);
  output.write(static_cast<const char*>(data),
               static_cast<std::streamsize>(size));
  if (!output) {
    throw std::runtime_error("无法写入诊断缓冲区: " + path.string());
  }
}

// 从完成清单中读取已处理的 detection_id.
std::unordered_set<int64_t> LoadCompleted(const fs::path& path) {
  std::unordered_set<int64_t> completed;
  std::ifstream input(path);
  int64_t value = 0;
  while (input >> value) {
    completed.insert(value);
  }
  return completed;
}

// 写出本次常驻评测的延迟摘要.
void WriteTimingSummary(const fs::path& path,
                        const std::vector<Timing>& timings) {
  const auto mean = [&timings](auto accessor) {
    return std::accumulate(
               timings.begin(), timings.end(), 0.0,
               [&accessor](double sum, const Timing& timing) {
                 return sum + accessor(timing);
               }) /
           timings.size();
  };
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    throw std::runtime_error("getrusage 读取失败.");
  }
  std::ofstream output(path);
  output << std::fixed << std::setprecision(6)
         << "{\n  \"processed_detections\": " << timings.size()
         << ",\n  \"preprocess_mean_ms\": "
         << mean([](const Timing& value) { return value.preprocess_ms; })
         << ",\n  \"npu_mean_ms\": "
         << mean([](const Timing& value) { return value.npu_ms; })
         << ",\n  \"postprocess_mean_ms\": "
         << mean([](const Timing& value) { return value.postprocess_ms; })
         << ",\n  \"peak_rss_kb\": " << usage.ru_maxrss << "\n}\n";
}

// 执行完整板端评测流程.
void Run(const Options& options) {
  fs::create_directories(options.output_dir);
  const auto detections = LoadManifest(options.manifest, options.limit);
  const fs::path predictions_path = options.output_dir / "predictions.jsonl";
  const fs::path completed_path = options.output_dir / "processed_ids.txt";
  auto completed = LoadCompleted(completed_path);

  std::cout << "[1/3] 加载 DLA: " << options.model << std::endl;
  NeuronModel model(options.model);
  cv::Mat current_image;
  int64_t current_image_id = -1;
  const auto load_image = [&](int64_t image_id) {
    return cv::imread(ImagePath(options.images, image_id).string(),
                      cv::IMREAD_COLOR);
  };
  current_image = load_image(detections.front().image_id);
  if (current_image.empty()) {
    throw std::runtime_error("无法读取首张 COCO 图片.");
  }
  auto warmup_input = Preprocess(current_image, detections.front()).first;
  std::cout << "[2/3] NPU 预热 " << options.warmup << " 次." << std::endl;
  for (int index = 0; index < options.warmup; ++index) {
    model.Infer(warmup_input);
  }

  std::ofstream predictions(predictions_path, std::ios::app);
  std::ofstream completed_output(completed_path, std::ios::app);
  std::vector<Timing> timings;
  size_t processed_count = completed.size();
  std::cout << "[3/3] 逐人体框评测,共 " << detections.size()
            << " 个,已完成 " << completed.size() << " 个." << std::endl;
  for (const auto& detection : detections) {
    if (completed.contains(detection.detection_id)) {
      continue;
    }
    if (current_image_id != detection.image_id) {
      current_image = load_image(detection.image_id);
      current_image_id = detection.image_id;
      if (current_image.empty()) {
        throw std::runtime_error("无法读取 COCO 图片,image_id=" +
                                 std::to_string(detection.image_id));
      }
    }
    const auto preprocess_start = std::chrono::steady_clock::now();
    auto [input, geometry] = Preprocess(current_image, detection);
    const auto preprocess_end = std::chrono::steady_clock::now();
    const double npu_ms = model.Infer(input);
    if (timings.empty()) {
      DumpBuffer(options.output_dir / "first_input.bin", input.data(),
                 input.size());
      DumpBuffer(options.output_dir / "first_output_0.bin",
                 model.output(0).data(), model.output(0).size());
      DumpBuffer(options.output_dir / "first_output_1.bin",
                 model.output(1).data(), model.output(1).size());
    }
    const auto postprocess_start = std::chrono::steady_clock::now();
    AppendPrediction(&predictions, detection, geometry, model);
    const auto postprocess_end = std::chrono::steady_clock::now();
    timings.push_back({
        std::chrono::duration<double, std::milli>(preprocess_end -
                                                  preprocess_start)
            .count(),
        npu_ms,
        std::chrono::duration<double, std::milli>(postprocess_end -
                                                  postprocess_start)
            .count(),
    });
    completed_output << detection.detection_id << '\n';
    completed.insert(detection.detection_id);
    ++processed_count;
    if (processed_count % options.progress_interval == 0 ||
        processed_count == detections.size()) {
      predictions.flush();
      completed_output.flush();
      std::cout << "[board] " << processed_count << '/' << detections.size()
                << ",image_id=" << detection.image_id << std::endl;
    }
  }
  if (!timings.empty()) {
    WriteTimingSummary(options.output_dir / "timing_summary.json", timings);
  }
  std::cout << "[OK] 板端预测完成: " << predictions_path << std::endl;
}

}  // namespace

// 解析参数并执行板端评测.
int main(int argc, char** argv) {
  try {
    Run(ParseOptions(argc, argv));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << std::endl;
    return 1;
  }
}
