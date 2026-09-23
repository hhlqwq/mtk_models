// 在 Genio 720 板端使用 C++ 完成图片预处理、NPU 推理与实例掩码还原.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "neuron/api/RuntimeAPI.h"

namespace fs = std::filesystem;

namespace {

constexpr int kSize = 640;
constexpr int kCoefficientCount = 32;
constexpr std::array<int, 3> kStrides = {8, 16, 32};
constexpr std::array<const char*, 10> kNames = {
    "box_8", "score_8", "coefficient_8", "box_16", "score_16",
    "coefficient_16", "box_32", "score_32", "coefficient_32", "prototype"};

struct TensorInfo {
  std::string name;
  int index = 0;
  int channels = 0;
  int height = 0;
  int width = 0;
  float scale = 0.0F;
  int zero_point = 0;
};

struct Options {
  fs::path model;
  fs::path config;
  fs::path image;
  fs::path output_dir;
  float confidence = 0.4F;
  float iou = 0.9F;
  int max_detections = 100;
  std::array<int, 2> point = {-1, -1};
  std::array<int, 4> box = {-1, -1, -1, -1};
  bool check_image = false;
};

struct Geometry {
  int width = 0;
  int height = 0;
  int left = 0;
  int top = 0;
  float ratio = 0.0F;
};

struct Detection {
  cv::Rect2f box;
  float score = 0.0F;
  std::array<float, kCoefficientCount> coefficient{};
  cv::Mat mask;
};

// 检查 Runtime 返回值并保留具体操作名称.
void CheckRuntime(int result, const std::string& operation) {
  if (result != NEURONRUNTIME_NO_ERROR) {
    throw std::runtime_error(operation + " 失败,错误码 " + std::to_string(result));
  }
}

// 解析单图冒烟参数; --check-image 可在没有权重时验证板端 C++ 图片路径.
Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string key = argv[index];
    if (key == "--check-image") {
      options.check_image = true;
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("缺少参数值: " + key);
    }
    const std::string value = argv[++index];
    if (key == "--model") options.model = value;
    else if (key == "--config") options.config = value;
    else if (key == "--image") options.image = value;
    else if (key == "--output-dir") options.output_dir = value;
    else if (key == "--confidence") options.confidence = std::stof(value);
    else if (key == "--iou") options.iou = std::stof(value);
    else if (key == "--max-det") options.max_detections = std::stoi(value);
    else if (key == "--point") {
      options.point[0] = std::stoi(value);
      if (index + 1 >= argc) throw std::invalid_argument("--point 需要两个坐标.");
      options.point[1] = std::stoi(argv[++index]);
    } else if (key == "--box") {
      options.box[0] = std::stoi(value);
      if (index + 3 >= argc) throw std::invalid_argument("--box 需要四个坐标.");
      for (int part = 1; part < 4; ++part) options.box[part] = std::stoi(argv[++index]);
    } else {
      throw std::invalid_argument("未知参数: " + key);
    }
  }
  if (options.image.empty() || options.output_dir.empty() ||
      (!options.check_image && (options.model.empty() || options.config.empty()))) {
    throw std::invalid_argument("需要 --image --output-dir;真实推理还需要 --model --config.");
  }
  if (options.confidence < 0 || options.confidence > 1 || options.iou < 0 ||
      options.iou > 1 || options.max_detections <= 0 || options.max_detections > 1000) {
    throw std::invalid_argument("置信度、IoU 或最大实例数无效.");
  }
  if (options.point[0] >= 0 && options.box[0] >= 0) {
    throw std::invalid_argument("点提示和框提示互斥.");
  }
  return options;
}

// 严格读取由本模型的量化脚本生成的固定七列张量契约.
std::array<TensorInfo, 11> LoadConfig(const fs::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("无法读取张量配置: " + path.string());
  std::array<TensorInfo, 11> result;
  std::array<bool, 11> seen{};
  std::string line;
  int rows = 0;
  while (std::getline(input, line)) {
    std::istringstream stream(line);
    std::vector<std::string> columns;
    std::string value;
    while (std::getline(stream, value, ',')) columns.push_back(value);
    if (columns.size() != 7) throw std::runtime_error("张量配置字段数错误.");
    TensorInfo tensor{columns[0], std::stoi(columns[1]), std::stoi(columns[2]),
                      std::stoi(columns[3]), std::stoi(columns[4]),
                      std::stof(columns[5]), std::stoi(columns[6])};
    int slot = -1;
    if (tensor.name == "input") slot = 0;
    for (int i = 0; i < 10; ++i) {
      if (tensor.name == kNames[i]) slot = i + 1;
    }
    if (slot < 0 || seen[slot] || tensor.scale <= 0 || tensor.zero_point < -128 ||
        tensor.zero_point > 127 || tensor.index < 0) {
      throw std::runtime_error("张量配置含未知、重复或无效项目: " + tensor.name);
    }
    seen[slot] = true;
    result[slot] = std::move(tensor);
    ++rows;
  }
  if (rows != 11 || !std::all_of(seen.begin(), seen.end(), [](bool value) {return value;})) {
    throw std::runtime_error("张量配置缺少输入或输出.");
  }
  if (result[0].channels != 3 || result[0].height != kSize || result[0].width != kSize) {
    throw std::runtime_error("输入张量必须为 [1,3,640,640].");
  }
  for (int level = 0; level < 3; ++level) {
    for (int kind = 0; kind < 3; ++kind) {
      const auto& tensor = result[1 + level * 3 + kind];
      if (tensor.channels != (kind == 0 ? 64 : kind == 1 ? 1 : 32) ||
          tensor.height != kSize / kStrides[level] ||
          tensor.width != kSize / kStrides[level] || tensor.index > 9) {
        throw std::runtime_error("检测头张量形状或索引错误: " + tensor.name);
      }
    }
  }
  if (result[10].channels != 32 || result[10].height != 160 ||
      result[10].width != 160 || result[10].index > 9) {
    throw std::runtime_error("掩码原型形状或索引错误.");
  }
  std::array<bool, 10> indices{};
  for (int slot = 1; slot <= 10; ++slot) {
    if (indices[result[slot].index]) throw std::runtime_error("输出索引重复.");
    indices[result[slot].index] = true;
  }
  return result;
}

// 用与导出脚本一致的居中 letterbox 和 RGB NCHW 量化准备单图.
std::pair<std::vector<int8_t>, Geometry> Preprocess(const cv::Mat& image,
                                                     const TensorInfo& input) {
  if (image.empty()) throw std::runtime_error("图片为空.");
  Geometry geo{image.cols, image.rows};
  geo.ratio = std::min(static_cast<float>(kSize) / image.cols,
                       static_cast<float>(kSize) / image.rows);
  const int width = std::lround(image.cols * geo.ratio);
  const int height = std::lround(image.rows * geo.ratio);
  geo.left = (kSize - width) / 2;
  geo.top = (kSize - height) / 2;
  cv::Mat canvas(kSize, kSize, CV_8UC3, cv::Scalar(114, 114, 114));
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(width, height));
  resized.copyTo(canvas(cv::Rect(geo.left, geo.top, width, height)));
  std::vector<int8_t> values(3 * kSize * kSize);
  for (int row = 0; row < kSize; ++row) {
    const auto* pixels = canvas.ptr<cv::Vec3b>(row);
    for (int column = 0; column < kSize; ++column) {
      for (int channel = 0; channel < 3; ++channel) {
        const float normalized = pixels[column][2 - channel] / 255.0F;
        const int integer = static_cast<int>(std::nearbyint(normalized / input.scale)) +
                            input.zero_point;
        values[(channel * kSize + row) * kSize + column] =
            static_cast<int8_t>(std::clamp(integer, -128, 127));
      }
    }
  }
  return {std::move(values), geo};
}

class NeuronModel {
 public:
  // 创建 MDLA Runtime,加载 DLA 并检查每个输出的原生缓冲区长度.
  NeuronModel(const fs::path& path, const std::array<TensorInfo, 11>& config)
      : config_(config) {
    EnvOptions options{};
    options.deviceKind = kEnvOptHardware;
    options.MDLACoreOption = Auto;
    options.CPUThreadNum = 1;
    options.suppressInputConversion = true;
    options.suppressOutputConversion = true;
    CheckRuntime(NeuronRuntime_create(&options, &runtime_), "NeuronRuntime_create");
    try {
      CheckRuntime(NeuronRuntime_loadNetworkFromFile(runtime_, path.c_str()),
                   "NeuronRuntime_loadNetworkFromFile");
      size_t input_count = 0, output_count = 0, input_size = 0;
      CheckRuntime(NeuronRuntime_getInputNumber(runtime_, &input_count), "getInputNumber");
      CheckRuntime(NeuronRuntime_getOutputNumber(runtime_, &output_count), "getOutputNumber");
      CheckRuntime(NeuronRuntime_getSingleInputPaddedSize(runtime_, &input_size),
                   "getSingleInputPaddedSize");
      if (input_count != 1 || output_count != 10 || input_size != 3 * kSize * kSize) {
        throw std::runtime_error("DLA 输入输出数量或输入字节数不匹配.");
      }
      input_.resize(input_size);
      outputs_.resize(output_count);
      for (int slot = 1; slot <= 10; ++slot) {
        const auto& tensor = config_[slot];
        size_t size = 0;
        CheckRuntime(NeuronRuntime_getOutputPaddedSize(runtime_, tensor.index, &size),
                     "getOutputPaddedSize");
        const size_t width = static_cast<size_t>(tensor.width);
        const size_t plain = tensor.channels * tensor.height * width;
        const size_t padded = tensor.channels * tensor.height * ((width + 15) / 16 * 16);
        if (size != plain && size != padded) {
          throw std::runtime_error("DLA 输出字节数不匹配: " + tensor.name +
                                   ",实际 " + std::to_string(size));
        }
        outputs_[tensor.index].resize(size);
      }
    } catch (...) {
      NeuronRuntime_release(runtime_);
      runtime_ = nullptr;
      throw;
    }
  }

  NeuronModel(const NeuronModel&) = delete;
  NeuronModel& operator=(const NeuronModel&) = delete;

  // 确保异常路径也释放 Runtime.
  ~NeuronModel() {
    if (runtime_ != nullptr) NeuronRuntime_release(runtime_);
  }

  // 登记全部 IO 并测量一次真实硬件推理耗时.
  double Infer(const std::vector<int8_t>& input) {
    if (input.size() != input_.size()) throw std::runtime_error("输入字节数不匹配.");
    std::copy(input.begin(), input.end(), input_.begin());
    const BufferAttribute attribute{NON_ION_FD};
    CheckRuntime(NeuronRuntime_setSingleInput(runtime_, input_.data(), input_.size(), attribute),
                 "setSingleInput");
    for (size_t index = 0; index < outputs_.size(); ++index) {
      CheckRuntime(NeuronRuntime_setOutput(runtime_, index, outputs_[index].data(),
                                          outputs_[index].size(), attribute), "setOutput");
    }
    const auto start = std::chrono::steady_clock::now();
    CheckRuntime(NeuronRuntime_inference(runtime_), "NeuronRuntime_inference");
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }

  // 按语义编号访问输出原生字节,不假设编译器保留输出顺序.
  const std::vector<int8_t>& Output(int slot) const {
    return outputs_.at(config_.at(slot).index);
  }

 private:
  const std::array<TensorInfo, 11>& config_;
  void* runtime_ = nullptr;
  std::vector<int8_t> input_;
  std::vector<std::vector<int8_t>> outputs_;
};

// 按 NCHW 和 16 元素行宽对齐读取、反量化一个原始输出元素.
float ReadValue(const NeuronModel& model, const TensorInfo& info, int slot,
                int channel, int row, int column) {
  const auto& bytes = model.Output(slot);
  const size_t plane = static_cast<size_t>(info.channels) * info.height * info.width;
  const int padded_width = bytes.size() == plane ? info.width :
                           (info.width + 15) / 16 * 16;
  const size_t offset = (static_cast<size_t>(channel) * info.height + row) *
                        padded_width + column;
  return (static_cast<int>(bytes.at(offset)) - info.zero_point) * info.scale;
}

// 计算 Sigmoid,避免极端量化 logits 导致浮点溢出.
float Sigmoid(float value) {
  const float clipped = std::clamp(value, -30.0F, 30.0F);
  return 1.0F / (1.0F + std::exp(-clipped));
}

// 计算 xyxy 框 IoU.
float BoxIou(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
  const float x2 = std::min(a.x + a.width, b.x + b.width);
  const float y2 = std::min(a.y + a.height, b.y + b.height);
  const float intersection = std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
  return intersection / std::max(a.area() + b.area() - intersection, 1.0e-9F);
}

// 在 CPU 上执行 DFL、类别无关 NMS 与原图坐标反变换.
std::vector<Detection> Decode(const NeuronModel& model,
                              const std::array<TensorInfo, 11>& config,
                              const Geometry& geo, const Options& options) {
  std::vector<Detection> candidates;
  for (int level = 0; level < 3; ++level) {
    const int side = kSize / kStrides[level];
    const int box_slot = 1 + level * 3;
    for (int row = 0; row < side; ++row) {
      for (int column = 0; column < side; ++column) {
        const float score = Sigmoid(ReadValue(model, config[box_slot + 1],
                                             box_slot + 1, 0, row, column));
        if (score <= options.confidence) continue;
        std::array<float, 4> distance{};
        for (int axis = 0; axis < 4; ++axis) {
          float logits[16], maximum = -std::numeric_limits<float>::infinity();
          for (int bin = 0; bin < 16; ++bin) {
            logits[bin] = ReadValue(model, config[box_slot], box_slot,
                                    axis * 16 + bin, row, column);
            maximum = std::max(maximum, logits[bin]);
          }
          float sum = 0.0F, weighted = 0.0F;
          for (int bin = 0; bin < 16; ++bin) {
            const float weight = std::exp(logits[bin] - maximum);
            sum += weight;
            weighted += weight * bin;
          }
          distance[axis] = weighted / sum;
        }
        const float anchor_x = column + 0.5F, anchor_y = row + 0.5F;
        const float stride = static_cast<float>(kStrides[level]);
        const float x1 = (anchor_x - distance[0]) * stride;
        const float y1 = (anchor_y - distance[1]) * stride;
        const float x2 = (anchor_x + distance[2]) * stride;
        const float y2 = (anchor_y + distance[3]) * stride;
        Detection detection{cv::Rect2f(x1, y1, x2 - x1, y2 - y1), score};
        for (int channel = 0; channel < kCoefficientCount; ++channel) {
          detection.coefficient[channel] = ReadValue(model, config[box_slot + 2],
                                                     box_slot + 2, channel, row, column);
        }
        candidates.push_back(std::move(detection));
      }
    }
  }
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const auto& left, const auto& right) {return left.score > right.score;});
  if (candidates.size() > 30000) candidates.resize(30000);
  std::vector<Detection> selected;
  for (const auto& item : candidates) {
    bool suppressed = false;
    for (const auto& kept : selected) {
      if (BoxIou(item.box, kept.box) > options.iou) {
        suppressed = true;
        break;
      }
    }
    if (suppressed) continue;
    selected.push_back(item);
    if (static_cast<int>(selected.size()) >= options.max_detections) break;
  }
  const cv::Rect2f full(0, 0, kSize, kSize);
  for (auto& item : selected) {
    float x1 = item.box.x, y1 = item.box.y;
    float x2 = x1 + item.box.width, y2 = y1 + item.box.height;
    if (x1 < 20) x1 = 0;
    if (y1 < 20) y1 = 0;
    if (x2 > kSize - 20) x2 = kSize;
    if (y2 > kSize - 20) y2 = kSize;
    cv::Rect2f adjusted(x1, y1, x2 - x1, y2 - y1);
    if (BoxIou(adjusted, full) > 0.9F) adjusted = full;
    x1 = std::clamp((adjusted.x - geo.left) / geo.ratio, 0.0F,
                    static_cast<float>(geo.width));
    y1 = std::clamp((adjusted.y - geo.top) / geo.ratio, 0.0F,
                    static_cast<float>(geo.height));
    x2 = std::clamp((adjusted.x + adjusted.width - geo.left) / geo.ratio, 0.0F,
                    static_cast<float>(geo.width));
    y2 = std::clamp((adjusted.y + adjusted.height - geo.top) / geo.ratio, 0.0F,
                    static_cast<float>(geo.height));
    item.box = cv::Rect2f(x1, y1, std::max(0.0F, x2 - x1), std::max(0.0F, y2 - y1));
  }
  return selected;
}

// 逐实例解码 160x160 原型,执行裁边、插值及框裁剪.
void DecodeMasks(std::vector<Detection>* detections, const NeuronModel& model,
                 const std::array<TensorInfo, 11>& config, const Geometry& geo) {
  const float gain = std::min(160.0F / geo.height, 160.0F / geo.width);
  const int left = static_cast<int>((160 - geo.width * gain) / 2);
  const int top = static_cast<int>((160 - geo.height * gain) / 2);
  const int right = static_cast<int>(160 - (160 - geo.width * gain) / 2);
  const int bottom = static_cast<int>(160 - (160 - geo.height * gain) / 2);
  const cv::Rect crop(left, top, right - left, bottom - top);
  if (crop.width <= 0 || crop.height <= 0) throw std::runtime_error("原型裁边尺寸无效.");
  std::array<cv::Mat, kCoefficientCount> prototypes;
  for (int channel = 0; channel < kCoefficientCount; ++channel) {
    prototypes[channel] = cv::Mat(160, 160, CV_32F);
    for (int row = 0; row < 160; ++row) {
      float* dst = prototypes[channel].ptr<float>(row);
      for (int column = 0; column < 160; ++column) {
        dst[column] = ReadValue(model, config[10], 10, channel, row, column);
      }
    }
  }
  for (size_t index = 0; index < detections->size(); ++index) {
    auto& item = detections->at(index);
    cv::Mat logits = cv::Mat::zeros(160, 160, CV_32F);
    for (int channel = 0; channel < kCoefficientCount; ++channel) {
      logits += prototypes[channel] * item.coefficient[channel];
    }
    for (int row = 0; row < logits.rows; ++row) {
      float* values = logits.ptr<float>(row);
      for (int column = 0; column < logits.cols; ++column) {
        values[column] = Sigmoid(values[column]);
      }
    }
    cv::Mat scaled;
    cv::resize(logits(crop), scaled, cv::Size(geo.width, geo.height));
    item.mask = cv::Mat::zeros(geo.height, geo.width, CV_8U);
    const int x1 = std::clamp(static_cast<int>(std::ceil(item.box.x)), 0, geo.width);
    const int y1 = std::clamp(static_cast<int>(std::ceil(item.box.y)), 0, geo.height);
    const int x2 = std::clamp(static_cast<int>(std::ceil(item.box.x + item.box.width)),
                              0, geo.width);
    const int y2 = std::clamp(static_cast<int>(std::ceil(item.box.y + item.box.height)),
                              0, geo.height);
    for (int row = y1; row < y2; ++row) {
      const float* src = scaled.ptr<float>(row);
      auto* dst = item.mask.ptr<uint8_t>(row);
      for (int column = x1; column < x2; ++column) {
        dst[column] = src[column] > 0.5F ? 255 : 0;
      }
    }
    if ((index + 1) % 25 == 0) {
      std::cout << "[MASK] " << index + 1 << '/' << detections->size() << std::endl;
    }
  }
}

// 依用户给出的原图点或 xyxy 框,选择目标实例.
std::vector<size_t> SelectMasks(const std::vector<Detection>& items,
                                const Options& options, const Geometry& geo) {
  std::vector<size_t> selected;
  if (options.point[0] >= 0) {
    const auto [x, y] = options.point;
    if (x >= geo.width || y >= geo.height) throw std::invalid_argument("点提示越界.");
    for (size_t index = 0; index < items.size(); ++index) {
      if (items[index].mask.at<uint8_t>(y, x)) selected.push_back(index);
    }
  } else if (options.box[0] >= 0) {
    const auto [x1, y1, x2, y2] = options.box;
    if (x2 <= x1 || y2 <= y1 || x2 > geo.width || y2 > geo.height) {
      throw std::invalid_argument("提示框无效或越界.");
    }
    const cv::Rect region(x1, y1, x2 - x1, y2 - y1);
    float best_score = 0.0F;
    size_t best_index = 0;
    for (size_t index = 0; index < items.size(); ++index) {
      const float mask_area = static_cast<float>(cv::countNonZero(items[index].mask));
      const float intersection = static_cast<float>(cv::countNonZero(items[index].mask(region)));
      const float overlap = intersection / std::max(region.area() + mask_area - intersection, 1.0F);
      if (overlap > best_score) {
        best_score = overlap;
        best_index = index;
      }
    }
    if (best_score > 0) selected.push_back(best_index);
  } else {
    for (size_t index = 0; index < items.size(); ++index) selected.push_back(index);
  }
  return selected;
}

// 保存板端输入、每个实例 PNG、叠加图及耗时 JSON.
void SaveResults(const Options& options, const cv::Mat& image,
                 const std::vector<int8_t>& input,
                 const std::vector<Detection>& items,
                 const std::vector<size_t>& selected,
                 double preprocess_ms, double model_load_ms, double npu_ms,
                 double postprocess_ms,
                 double end_to_end_ms) {
  fs::create_directories(options.output_dir / "masks");
  const fs::path input_path = options.output_dir / "input_int8.bin";
  std::ofstream(input_path, std::ios::binary).write(
      reinterpret_cast<const char*>(input.data()), input.size());
  cv::Mat overlay = image.clone();
  std::ofstream result(options.output_dir / "results.json");
  if (!result) throw std::runtime_error("无法写入结果 JSON.");
  result << std::fixed << std::setprecision(6)
         << "{\"backend\":\"cpp_neuron_runtime_hw\",\"instance_count_before_prompt\":"
         << items.size() << ",\"selected_count\":" << selected.size()
         << ",\"preprocess_ms\":" << preprocess_ms
         << ",\"model_load_ms\":" << model_load_ms << ",\"npu_ms\":" << npu_ms
         << ",\"postprocess_ms\":" << postprocess_ms << ",\"end_to_end_ms\":"
         << end_to_end_ms << ",\"peak_rss_kib\":";
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("读取 RSS 失败.");
  result << usage.ru_maxrss << ",\"detections\":[";
  for (size_t position = 0; position < selected.size(); ++position) {
    const size_t index = selected[position];
    const auto& item = items[index];
    const fs::path name = fs::path("masks") / (std::to_string(index) + ".png");
    if (!cv::imwrite((options.output_dir / name).string(), item.mask)) {
      throw std::runtime_error("无法保存实例掩码.");
    }
    const cv::Scalar color((index * 67 + 43) % 256, (index * 97 + 101) % 256,
                           (index * 137 + 179) % 256);
    cv::Mat painted(image.size(), CV_8UC3, color);
    cv::Mat mixed;
    cv::addWeighted(overlay, 0.5, painted, 0.5, 0, mixed);
    mixed.copyTo(overlay, item.mask);
    if (position) result << ',';
    result << "{\"index\":" << index << ",\"score\":" << item.score
           << ",\"box_xyxy\":[" << item.box.x << ',' << item.box.y << ','
           << item.box.x + item.box.width << ',' << item.box.y + item.box.height
           << "],\"mask\":\"" << name.generic_string() << "\"}";
  }
  result << "]}\n";
  if (!cv::imwrite((options.output_dir / "overlay.jpg").string(), overlay)) {
    throw std::runtime_error("无法保存叠加图.");
  }
}

// 执行一张图片的完整 C++ 路径,可单独检查无权重时的图片预处理.
void Run(const Options& options) {
  if (fs::exists(options.output_dir)) throw std::runtime_error("输出目录已存在,拒绝覆盖.");
  fs::create_directories(options.output_dir);
  const auto start = std::chrono::steady_clock::now();
  const cv::Mat image = cv::imread(options.image.string(), cv::IMREAD_COLOR);
  if (image.empty()) throw std::runtime_error("无法读取图片: " + options.image.string());
  if (options.check_image) {
    TensorInfo input{"input", 0, 3, 640, 640, 1.0F / 255, -128};
    const auto [bytes, geo] = Preprocess(image, input);
    std::ofstream(options.output_dir / "input_int8.bin", std::ios::binary).write(
        reinterpret_cast<const char*>(bytes.data()), bytes.size());
    std::cout << "[OK] C++ 图片预处理: " << geo.width << 'x' << geo.height
              << ",输入字节=" << bytes.size() << std::endl;
    return;
  }
  const auto config = LoadConfig(options.config);
  const auto [bytes, geo] = Preprocess(image, config[0]);
  const auto preprocess_end = std::chrono::steady_clock::now();
  std::cout << "[1/3] C++ 预处理完成,加载硬件 DLA." << std::endl;
  NeuronModel model(options.model, config);
  const auto load_end = std::chrono::steady_clock::now();
  const double npu_ms = model.Infer(bytes);
  const auto inference_end = std::chrono::steady_clock::now();
  std::cout << "[2/3] 硬件推理完成,解码实例." << std::endl;
  auto detections = Decode(model, config, geo, options);
  DecodeMasks(&detections, model, config, geo);
  const auto selected = SelectMasks(detections, options, geo);
  const auto end = std::chrono::steady_clock::now();
  const double preprocess_ms = std::chrono::duration<double, std::milli>(
      preprocess_end - start).count();
  const double model_load_ms = std::chrono::duration<double, std::milli>(
      load_end - preprocess_end).count();
  const double postprocess_ms = std::chrono::duration<double, std::milli>(
      end - inference_end).count();
  const double end_to_end_ms = std::chrono::duration<double, std::milli>(end - start).count();
  SaveResults(options, image, bytes, detections, selected, preprocess_ms, model_load_ms, npu_ms,
              postprocess_ms, end_to_end_ms);
  for (int slot = 1; slot <= 10; ++slot) {
    const auto& output = model.Output(slot);
    const fs::path path = options.output_dir / (std::string(kNames[slot - 1]) + ".bin");
    std::ofstream(path, std::ios::binary).write(
        reinterpret_cast<const char*>(output.data()), output.size());
  }
  std::cout << "[3/3] 已保存 " << selected.size() << " 个实例,纯推理调用 "
            << npu_ms << " ms." << std::endl;
}

}  // namespace

// 统一返回非零退出码,让部署脚本正确判断冒烟失败.
int main(int argc, char** argv) {
  try {
    Run(ParseOptions(argc, argv));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << std::endl;
    return 1;
  }
}
