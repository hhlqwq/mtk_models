// 在 Genio 720 上用 ONNX Runtime C API 和 C++ 完成 YOLO-World 全量推理.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "onnxruntime_c_api.h"

namespace fs = std::filesystem;

struct Options {
  fs::path model;
  fs::path images;
  fs::path output_dir;
};

struct Transform {
  float scale;
  int pad_x;
  int pad_y;
  int width;
  int height;
};

struct Detection {
  int class_id;
  float score;
  std::array<float, 4> box;
};

// 解析全量推理所需的三个路径.
Options ParseArgs(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) throw std::invalid_argument("参数缺少值.");
    const std::string key = argv[index];
    if (key == "--model") options.model = argv[index + 1];
    else if (key == "--images") options.images = argv[index + 1];
    else if (key == "--output-dir") options.output_dir = argv[index + 1];
    else throw std::invalid_argument("未知参数: " + key);
  }
  if (options.model.empty() || options.images.empty() ||
      options.output_dir.empty()) {
    throw std::invalid_argument("需要 --model、--images 和 --output-dir.");
  }
  return options;
}

// 以官方图片编号排序,不允许仅对 COCO 子集测试.
std::vector<fs::path> ListImages(const fs::path& directory) {
  std::vector<fs::path> images;
  for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
    if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());
  if (images.size() != 5000) {
    throw std::runtime_error("COCO val2017 图片数量不是 5000.");
  }
  for (const fs::path& image : images) {
    const std::string name = image.stem().string();
    if (name.size() != 12 ||
        !std::all_of(name.begin(), name.end(),
                     [](char value) { return value >= '0' && value <= '9'; })) {
      throw std::runtime_error("COCO 图片名称不是 12 位编号: " + name);
    }
  }
  return images;
}

// 复现项目黑边正方形填充、640 缩放和 NCHW RGB 输入.
std::pair<std::vector<float>, Transform> Preprocess(const cv::Mat& image) {
  const int square_size = std::max(image.rows, image.cols);
  const int pad_y = (square_size - image.rows) / 2;
  const int pad_x = (square_size - image.cols) / 2;
  cv::Mat square(square_size, square_size, CV_8UC3, cv::Scalar(0, 0, 0));
  image.copyTo(square(cv::Rect(pad_x, pad_y, image.cols, image.rows)));
  cv::Mat resized;
  cv::resize(square, resized, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);
  std::vector<float> input(3 * 640 * 640);
  for (int row = 0; row < 640; ++row) {
    const auto* pixels = resized.ptr<cv::Vec3b>(row);
    for (int column = 0; column < 640; ++column) {
      for (int channel = 0; channel < 3; ++channel) {
        input[channel * 640 * 640 + row * 640 + column] =
            pixels[column][2 - channel] / 255.0F;
      }
    }
  }
  return {std::move(input), {640.0F / square_size, pad_x, pad_y,
                             image.cols, image.rows}};
}

// 将 ORT 状态码转换为带上下文的异常并释放状态对象.
void CheckOrt(const OrtApi* api, OrtStatus* status, const std::string& step) {
  if (status == nullptr) return;
  const std::string message = api->GetErrorMessage(status);
  api->ReleaseStatus(status);
  throw std::runtime_error("ONNX Runtime " + step + ": " + message);
}

// 动态加载板端 ORT,避免编译主机必须提供同架构链接库.
class BoardSession {
 public:
  BoardSession(const fs::path& model, const fs::path& profile_prefix) {
    library_ = dlopen("libonnxruntime.so.1.20.2", RTLD_NOW | RTLD_LOCAL);
    if (library_ == nullptr) {
      throw std::runtime_error("无法加载板端 libonnxruntime.so.1.20.2.");
    }
    try {
      using GetApiBase = const OrtApiBase* (*)();
      auto get_api_base = reinterpret_cast<GetApiBase>(
          dlsym(library_, "OrtGetApiBase"));
      if (get_api_base == nullptr) throw std::runtime_error("ORT API 入口缺失.");
      api_ = get_api_base()->GetApi(ORT_API_VERSION);
      if (api_ == nullptr) throw std::runtime_error("ORT API 版本不匹配.");
      CheckOrt(api_, api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING,
                                     "yoloworld_full", &env_), "CreateEnv");
      CheckOrt(api_, api_->CreateSessionOptions(&options_),
               "CreateSessionOptions");
      CheckOrt(api_, api_->SetSessionGraphOptimizationLevel(options_,
                                                             ORT_ENABLE_ALL),
               "SetSessionGraphOptimizationLevel");
      CheckOrt(api_, api_->EnableProfiling(options_, profile_prefix.c_str()),
               "EnableProfiling");
      const std::array<const char*, 3> keys = {
          "NEURON_FLAG_USE_FP16", "NEURON_FLAG_MIN_GROUP_SIZE",
          "NEURON_FLAG_OPTIMIZATION_STRING"};
      const std::array<const char*, 3> values = {
          "1", "100", "--opt=3 --num-mdla=1 --reshape-to-4d "
                      "--interval-coloring-converage=1.0"};
      CheckOrt(api_, api_->SessionOptionsAppendExecutionProvider(
                   options_, "NeuronExecutionProvider", keys.data(),
                   values.data(), keys.size()), "Append Neuron EP");
      CheckOrt(api_, api_->CreateSession(env_, model.c_str(), options_,
                                         &session_), "CreateSession");
      size_t inputs = 0, outputs = 0;
      CheckOrt(api_, api_->SessionGetInputCount(session_, &inputs),
               "SessionGetInputCount");
      CheckOrt(api_, api_->SessionGetOutputCount(session_, &outputs),
               "SessionGetOutputCount");
      if (inputs != 1 || outputs != 6) {
        throw std::runtime_error("YOLO-World 模型输入输出数量不匹配.");
      }
      OrtAllocator* allocator = nullptr;
      CheckOrt(api_, api_->GetAllocatorWithDefaultOptions(&allocator),
               "GetAllocatorWithDefaultOptions");
      for (size_t index = 0; index < 6; ++index) {
        char* name = nullptr;
        CheckOrt(api_, api_->SessionGetOutputName(session_, index, allocator,
                                                  &name), "SessionGetOutputName");
        output_names_.emplace_back(name);
        CheckOrt(api_, api_->AllocatorFree(allocator, name), "AllocatorFree");
      }
      CheckOrt(api_, api_->CreateCpuMemoryInfo(OrtArenaAllocator,
                                               OrtMemTypeDefault, &memory_),
               "CreateCpuMemoryInfo");
    } catch (...) {
      Release();
      throw;
    }
  }

  BoardSession(const BoardSession&) = delete;
  BoardSession& operator=(const BoardSession&) = delete;

  // 释放会话及所有 ORT 对象.
  ~BoardSession() { Release(); }

  // 用真实 Neuron EP 会话执行一次推理并校验六个输出张量.
  std::pair<std::array<std::vector<float>, 6>, double> Infer(
      std::vector<float>& input) {
    OrtValue* input_value = nullptr;
    const std::array<int64_t, 4> shape = {1, 3, 640, 640};
    CheckOrt(api_, api_->CreateTensorWithDataAsOrtValue(
                 memory_, input.data(), input.size() * sizeof(float),
                 shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                 &input_value), "CreateTensor");
    std::array<OrtValue*, 6> output_values{};
    std::array<const char*, 6> names{};
    for (size_t index = 0; index < names.size(); ++index) {
      names[index] = output_names_[index].c_str();
    }
    const char* input_name = "images";
    const OrtValue* input_values[] = {input_value};
    const auto start = std::chrono::steady_clock::now();
    OrtStatus* status = api_->Run(session_, nullptr, &input_name, input_values,
                                  1, names.data(), names.size(), output_values.data());
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    api_->ReleaseValue(input_value);
    try {
      CheckOrt(api_, status, "Run");
      std::array<std::vector<float>, 6> outputs;
      constexpr std::array<int, 3> widths = {80, 40, 20};
      for (size_t index = 0; index < outputs.size(); ++index) {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        CheckOrt(api_, api_->GetTensorTypeAndShape(output_values[index], &info),
                 "GetTensorTypeAndShape");
        size_t count = 0;
        ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        OrtStatus* type_status = api_->GetTensorElementType(info, &type);
        OrtStatus* count_status = api_->GetTensorShapeElementCount(
            info, &count);
        api_->ReleaseTensorTypeAndShapeInfo(info);
        CheckOrt(api_, type_status, "GetTensorElementType");
        CheckOrt(api_, count_status, "GetTensorShapeElementCount");
        const size_t width = widths[index / 2];
        const size_t channels = index % 2 == 0 ? 80 : 4;
        if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            count != channels * width * width) {
          throw std::runtime_error("ORT 输出元素数量不匹配: " +
                                   std::to_string(index));
        }
        void* data = nullptr;
        CheckOrt(api_, api_->GetTensorMutableData(output_values[index], &data),
                 "GetTensorMutableData");
        const auto* values = static_cast<const float*>(data);
        outputs[index].assign(values, values + count);
      }
      for (OrtValue* value : output_values) api_->ReleaseValue(value);
      return {std::move(outputs), elapsed_ms};
    } catch (...) {
      for (OrtValue* value : output_values) {
        if (value != nullptr) api_->ReleaseValue(value);
      }
      throw;
    }
  }

  // 结束 profiling 并返回原始事件文件路径供指标阶段核验.
  fs::path FinishProfiling() {
    OrtAllocator* allocator = nullptr;
    CheckOrt(api_, api_->GetAllocatorWithDefaultOptions(&allocator),
             "GetAllocatorWithDefaultOptions");
    char* output = nullptr;
    CheckOrt(api_, api_->SessionEndProfiling(session_, allocator, &output),
             "SessionEndProfiling");
    const fs::path path(output);
    CheckOrt(api_, api_->AllocatorFree(allocator, output), "AllocatorFree");
    return path;
  }

 private:
  // 在构造异常及正常退出时回收部分或全部资源.
  void Release() {
    if (api_ != nullptr) {
      if (memory_ != nullptr) api_->ReleaseMemoryInfo(memory_);
      if (session_ != nullptr) api_->ReleaseSession(session_);
      if (options_ != nullptr) api_->ReleaseSessionOptions(options_);
      if (env_ != nullptr) api_->ReleaseEnv(env_);
    }
    if (library_ != nullptr) dlclose(library_);
  }

  void* library_ = nullptr;
  const OrtApi* api_ = nullptr;
  OrtEnv* env_ = nullptr;
  OrtSessionOptions* options_ = nullptr;
  OrtSession* session_ = nullptr;
  OrtMemoryInfo* memory_ = nullptr;
  std::vector<std::string> output_names_;
};

// 计算两个候选框的交并比.
float IoU(const Detection& left, const Detection& right) {
  const float width = std::max(0.0F, std::min(left.box[2], right.box[2]) -
                                       std::max(left.box[0], right.box[0]));
  const float height = std::max(0.0F, std::min(left.box[3], right.box[3]) -
                                        std::max(left.box[1], right.box[1]));
  const float intersection = width * height;
  const float area_left = std::max(0.0F, left.box[2] - left.box[0]) *
                          std::max(0.0F, left.box[3] - left.box[1]);
  const float area_right = std::max(0.0F, right.box[2] - right.box[0]) *
                           std::max(0.0F, right.box[3] - right.box[1]);
  return intersection / std::max(area_left + area_right - intersection, 1e-12F);
}

// 解码三尺度距离框、类别 Sigmoid 和逐类别 NMS.
std::vector<Detection> Decode(const std::array<std::vector<float>, 6>& outputs,
                              const Transform& transform) {
  constexpr std::array<int, 3> widths = {80, 40, 20};
  constexpr std::array<int, 3> strides = {8, 16, 32};
  std::array<std::vector<Detection>, 80> by_class;
  for (size_t level = 0; level < 3; ++level) {
    const int width = widths[level];
    const int cells = width * width;
    const auto& classes = outputs[level * 2];
    const auto& distances = outputs[level * 2 + 1];
    for (int cell = 0; cell < cells; ++cell) {
      const float center_x = (cell % width + 0.5F) * strides[level];
      const float center_y = (cell / width + 0.5F) * strides[level];
      const std::array<float, 4> box = {
          center_x - distances[cell] * strides[level],
          center_y - distances[cells + cell] * strides[level],
          center_x + distances[2 * cells + cell] * strides[level],
          center_y + distances[3 * cells + cell] * strides[level]};
      for (int category = 0; category < 80; ++category) {
        const float logit = classes[category * cells + cell];
        const float score = 1.0F / (1.0F +
            std::exp(-std::clamp(logit, -30.0F, 30.0F)));
        if (score > 0.001F) {
          by_class[category].push_back({category, score, box});
        }
      }
    }
  }
  std::vector<Detection> kept;
  for (auto& candidates : by_class) {
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Detection& left, const Detection& right) {
                       return left.score > right.score;
                     });
    std::vector<bool> suppressed(candidates.size(), false);
    size_t class_count = 0;
    for (size_t index = 0; index < candidates.size(); ++index) {
      if (suppressed[index]) continue;
      kept.push_back(candidates[index]);
      if (++class_count == 300) break;
      for (size_t other = index + 1; other < candidates.size(); ++other) {
        if (!suppressed[other] && IoU(candidates[index], candidates[other]) >
                                      0.65F) {
          suppressed[other] = true;
        }
      }
    }
  }
  std::stable_sort(kept.begin(), kept.end(),
                   [](const Detection& left, const Detection& right) {
                     return left.score > right.score;
                   });
  if (kept.size() > 300) kept.resize(300);
  for (Detection& detection : kept) {
    detection.box[0] = std::clamp(detection.box[0] / transform.scale -
                                      transform.pad_x, 0.0F,
                                  static_cast<float>(transform.width));
    detection.box[2] = std::clamp(detection.box[2] / transform.scale -
                                      transform.pad_x, 0.0F,
                                  static_cast<float>(transform.width));
    detection.box[1] = std::clamp(detection.box[1] / transform.scale -
                                      transform.pad_y, 0.0F,
                                  static_cast<float>(transform.height));
    detection.box[3] = std::clamp(detection.box[3] / transform.scale -
                                      transform.pad_y, 0.0F,
                                  static_cast<float>(transform.height));
  }
  return kept;
}

// 序列化一张图片的 C++ 预测和 ORT 推理耗时.
void WritePrediction(std::ostream& output, const fs::path& image,
                     double inference_ms,
                     const std::vector<Detection>& detections) {
  output << "{\"image\":\"" << image.filename().string()
         << "\",\"inference_ms\":" << inference_ms
         << ",\"detections\":[";
  for (size_t index = 0; index < detections.size(); ++index) {
    if (index != 0) output << ',';
    const Detection& detection = detections[index];
    output << "{\"class_id\":" << detection.class_id
           << ",\"score\":" << detection.score << ",\"bbox_xyxy\":["
           << detection.box[0] << ',' << detection.box[1] << ','
           << detection.box[2] << ',' << detection.box[3] << "]}";
  }
  output << "]}\n";
}

// 执行预热、全量 5000 张推理并保存 profiling 事件路径.
int main(int argc, char** argv) {
  try {
    const Options options = ParseArgs(argc, argv);
    const std::vector<fs::path> images = ListImages(options.images);
    fs::create_directories(options.output_dir);
    BoardSession session(options.model, options.output_dir / "ort_neuron");
    const cv::Mat first = cv::imread(images.front().string());
    if (first.empty()) throw std::runtime_error("首张图片无法解码.");
    auto [warmup_input, transform] = Preprocess(first);
    for (int index = 0; index < 3; ++index) {
      session.Infer(warmup_input);
      std::cout << "[WARMUP] " << index + 1 << "/3.\n";
    }
    // 仅预热阶段开启 profiling,避免 5000 张的事件文件耗尽板端空间.
    const fs::path profile = session.FinishProfiling();
    std::ofstream profile_path(options.output_dir / "profile_path.txt");
    profile_path << profile.string() << '\n';
    if (!profile_path) throw std::runtime_error("profiling 路径写入失败.");
    std::ofstream output(options.output_dir / "results.jsonl");
    if (!output) throw std::runtime_error("无法写入预测文件.");
    output << std::setprecision(9);
    for (size_t index = 0; index < images.size(); ++index) {
      const cv::Mat image = cv::imread(images[index].string());
      if (image.empty()) {
        throw std::runtime_error("无法解码图片: " + images[index].string());
      }
      auto [input, transform] = Preprocess(image);
      const auto [raw, inference_ms] = session.Infer(input);
      const auto detections = Decode(raw, transform);
      WritePrediction(output, images[index], inference_ms, detections);
      output.flush();
      if (!output) throw std::runtime_error("预测文件写入失败.");
      if ((index + 1) % 50 == 0 || index + 1 == images.size()) {
        std::cout << "[PROGRESS] YOLO-World " << index + 1 << '/'
                  << images.size() << " 张.\n";
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << '\n';
    return 1;
  }
}
