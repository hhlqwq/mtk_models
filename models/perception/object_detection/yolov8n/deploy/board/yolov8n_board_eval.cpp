// 在 Genio 720 板端使用 C++ 完成图片预处理、NPU 推理与YOLOv8n 检测与评测.
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
#include <numeric>
#include <unordered_set>
#include <cctype>
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
#include "common/cpp/board.h"
#include "common/cpp/detection.h"

namespace fs = std::filesystem;

namespace {

constexpr int kSize = 640;
constexpr std::array<int, 3> kStrides = {8, 16, 32};
constexpr std::array<const char*, 6> kNames = {
    "box_8", "score_8", "box_16", "score_16", "box_32", "score_32"};
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

// 读取当前进程的历史峰值常驻内存,Linux 返回单位为 KiB.
using mtk::GetPeakRssKb;

// 检查 Runtime 返回值并保留具体操作名称.
using mtk::CheckRuntime;

// 计算稳定的 Sigmoid.
using mtk::Sigmoid;

// 计算两个 xyxy 检测框的 IoU.
using mtk::BoxIou;

// 解析正整数参数.
using mtk::ParseInteger;

// 解析浮点参数.
using mtk::ParseFloat;

// 解析命令行参数.
Options ParseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (index + 1 >= argc) {
            throw std::invalid_argument("参数缺少值: " + key);
        }
        const std::string value = argv[++index];
        if (key == "--config") {
            options.config = value;
        } else if (key == "--model") {
            options.model = value;
        } else if (key == "--images") {
            options.images = value;
        } else if (key == "--output-dir") {
            options.output_dir = value;
        } else if (key == "--limit") {
            options.limit = ParseInteger(value, key);
        } else if (key == "--warmup") {
            options.warmup = ParseInteger(value, key);
        } else if (key == "--progress-interval") {
            options.progress_interval = ParseInteger(value, key);
        } else if (key == "--max-det") {
            options.max_detections = ParseInteger(value, key);
        } else if (key == "--confidence") {
            options.confidence = ParseFloat(value, key);
        } else if (key == "--iou") {
            options.iou = ParseFloat(value, key);
        } else {
            throw std::invalid_argument("未知参数: " + key);
        }
    }
    if (options.config.empty() || options.model.empty() ||
        options.images.empty() || options.output_dir.empty()) {
        throw std::invalid_argument(
            "必须提供 --model、--images 和 --output-dir.");
    }
    if (options.limit < 0 || options.warmup < 0 ||
        options.progress_interval <= 0 || options.max_detections <= 0 ||
        options.confidence < 0 || options.confidence > 1 || options.iou < 0 ||
        options.iou > 1) {
        throw std::invalid_argument("次数参数必须位于有效范围.");
    }
    return options;
}

// 返回目录中的 JPEG 或 PNG 图片,并按文件名排序.
using mtk::ListImages;

// 将 COCO 文件名转换为数值 image_id.
using mtk::ImageIdFromPath;

// 从完成清单中读取已处理 image_id.
using mtk::LoadCompleted;

// 严格读取由本模型的量化脚本生成的固定七列张量契约.
std::array<TensorInfo, 7> LoadConfig(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("无法读取张量配置: " + path.string());
    std::array<TensorInfo, 7> result;
    std::array<bool, 7> seen{};
    std::string line;
    int rows = 0;
    while (std::getline(input, line)) {
        std::istringstream stream(line);
        std::vector<std::string> columns;
        std::string value;
        while (std::getline(stream, value, ',')) columns.push_back(value);
        if (columns.size() != 7)
            throw std::runtime_error("张量配置字段数错误.");
        TensorInfo tensor{columns[0],
                          std::stoi(columns[1]),
                          std::stoi(columns[2]),
                          std::stoi(columns[3]),
                          std::stoi(columns[4]),
                          std::stof(columns[5]),
                          std::stoi(columns[6])};
        int slot = -1;
        if (tensor.name == "input") slot = 0;
        for (int i = 0; i < 6; ++i) {
            if (tensor.name == kNames[i]) slot = i + 1;
        }
        if (slot < 0 || seen[slot] || tensor.scale <= 0 ||
            tensor.zero_point < -128 || tensor.zero_point > 127 ||
            tensor.index < 0) {
            throw std::runtime_error("张量配置含未知、重复或无效项目: " +
                                     tensor.name);
        }
        seen[slot] = true;
        result[slot] = std::move(tensor);
        ++rows;
    }
    if (rows != 7 || !std::all_of(seen.begin(), seen.end(),
                                  [](bool value) { return value; })) {
        throw std::runtime_error("张量配置缺少输入或输出.");
    }
    if (result[0].channels != 3 || result[0].height != kSize ||
        result[0].width != kSize) {
        throw std::runtime_error("输入张量必须为 [1,3,640,640].");
    }
    for (int level = 0; level < 3; ++level) {
        for (int kind = 0; kind < 2; ++kind) {
            const auto& tensor = result[1 + level * 2 + kind];
            if (tensor.channels != (kind == 0 ? 64 : 80) ||
                tensor.height != kSize / kStrides[level] ||
                tensor.width != kSize / kStrides[level] || tensor.index > 5) {
                throw std::runtime_error("检测头张量形状或索引错误: " +
                                         tensor.name);
            }
        }
    }
    std::array<bool, 6> indices{};
    for (int slot = 1; slot <= 6; ++slot) {
        if (indices[result[slot].index])
            throw std::runtime_error("输出索引重复.");
        indices[result[slot].index] = true;
    }
    return result;
}

// 用与导出脚本一致的居中 letterbox 和 RGB NCHW 量化准备单图.
std::pair<std::vector<int8_t>, LetterboxInfo> Preprocess(
    const cv::Mat& image, const TensorInfo& input) {
    if (image.empty()) throw std::runtime_error("图片为空.");
    LetterboxInfo geo;
    geo.original_width = image.cols;
    geo.original_height = image.rows;
    geo.scale = std::min(static_cast<float>(kSize) / image.cols,
                         static_cast<float>(kSize) / image.rows);
    // Python round 在 .5 处取最近偶数,此处必须和导出基线保持一致.
    const int width = static_cast<int>(std::nearbyint(image.cols * geo.scale));
    const int height = static_cast<int>(std::nearbyint(image.rows * geo.scale));
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
                const int integer =
                    static_cast<int>(std::nearbyint(normalized / input.scale)) +
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
    NeuronModel(const fs::path& path, const std::array<TensorInfo, 7>& config)
        : config_(config) {
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
                NeuronRuntime_loadNetworkFromFile(runtime_, path.c_str()),
                "NeuronRuntime_loadNetworkFromFile");
            size_t input_count = 0, output_count = 0, input_size = 0;
            CheckRuntime(NeuronRuntime_getInputNumber(runtime_, &input_count),
                         "getInputNumber");
            CheckRuntime(NeuronRuntime_getOutputNumber(runtime_, &output_count),
                         "getOutputNumber");
            CheckRuntime(
                NeuronRuntime_getSingleInputPaddedSize(runtime_, &input_size),
                "getSingleInputPaddedSize");
            if (input_count != 1 || output_count != 6 ||
                input_size != 3 * kSize * kSize) {
                throw std::runtime_error("DLA 输入输出数量或输入字节数不匹配.");
            }
            input_.resize(input_size);
            outputs_.resize(output_count);
            for (int slot = 1; slot <= 6; ++slot) {
                const auto& tensor = config_[slot];
                size_t size = 0;
                CheckRuntime(NeuronRuntime_getOutputPaddedSize(
                                 runtime_, tensor.index, &size),
                             "getOutputPaddedSize");
                const size_t width = static_cast<size_t>(tensor.width);
                const size_t plain = tensor.channels * tensor.height * width;
                const size_t padded =
                    tensor.channels * tensor.height * ((width + 15) / 16 * 16);
                if (size != plain && size != padded) {
                    throw std::runtime_error(
                        "DLA 输出字节数不匹配: " + tensor.name + ",实际 " +
                        std::to_string(size));
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
        if (input.size() != input_.size())
            throw std::runtime_error("输入字节数不匹配.");
        std::copy(input.begin(), input.end(), input_.begin());
        const BufferAttribute attribute{NON_ION_FD};
        CheckRuntime(NeuronRuntime_setSingleInput(runtime_, input_.data(),
                                                  input_.size(), attribute),
                     "setSingleInput");
        for (size_t index = 0; index < outputs_.size(); ++index) {
            CheckRuntime(
                NeuronRuntime_setOutput(runtime_, index, outputs_[index].data(),
                                        outputs_[index].size(), attribute),
                "setOutput");
        }
        const auto start = std::chrono::steady_clock::now();
        CheckRuntime(NeuronRuntime_inference(runtime_),
                     "NeuronRuntime_inference");
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
            .count();
    }

    // 按语义编号访问输出原生字节,不假设编译器保留输出顺序.
    const std::vector<int8_t>& Output(int slot) const {
        return outputs_.at(config_.at(slot).index);
    }

   private:
    const std::array<TensorInfo, 7>& config_;
    void* runtime_ = nullptr;
    std::vector<int8_t> input_;
    std::vector<std::vector<int8_t>> outputs_;
};

// 按 NCHW 和 16 元素行宽对齐读取、反量化一个原始输出元素.
float ReadValue(const NeuronModel& model, const TensorInfo& info, int slot,
                int channel, int row, int column) {
    const auto& bytes = model.Output(slot);
    const size_t plane =
        static_cast<size_t>(info.channels) * info.height * info.width;
    const int padded_width =
        bytes.size() == plane ? info.width : (info.width + 15) / 16 * 16;
    const size_t offset =
        (static_cast<size_t>(channel) * info.height + row) * padded_width +
        column;
    return (static_cast<int>(bytes.at(offset)) - info.zero_point) * info.scale;
}

constexpr std::array<int, 80> kCocoCategoryIds = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
};

// 解码原始分布头,选择每个位置的最佳类别,与浮点评测保持一致.
std::vector<Detection> Decode(const NeuronModel& model,
                              const std::array<TensorInfo, 7>& config,
                              float confidence) {
    std::vector<Detection> candidates;
    for (int level = 0; level < 3; ++level) {
        const int side = kSize / kStrides[level];
        const int slot = 1 + level * 2;
        for (int row = 0; row < side; ++row) {
            for (int column = 0; column < side; ++column) {
                float maximum_score = -std::numeric_limits<float>::infinity();
                int category = 0;
                for (int channel = 0; channel < 80; ++channel) {
                    const float value =
                        ReadValue(model, config[slot + 1], slot + 1, channel,
                                  row, column);
                    if (value > maximum_score) {
                        maximum_score = value;
                        category = channel;
                    }
                }
                const float score = Sigmoid(maximum_score);
                if (score <= confidence) continue;
                std::array<float, 4> distance{};
                for (int axis = 0; axis < 4; ++axis) {
                    float logits[16],
                        maximum = -std::numeric_limits<float>::infinity();
                    for (int bin = 0; bin < 16; ++bin) {
                        logits[bin] = ReadValue(model, config[slot], slot,
                                                axis * 16 + bin, row, column);
                        maximum = std::max(maximum, logits[bin]);
                    }
                    float sum = 0, weighted = 0;
                    for (int bin = 0; bin < 16; ++bin) {
                        const float weight = std::exp(logits[bin] - maximum);
                        sum += weight;
                        weighted += weight * bin;
                    }
                    distance[axis] = weighted / sum;
                }
                const float stride = kStrides[level];
                candidates.push_back({(column + 0.5F - distance[0]) * stride,
                                      (row + 0.5F - distance[1]) * stride,
                                      (column + 0.5F + distance[2]) * stride,
                                      (row + 0.5F + distance[3]) * stride,
                                      score, category});
            }
        }
    }
    return candidates;
}
// 执行与 torchvision batched_nms 等价的逐类别 NMS.
using mtk::ApplyNms;

// 将检测框映射回原图并裁剪到图像边界.
using mtk::Rescale;

// 转义 JSON 字符串中的特殊字符.
using mtk::EscapeJson;

// 将单张图片的检测结果追加为 JSONL.
using mtk::AppendPredictions;

// 将 JSONL 检测记录转换为 COCOeval 接受的 JSON 数组.
using mtk::FinalizePredictions;

// 计算延迟统计并写入 JSON.
using mtk::WriteSummary;

// 执行完整板端评测流程.
void Run(const Options& options) {
    if (fs::exists(options.output_dir))
        throw std::runtime_error("结果目录已存在,请使用新的运行目录.");
    fs::create_directories(options.output_dir);
    const auto images = ListImages(options.images, options.limit);
    const fs::path predictions_jsonl = options.output_dir / "predictions.jsonl";
    const fs::path predictions_json = options.output_dir / "predictions.json";
    const fs::path completed_path = options.output_dir / "processed_ids.txt";
    const fs::path timings_path = options.output_dir / "timings.csv";
    auto completed = LoadCompleted(completed_path);

    std::cout << "[1/4] 加载 DLA: " << options.model << std::endl;
    const auto config = LoadConfig(options.config);
    NeuronModel model(options.model, config);
    const auto warmup_input =
        Preprocess(cv::imread(images.front().string()), config[0]).first;
    std::cout << "[2/4] NPU 预热 " << options.warmup << " 次." << std::endl;
    for (int index = 0; index < options.warmup; ++index) {
        model.Infer(warmup_input);
    }

    std::ofstream predictions(predictions_jsonl, std::ios::app);
    std::ofstream completed_output(completed_path, std::ios::app);
    const bool timings_exist =
        fs::exists(timings_path) && fs::file_size(timings_path) > 0;
    std::ofstream timings_output(timings_path, std::ios::app);
    if (!timings_exist) {
        timings_output << "image_id,preprocess_ms,npu_ms,postprocess_ms,e2e_ms,"
                          "detections\n";
    }

    std::vector<Timing> current_timings;
    size_t processed_count = completed.size();
    std::cout << "[3/4] 板端逐图评测,共 " << images.size() << " 张,已完成 "
              << completed.size() << " 张." << std::endl;
    for (const fs::path& image_path : images) {
        const int64_t image_id = ImageIdFromPath(image_path);
        if (completed.contains(image_id)) {
            continue;
        }
        const auto e2e_start = std::chrono::steady_clock::now();
        const auto preprocess_start = e2e_start;
        auto [input, letterbox_info] =
            Preprocess(cv::imread(image_path.string()), config[0]);
        const auto preprocess_end = std::chrono::steady_clock::now();
        const double npu_ms = model.Infer(input);
        const auto postprocess_start = std::chrono::steady_clock::now();
        auto detections = ApplyNms(Decode(model, config, options.confidence),
                                   options.iou, options.max_detections, true);
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
        AppendPredictions(&predictions, image_id, detections, kCocoCategoryIds);
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
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << std::endl;
        return 1;
    }
}
