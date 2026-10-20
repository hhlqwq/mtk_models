// 两个 ImageNet 归一化模型共享的输入量化与单图命令行契约.
#ifndef MTK_MODELS_COMMON_CPP_IMAGE_H_
#define MTK_MODELS_COMMON_CPP_IMAGE_H_

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "board.h"

namespace mtk {

struct Int8Config {
    float input_scale = 0, output_scale = 0;
    int input_zero = 0, output_zero = 0;
};

// 每行分别读取输入和输出 scale、zero_point,严格拒绝额外字段.
inline Int8Config ReadInt8Config(const fs::path& path) {
    std::ifstream stream(path);
    Int8Config config;
    char first = 0, second = 0;
    std::string extra;
    if (!(stream >> config.input_scale >> first >> config.input_zero >>
          config.output_scale >> second >> config.output_zero) ||
        first != ',' || second != ',' || (stream >> extra) ||
        !std::isfinite(config.input_scale) || config.input_scale <= 0 ||
        !std::isfinite(config.output_scale) || config.output_scale <= 0 ||
        config.input_zero < -128 || config.input_zero > 127 ||
        config.output_zero < -128 || config.output_zero > 127) {
        throw std::runtime_error("INT8 张量配置无效: " + path.string());
    }
    return config;
}

// 两个模型显式选择尺寸和插值方式,保留既有 RGB CHW ImageNet 协议.
inline std::vector<int8_t> ImageNetInput(const cv::Mat& image, int size, int interpolation,
                                        const Int8Config& config) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(size, size), 0, 0, interpolation);
    const std::array<float, 3> mean{0.485F, 0.456F, 0.406F};
    const std::array<float, 3> deviation{0.229F, 0.224F, 0.225F};
    std::vector<int8_t> tensor(3 * size * size);
    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            const auto pixel = resized.at<cv::Vec3b>(row, col);
            for (int channel = 0; channel < 3; ++channel) {
                const float value = (pixel[2 - channel] / 255.0F - mean[channel]) / deviation[channel];
                const int quantized = static_cast<int>(std::nearbyint(value / config.input_scale)) +
                                      config.input_zero;
                tensor[(channel * size + row) * size + col] =
                    static_cast<int8_t>(std::clamp(quantized, -128, 127));
            }
        }
    }
    return tensor;
}

// 单图入口使用与 FastSAM 一致的 model、config、image、output-dir 与 quiet 参数.
inline std::map<std::string, std::string> ImageArguments(int argc, char** argv) {
    std::map<std::string, std::string> result;
    const std::vector<std::string> names{"--model", "--config", "--image", "--output-dir"};
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--quiet" || key == "--check-image") {
            if (!result.emplace(key, "1").second) throw std::invalid_argument("重复参数: " + key);
        } else if (std::find(names.begin(), names.end(), key) != names.end() && index + 1 < argc) {
            if (!result.emplace(key, argv[++index]).second) throw std::invalid_argument("重复参数: " + key);
        } else {
            throw std::invalid_argument("未知参数或缺少参数值: " + key);
        }
    }
    for (const auto& name : names) {
        if (!result.count(name) && (name != "--model" || !result.count("--check-image"))) {
            throw std::invalid_argument("缺少参数: " + name);
        }
    }
    return result;
}

// 写入完整二进制张量,所有写入错误立即中止当前图片.
template <typename T>
void WriteTensor(const fs::path& path, const std::vector<T>& values) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    if (!stream) throw std::runtime_error("写入张量失败: " + path.string());
}

// 记录实际单图 SDK 调用耗时、原图尺寸与当前 C++ 进程峰值 RSS.
inline void WriteImageTiming(const fs::path& path, double milliseconds, const cv::Mat& image) {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0 || !std::isfinite(milliseconds) || milliseconds <= 0) {
        throw std::runtime_error("推理耗时或峰值内存无效.");
    }
    std::ofstream stream(path);
    stream << std::setprecision(12) << "{\"backend\":\"cpp_neuron_runtime_hw\",\"npu_ms\":"
           << milliseconds << ",\"peak_rss_kib\":" << usage.ru_maxrss
           << ",\"height\":" << image.rows << ",\"width\":" << image.cols << "}\n";
    if (!stream) throw std::runtime_error("写入推理统计失败.");
}

}  // namespace mtk
#endif  // MTK_MODELS_COMMON_CPP_IMAGE_H_
