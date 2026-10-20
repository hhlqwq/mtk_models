// 完成单图预处理、G720 SDK 硬件推理与模型后处理.
#include <exception>
#include <iostream>

#include "common/cpp/image.h"
#include "common/cpp/runtime_g720.h"

namespace {
constexpr int kSize = 518;

// 模型尺寸和插值规则在模型源码中声明,不在 Python 中执行预处理.
std::vector<int8_t> Preprocess(const cv::Mat& image, const mtk::Int8Config& config) {
    return mtk::ImageNetInput(image, kSize, cv::INTER_CUBIC, config);
}

// 执行当前模型后处理,保存可直接用于指标计算的连续 FP32 输出.
void Postprocess(const std::vector<int8_t>& raw, const mtk::Int8Config& config,
                 const cv::Mat& image, const mtk::fs::path& output) {
    cv::Mat depth(kSize, kSize, CV_32F);
    for (int row = 0; row < kSize; ++row) {
        for (int col = 0; col < kSize; ++col) {
            depth.at<float>(row, col) = (static_cast<int>(raw[row * kSize + col]) -
                                         config.output_zero) * config.output_scale;
        }
    }
    cv::Scalar mean, deviation;
    cv::meanStdDev(depth, mean, deviation);
    if (!cv::checkRange(depth) || deviation[0] <= 0) throw std::runtime_error("深度图无效或恒定.");
    cv::Mat restored;
    cv::resize(depth, restored, image.size(), 0, 0, cv::INTER_LINEAR);
    mtk::WriteTensor(output / "depth.bin", std::vector<float>(restored.ptr<float>(),
                     restored.ptr<float>() + restored.total()));
}

// 与 FastSAM 相同,单图命令直接接收图片,平台实现封装 SDK 调用.
int Run(int argc, char** argv) {
    const auto args = mtk::ImageArguments(argc, argv);
    const auto config = mtk::ReadInt8Config(args.at("--config"));
    const cv::Mat image = cv::imread(args.at("--image"), cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("无法读取图片: " + args.at("--image"));
    const auto input = Preprocess(image, config);
    const mtk::fs::path output = args.at("--output-dir");
    if (!mtk::fs::create_directories(output)) throw std::runtime_error("输出目录已存在: " + output.string());
    if (args.count("--check-image")) {
        mtk::WriteTensor(output / "input_int8.bin", input);
        return 0;
    }
    mtk::Int8Runtime runtime(args.at("--model"), 3 * kSize, kSize, 518, 518);
    const double milliseconds = runtime.Infer(input, config.input_zero);
    const auto raw = runtime.Output();
    Postprocess(raw, config, image, output);
    mtk::WriteTensor(output / "output_int8.bin", raw);
    mtk::WriteImageTiming(output / "timing.json", milliseconds, image);
    if (!args.count("--quiet")) std::cout << "[C++] NPU 推理完成: " << milliseconds << " ms; " << output << '\n';
    return 0;
}

}  // namespace

// 统一将可读错误写入标准错误,以非零返回码交给流程入口处理.
int main(int argc, char** argv) {
    try {
        return Run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "[错误] " << error.what() << '\n';
        return 1;
    }
}
