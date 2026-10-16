// FastSAM 的模型张量约定与板端接口,不暴露芯片 SDK 类型.
#ifndef MTK_MODELS_FASTSAM_RUNTIME_H_
#define MTK_MODELS_FASTSAM_RUNTIME_H_

#include <array>
#include <filesystem>
#include <memory>
#include <vector>

namespace fastsam {

constexpr int kSize = 640;
constexpr int kCoefficientCount = 32;
constexpr std::array<int, 3> kStrides = {8, 16, 32};
constexpr std::array<const char*, 10> kNames = {
    "box_8", "score_8", "coefficient_8", "box_16", "score_16",
    "coefficient_16", "box_32", "score_32", "coefficient_32", "prototype"};

// 后处理只读取连续的 CHW 浮点张量,平台实现负责量化和内存布局转换.
struct OutputTensor {
    int channels = 0;
    int height = 0;
    int width = 0;
    std::vector<float> values;
};

using OutputTensors = std::array<OutputTensor, 10>;

class FastSamRuntime {
 public:
    // 读取张量配置; 空路径仅用于不加载模型的 --check-image 模式.
    explicit FastSamRuntime(const std::filesystem::path& config);
    // 释放平台运行时及其缓冲区.
    ~FastSamRuntime();
    FastSamRuntime(const FastSamRuntime&) = delete;
    FastSamRuntime& operator=(const FastSamRuntime&) = delete;

    // 接收归一化的 RGB CHW 输入,由平台实现转换为实际输入格式.
    void PrepareInput(const std::vector<float>& input);
    // 加载本平台编译的模型,不假设 G720 模型可以供 G5100 使用.
    void Load(const std::filesystem::path& model);
    // 执行一次硬件推理,返回纯 Runtime 推理调用的毫秒数.
    double Infer();
    // 将原生输出恢复为固定语义顺序的连续 CHW 浮点张量.
    OutputTensors Outputs() const;
    // 保存本平台原始输入输出,保留已有诊断文件格式.
    void SaveRawTensors(const std::filesystem::path& directory) const;
    // 返回结果记录使用的后端名称.
    const char* BackendName() const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace fastsam

#endif  // MTK_MODELS_FASTSAM_RUNTIME_H_
