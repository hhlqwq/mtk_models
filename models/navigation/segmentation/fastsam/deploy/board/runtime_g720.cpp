// G720 专用实现: Neuron Runtime、INT8 量化与原生缓冲区对齐.
#include "fastsam_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "neuron/api/RuntimeAPI.h"
#include "common/cpp/board.h"

namespace fastsam {
namespace fs = std::filesystem;
namespace {

struct TensorInfo {
    std::string name;
    int index = 0;
    int channels = 0;
    int height = 0;
    int width = 0;
    float scale = 0.0F;
    int zero_point = 0;
};

// 检查 Runtime 返回值并保留具体操作名称.
using mtk::CheckRuntime;

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
        for (int i = 0; i < 10; ++i) {
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
    if (rows != 11 || !std::all_of(seen.begin(), seen.end(),
                                   [](bool value) { return value; })) {
        throw std::runtime_error("张量配置缺少输入或输出.");
    }
    if (result[0].channels != 3 || result[0].height != kSize ||
        result[0].width != kSize) {
        throw std::runtime_error("输入张量必须为 [1,3,640,640].");
    }
    for (int level = 0; level < 3; ++level) {
        for (int kind = 0; kind < 3; ++kind) {
            const auto& tensor = result[1 + level * 3 + kind];
            if (tensor.channels != (kind == 0   ? 64
                                    : kind == 1 ? 1
                                                : 32) ||
                tensor.height != kSize / kStrides[level] ||
                tensor.width != kSize / kStrides[level] || tensor.index > 9) {
                throw std::runtime_error("检测头张量形状或索引错误: " +
                                         tensor.name);
            }
        }
    }
    if (result[10].channels != 32 || result[10].height != 160 ||
        result[10].width != 160 || result[10].index > 9) {
        throw std::runtime_error("掩码原型形状或索引错误.");
    }
    std::array<bool, 10> indices{};
    for (int slot = 1; slot <= 10; ++slot) {
        if (indices[result[slot].index])
            throw std::runtime_error("输出索引重复.");
        indices[result[slot].index] = true;
    }
    return result;
}

// 保存原生 INT8 数据,文件名沿用已有板端诊断约定.
void SaveBytes(const fs::path& path, const std::vector<int8_t>& bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    if (!stream) throw std::runtime_error("无法保存原生张量: " + path.string());
}

}  // namespace

struct FastSamRuntime::Impl {
    // 空配置仅供原有图片检查模式使用,不允许加载模型.
    explicit Impl(const fs::path& path) : has_config_(!path.empty()) {
        if (has_config_) {
            config_ = LoadConfig(path);
        } else {
            config_[0] =
                TensorInfo{"input", 0, 3, kSize, kSize, 1.0F / 255, -128};
        }
    }

    // 释放 G720 运行时,保证异常时也不泄漏资源.
    ~Impl() {
        if (runtime_ != nullptr) NeuronRuntime_release(runtime_);
    }

    // 将归一化的 RGB CHW 浮点输入转换为 G720 模型所需的 INT8.
    void PrepareInput(const std::vector<float>& input) {
        if (input.size() != 3 * kSize * kSize) {
            throw std::runtime_error("FastSAM 输入元素数量不匹配.");
        }
        const auto& info = config_[0];
        input_.resize(input.size());
        for (size_t index = 0; index < input.size(); ++index) {
            if (!std::isfinite(input[index]) || input[index] < 0 ||
                input[index] > 1) {
                throw std::runtime_error(
                    "FastSAM 输入需要为 [0,1] 范围的有限浮点数.");
            }
            const int integer =
                static_cast<int>(std::nearbyint(input[index] / info.scale)) +
                info.zero_point;
            input_[index] = static_cast<int8_t>(std::clamp(integer, -128, 127));
        }
        inferred_ = false;
    }

    // 加载 G720 DLA,验证实际 IO 数量、字节数及输出行宽对齐.
    void Load(const fs::path& path) {
        if (!has_config_ || runtime_ != nullptr) {
            throw std::runtime_error(
                "加载模型需要有效配置,且同一实例不能重复加载.");
        }
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
            if (input_count != 1 || output_count != 10 ||
                input_size != 3 * kSize * kSize) {
                throw std::runtime_error("DLA 输入输出数量或输入字节数不匹配.");
            }
            if (input_.size() != input_size) {
                throw std::runtime_error("请先准备完整的模型输入.");
            }
            outputs_.resize(output_count);
            for (int slot = 1; slot <= 10; ++slot) {
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

    // 输入转换与模型加载不计入纯硬件推理耗时.
    double Infer() {
        if (runtime_ == nullptr || input_.size() != 3 * kSize * kSize) {
            throw std::runtime_error("推理前需要准备输入并加载模型.");
        }
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
        inferred_ = true;
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
            .count();
    }

    // 恢复模型语义顺序,去除 G720 行宽填充并反量化.
    OutputTensors Outputs() const {
        if (!inferred_) throw std::runtime_error("推理完成后才能读取输出.");
        OutputTensors result;
        for (size_t slot = 0; slot < result.size(); ++slot) {
            const auto& info = config_[slot + 1];
            const auto& bytes = outputs_.at(info.index);
            auto& tensor = result[slot];
            tensor.channels = info.channels;
            tensor.height = info.height;
            tensor.width = info.width;
            const size_t count =
                static_cast<size_t>(info.channels) * info.height * info.width;
            tensor.values.resize(count);
            const int row_width = bytes.size() == count
                                      ? info.width
                                      : (info.width + 15) / 16 * 16;
            for (int channel = 0; channel < info.channels; ++channel) {
                for (int row = 0; row < info.height; ++row) {
                    for (int column = 0; column < info.width; ++column) {
                        const size_t base =
                            static_cast<size_t>(channel) * info.height + row;
                        tensor.values[base * info.width + column] =
                            (static_cast<int>(
                                 bytes.at(base * row_width + column)) -
                             info.zero_point) *
                            info.scale;
                    }
                }
            }
        }
        return result;
    }

    // 保留原始输入与十个输出文件,图片检查模式只保存输入.
    void SaveRawTensors(const fs::path& directory) const {
        SaveBytes(directory / "input_int8.bin", input_);
        if (!inferred_) return;
        for (size_t slot = 0; slot < kNames.size(); ++slot) {
            SaveBytes(directory / (std::string(kNames[slot]) + ".bin"),
                      outputs_.at(config_[slot + 1].index));
        }
    }

    std::array<TensorInfo, 11> config_{};
    bool has_config_ = false;
    bool inferred_ = false;
    void* runtime_ = nullptr;
    std::vector<int8_t> input_;
    std::vector<std::vector<int8_t>> outputs_;
};

// 平台 SDK 和原生张量描述仅保存在实现文件中.
FastSamRuntime::FastSamRuntime(const fs::path& config)
    : impl_(std::make_unique<Impl>(config)) {}

// 在实现类型完整的位置释放平台资源.
FastSamRuntime::~FastSamRuntime() = default;

// 转交 G720 输入转换,通用算法不接触量化参数.
void FastSamRuntime::PrepareInput(const std::vector<float>& input) {
    impl_->PrepareInput(input);
}

// 转交 G720 DLA 加载.
void FastSamRuntime::Load(const fs::path& model) { impl_->Load(model); }

// 转交一次 G720 硬件推理.
double FastSamRuntime::Infer() { return impl_->Infer(); }

// 返回与 Runtime 类型无关的标准张量.
OutputTensors FastSamRuntime::Outputs() const { return impl_->Outputs(); }

// 保存兼容现有部署工具的原生诊断文件.
void FastSamRuntime::SaveRawTensors(const fs::path& directory) const {
    impl_->SaveRawTensors(directory);
}

// 保留已有结果 JSON 的后端标识.
const char* FastSamRuntime::BackendName() const {
    return "cpp_neuron_runtime_hw";
}

}  // namespace fastsam
