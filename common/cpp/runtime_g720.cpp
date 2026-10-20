// G720 Neuron Runtime 适配,独立管理原生缓冲区与生命周期.
#include "runtime_g720.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>

#include "board.h"
#include "neuron/api/RuntimeAPI.h"

namespace mtk {
namespace {

// 仅接受已知的连续布局或 16 字节行补齐,拒绝猜测其他平台布局.
size_t RowStride(size_t bytes, size_t rows, size_t width) {
    if (rows == 0 || width == 0) throw std::runtime_error("张量尺寸必须为正.");
    const size_t padded = (width + 15) / 16 * 16;
    if (bytes != rows * width && bytes != rows * padded) {
        throw std::runtime_error("G720 张量字节数与逻辑形状不匹配: " + std::to_string(bytes));
    }
    return bytes / rows;
}

}  // namespace

struct Int8Runtime::Impl {
    // 保证模型加载中途失败也释放已创建的 Runtime.
    Impl(const std::filesystem::path& model, size_t in_rows, size_t in_width,
         size_t out_rows, size_t out_width)
        : input_rows(in_rows), input_width(in_width), output_rows(out_rows),
          output_width(out_width) {
        EnvOptions options{};
        options.deviceKind = kEnvOptHardware;
        options.MDLACoreOption = Auto;
        options.CPUThreadNum = 1;
        options.suppressInputConversion = true;
        options.suppressOutputConversion = true;
        CheckRuntime(NeuronRuntime_create(&options, &runtime), "create");
        try {
            CheckRuntime(NeuronRuntime_loadNetworkFromFile(runtime, model.c_str()), "load");
            size_t inputs = 0, outputs = 0, input_bytes = 0, output_bytes = 0;
            CheckRuntime(NeuronRuntime_getInputNumber(runtime, &inputs), "input count");
            CheckRuntime(NeuronRuntime_getOutputNumber(runtime, &outputs), "output count");
            if (inputs != 1 || outputs != 1) throw std::runtime_error("要求单输入单输出模型.");
            CheckRuntime(NeuronRuntime_getSingleInputPaddedSize(runtime, &input_bytes), "input size");
            CheckRuntime(NeuronRuntime_getOutputPaddedSize(runtime, 0, &output_bytes), "output size");
            input_stride = RowStride(input_bytes, input_rows, input_width);
            output_stride = RowStride(output_bytes, output_rows, output_width);
            input.resize(input_bytes);
            output.resize(output_bytes);
        } catch (...) {
            NeuronRuntime_release(runtime);
            runtime = nullptr;
            throw;
        }
    }

    // 异常和正常退出均释放 Runtime.
    ~Impl() {
        if (runtime != nullptr) NeuronRuntime_release(runtime);
    }

    void* runtime = nullptr;
    size_t input_rows, input_width, output_rows, output_width;
    size_t input_stride = 0, output_stride = 0;
    std::vector<int8_t> input, output;
    bool inferred = false;
};

// 将逻辑张量描述交给当前平台实现.
Int8Runtime::Int8Runtime(const std::filesystem::path& model, size_t input_rows,
                        size_t input_width, size_t output_rows, size_t output_width)
    : impl_(std::make_unique<Impl>(model, input_rows, input_width, output_rows, output_width)) {}

// 在实现类型完整的位置销毁平台实例.
Int8Runtime::~Int8Runtime() = default;

// 输入转换、模型加载与后处理均不计入纯推理耗时.
double Int8Runtime::Infer(const std::vector<int8_t>& values, int zero_point) {
    auto& p = *impl_;
    if (values.size() != p.input_rows * p.input_width || zero_point < -128 || zero_point > 127) {
        throw std::runtime_error("输入元素数量或量化零点无效.");
    }
    p.inferred = false;
    std::fill(p.input.begin(), p.input.end(), static_cast<int8_t>(zero_point));
    for (size_t row = 0; row < p.input_rows; ++row) {
        std::copy_n(values.begin() + row * p.input_width, p.input_width,
                    p.input.begin() + row * p.input_stride);
    }
    const BufferAttribute attribute{NON_ION_FD};
    CheckRuntime(NeuronRuntime_setSingleInput(p.runtime, p.input.data(), p.input.size(), attribute),
                 "set input");
    CheckRuntime(NeuronRuntime_setOutput(p.runtime, 0, p.output.data(), p.output.size(), attribute),
                 "set output");
    const auto start = std::chrono::steady_clock::now();
    CheckRuntime(NeuronRuntime_inference(p.runtime), "inference");
    const double milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    p.inferred = true;
    return milliseconds;
}

// 只在推理成功后读取输出并去除 G720 补齐.
std::vector<int8_t> Int8Runtime::Output() const {
    const auto& p = *impl_;
    if (!p.inferred) throw std::runtime_error("尚无成功的推理输出.");
    std::vector<int8_t> values(p.output_rows * p.output_width);
    for (size_t row = 0; row < p.output_rows; ++row) {
        std::copy_n(p.output.begin() + row * p.output_stride, p.output_width,
                    values.begin() + row * p.output_width);
    }
    return values;
}

}  // namespace mtk
