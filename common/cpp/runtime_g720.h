// G720 单输入单输出 INT8 运行时,公共接口不暴露芯片 SDK 类型.
#ifndef MTK_MODELS_COMMON_CPP_RUNTIME_G720_H_
#define MTK_MODELS_COMMON_CPP_RUNTIME_G720_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace mtk {

class Int8Runtime {
    public:
        // 加载模型并核对逻辑张量与平台补齐布局,尺寸单位为元素.
        Int8Runtime(const std::filesystem::path& model, size_t input_rows, size_t input_width,
                    size_t output_rows, size_t output_width);
        // 释放平台运行时与原生缓冲区.
        ~Int8Runtime();
        Int8Runtime(const Int8Runtime&) = delete;
        Int8Runtime& operator=(const Int8Runtime&) = delete;

        // 接收连续 INT8 输入并补齐行宽,只返回实际 SDK 推理调用耗时.
        double Infer(const std::vector<int8_t>& input, int zero_point);
        // 去除平台行宽补齐,返回连续 INT8 输出.
        std::vector<int8_t> Output() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
};

}  // namespace mtk
#endif  // MTK_MODELS_COMMON_CPP_RUNTIME_G720_H_
