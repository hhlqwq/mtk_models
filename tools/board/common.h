// Linux 板端的文件、参数、运行时错误与耗时统计工具.
#ifndef MTK_MODELS_TOOLS_BOARD_COMMON_H_
#define MTK_MODELS_TOOLS_BOARD_COMMON_H_

#include <sys/resource.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace mtk {
namespace fs = std::filesystem;

// G720 Runtime 的零返回值表示成功,异常中保留操作名称和错误码.
inline void CheckRuntime(int result, const std::string& operation) {
    if (result != 0) {
        throw std::runtime_error(operation + " 失败,错误码 " + std::to_string(result));
    }
}

// 公用 整数参数解析.
inline int ParseInteger(const std::string& value, const std::string& name) {
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        throw std::invalid_argument("无效整数参数 " + name + ": " + value);
    }
}

// 公用 浮点参数解析.
inline float ParseFloat(const std::string& value, const std::string& name) {
    try {
        return std::stof(value);
    } catch (const std::exception&) {
        throw std::invalid_argument("无效浮点参数 " + name + ": " + value);
    }
}

// 公用 图片枚举,保留按文件名排序与数量限制.
inline std::vector<fs::path> ListImages(const fs::path& directory, int limit) {
    std::vector<fs::path> images;
    for (const auto& entry : fs::directory_iterator(directory)) {
        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char value) { return std::tolower(value); });
        if (entry.is_regular_file() &&
            (extension == ".jpg" || extension == ".jpeg" || extension == ".png")) {
            images.push_back(entry.path());
        }
    }
    std::sort(images.begin(), images.end());
    if (limit > 0 && images.size() > static_cast<size_t>(limit)) {
        images.resize(limit);
    }
    if (images.empty()) {
        throw std::runtime_error("没有找到 JPEG 或 PNG 图片: " + directory.string());
    }
    return images;
}

// 公用 COCO 图片编号读取.
inline int64_t ImageIdFromPath(const fs::path& image_path) {
    return std::stoll(image_path.stem().string());
}

// 公用 已完成图片集合读取.
inline std::unordered_set<int64_t> LoadCompleted(const fs::path& path) {
    std::unordered_set<int64_t> completed;
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            completed.insert(std::stoll(line));
        }
    }
    return completed;
}

// 公用 JSON 字符串转义.
inline std::string EscapeJson(const std::string& value) {
    std::ostringstream output;
    for (const char character : value) {
        if (character == '\\' || character == '"') {
            output << '\\';
        }
        output << character;
    }
    return output.str();
}

// 公用 JSONL 到 COCO JSON 数组转换.
inline void FinalizePredictions(const fs::path& jsonl_path, const fs::path& json_path) {
    std::ifstream input(jsonl_path);
    std::ofstream output(json_path);
    output << "[\n";
    std::string line;
    bool first = true;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        if (!first) {
            output << ",\n";
        }
        output << line;
        first = false;
    }
    output << "\n]\n";
}

// 按字节读取非空模型输入.
inline std::vector<char> ReadInput(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("无法读取输入: " + path.string());
    std::vector<char> bytes(std::istreambuf_iterator<char>{stream}, {});
    if (bytes.empty()) throw std::runtime_error("输入文件为空.");
    return bytes;
}

// 从有序非空耗时数组计算线性插值分位数.
inline double Percentile(const std::vector<double>& sorted, double fraction) {
    const double position = (sorted.size() - 1) * fraction;
    const auto lower = static_cast<size_t>(position);
    const auto upper = std::min(lower + 1, sorted.size() - 1);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

}  // namespace mtk
#endif  // MTK_MODELS_TOOLS_BOARD_COMMON_H_
