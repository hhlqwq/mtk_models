// 在板端将官方人体检测 JSON 转为 RTMPose C++ 评测清单.

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace fs = std::filesystem;

struct Detection {
  int image_id;
  size_t detection_id;
  double score;
  double x;
  double y;
  double width;
  double height;
};

// 解析命令行,只接受明确的输入和输出路径.
std::pair<fs::path, fs::path> ParseArgs(int argc, char** argv) {
  fs::path input;
  fs::path output;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      throw std::invalid_argument("参数缺少值.");
    }
    const std::string key = argv[index];
    if (key == "--detections") {
      input = argv[index + 1];
    } else if (key == "--output") {
      output = argv[index + 1];
    } else {
      throw std::invalid_argument("未知参数: " + key);
    }
  }
  if (input.empty() || output.empty()) {
    throw std::invalid_argument("需要 --detections 和 --output.");
  }
  return {input, output};
}

// 读取并校验全部官方检测框,保留原始数组序号.
std::vector<Detection> LoadDetections(const fs::path& input) {
  cv::FileStorage storage(input.string(), cv::FileStorage::READ |
                                       cv::FileStorage::FORMAT_JSON);
  if (!storage.isOpened()) {
    throw std::runtime_error("无法打开检测 JSON: " + input.string());
  }
  const cv::FileNode root = storage.root();
  if (!root.isSeq()) {
    throw std::runtime_error("检测 JSON 顶层必须是数组.");
  }
  std::vector<Detection> records;
  records.reserve(root.size());
  size_t index = 0;
  for (const cv::FileNode& item : root) {
    const cv::FileNode bbox = item["bbox"];
    if (item["category_id"].empty() || item["image_id"].empty() ||
        item["score"].empty() || !bbox.isSeq() || bbox.size() != 4) {
      throw std::runtime_error("检测框字段不完整,index=" +
                               std::to_string(index));
    }
    auto coordinate = bbox.begin();
    Detection record{static_cast<int>(item["image_id"]), index,
                     static_cast<double>(item["score"]),
                     static_cast<double>(*coordinate++),
                     static_cast<double>(*coordinate++),
                     static_cast<double>(*coordinate++),
                     static_cast<double>(*coordinate++)};
    if (static_cast<int>(item["category_id"]) != 1 ||
        record.width <= 0 || record.height <= 0) {
      throw std::runtime_error("无效人体检测框,index=" +
                               std::to_string(index));
    }
    records.push_back(record);
    ++index;
    if (index % 10000 == 0 || index == root.size()) {
      std::cout << "[MANIFEST] " << index << "/" << root.size() << '\n';
    }
  }
  std::sort(records.begin(), records.end(),
            [](const Detection& left, const Detection& right) {
              return std::tie(left.image_id, left.detection_id) <
                     std::tie(right.image_id, right.detection_id);
            });
  return records;
}

// 写入与现有 RTMPose 推理程序兼容的 TSV 清单.
void WriteManifest(const fs::path& output,
                   const std::vector<Detection>& records) {
  fs::create_directories(output.parent_path());
  std::ofstream stream(output);
  if (!stream) {
    throw std::runtime_error("无法写入清单: " + output.string());
  }
  stream << "image_id\tdetection_id\tbbox_score\tx\ty\tw\th\n";
  stream << std::setprecision(17);
  for (const Detection& item : records) {
    stream << item.image_id << '\t' << item.detection_id << '\t'
           << item.score << '\t' << item.x << '\t' << item.y << '\t'
           << item.width << '\t' << item.height << '\n';
  }
  if (!stream) {
    throw std::runtime_error("写入检测清单失败.");
  }
}

// 执行转换并以非零退出码报告异常.
int main(int argc, char** argv) {
  try {
    const auto [input, output] = ParseArgs(argc, argv);
    const std::vector<Detection> records = LoadDetections(input);
    WriteManifest(output, records);
    std::cout << "[OK] 人体检测框清单: " << output << ",共 "
              << records.size() << " 个框.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << '\n';
    return 1;
  }
}
