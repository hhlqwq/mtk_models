// 验证板端公共算法与文件工具,不依赖 SDK 或模型.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tools/board/detection.h"

namespace fs = std::filesystem;

struct Detection {
    float x1, y1, x2, y2, score;
    int class_id;
};
struct Geometry {
    int left, top;
    float scale;
    int original_width, original_height;
};
struct Timing {
    double preprocess_ms, npu_ms, postprocess_ms, e2e_ms;
};

// 不依赖 assert 或编译优化选项,验证失败即返回错误.
void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// 在指定临时目录测试数值边界、类别策略和结果文件.
int main(int argc, char** argv) {
    try {
        Require(argc == 2, "需要测试输出目录.");
        const fs::path work = argv[1];
        fs::create_directories(work);
        const Detection a{0, 0, 2, 2, 0.9F, 0};
        const Detection b{1, 0, 3, 2, 0.8F, 0};
        Require(std::abs(mtk::BoxIou(a, b) - 1.0F / 3) < 1.0e-6F, "IoU错误.");
        Require(mtk::BoxIou(0, 0, 0, 0, 0, 0, 0, 0) == 0, "退化框错误.");
        Require(mtk::Sigmoid(0) == 0.5F && std::isfinite(mtk::Sigmoid(1000)), "Sigmoid错误.");
        const std::vector<Detection> input{a, b, Detection{0, 0, 2, 2, 0.7F, 1}};
        for (const bool stable : {false, true}) {
            const auto kept = mtk::ApplyNms(input, 0.2F, 100, stable);
            Require(kept.size() == 2 && kept[1].class_id == 1, "类别内NMS错误.");
            Require(mtk::ApplyNms(input, 0.2F, 1, stable).size() == 1, "max_det错误.");
        }
        std::vector<Detection> boxes{Detection{2, 4, 22, 24, 1, 0}};
        mtk::Rescale(&boxes, Geometry{2, 4, 2.0F, 8, 9});
        Require(boxes[0].x1 == 0 && boxes[0].x2 == 8 && boxes[0].y2 == 9, "坐标还原错误.");
        Require(mtk::Percentile({1, 3, 5}, 0.5) == 3, "分位数错误.");
        Require(mtk::Percentile({7}, 0.95) == 7, "单元素分位数错误.");
        Require(mtk::ParseInteger("3", "count") == 3, "整数解析错误.");
        Require(mtk::ParseFloat("0.25", "iou") == 0.25F, "浮点解析错误.");
        bool rejected = false;
        try {
            mtk::CheckRuntime(5, "fixture");
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        Require(rejected, "运行时错误未抛出.");
        mtk::CheckRuntime(0, "fixture");
        std::ofstream(work / "input.bin", std::ios::binary) << "abc";
        Require(mtk::ReadInput(work / "input.bin").size() == 3, "二进制读取错误.");
        std::ofstream(work / "done.txt") << "139\n285\n139\n";
        Require(mtk::LoadCompleted(work / "done.txt").size() == 2, "完成集合错误.");
        fs::create_directories(work / "images");
        std::ofstream(work / "images/000139.JPG") << "fixture";
        std::ofstream(work / "images/000285.png") << "fixture";
        const auto images = mtk::ListImages(work / "images", 1);
        Require(images.size() == 1 && mtk::ImageIdFromPath(images[0]) == 139,
                "图片排序或限制错误.");
        const std::vector<int> categories{1, 3};
        std::ofstream predictions(work / "predictions.jsonl");
        mtk::AppendPredictions(&predictions, 139, input, categories);
        predictions.close();
        mtk::FinalizePredictions(work / "predictions.jsonl", work / "predictions.json");
        const auto prediction_bytes = mtk::ReadInput(work / "predictions.json");
        const std::string prediction_text(prediction_bytes.begin(), prediction_bytes.end());
        Require(prediction_text.find("\"category_id\":3") != std::string::npos, "类别映射错误.");
        Require(mtk::EscapeJson("a\"b\\c") == "a\\\"b\\\\c", "JSON转义错误.");
        mtk::WriteSummary(work / "summary.json", std::vector<Timing>{{1, 2, 3, 6}, {3, 4, 5, 12}},
                          2);
        Require(fs::file_size(work / "summary.json") > 0, "耗时汇总为空.");
        std::cout << "[通过] C++公共工具数值、文件与类别策略验证." << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
