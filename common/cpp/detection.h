// COCO 检测的类别内 NMS、坐标还原与结果记录.
#ifndef MTK_MODELS_COMMON_CPP_DETECTION_H_
#define MTK_MODELS_COMMON_CPP_DETECTION_H_
#include <array>
#include <cmath>
#include <iomanip>
#include <numeric>

#include "boxes.h"
#include "board.h"
namespace mtk {
// 读取当前 Linux 推理进程的峰值 RSS,单位为 KiB.
inline int64_t GetPeakRssKb() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0)
        throw std::runtime_error("无法读取峰值 RSS.");
    return usage.ru_maxrss;
}

// 公用 类别内非极大值抑制,保留模型的排序选择.
template <typename Detection>
std::vector<Detection> ApplyNms(std::vector<Detection> detections,
                                float iou_threshold, int max_detections,
                                bool stable_sort) {
    const auto compare = [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    };
    if (stable_sort) {
        std::stable_sort(detections.begin(), detections.end(), compare);
    } else {
        std::sort(detections.begin(), detections.end(), compare);
    }
    std::vector<Detection> kept;
    kept.reserve(std::min<size_t>(detections.size(), max_detections));
    for (const Detection& candidate : detections) {
        bool suppressed = false;
        for (const Detection& selected : kept) {
            if (candidate.class_id == selected.class_id &&
                BoxIou(candidate, selected) > iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(candidate);
            if (kept.size() == static_cast<size_t>(max_detections)) {
                break;
            }
        }
    }
    return kept;
}

// 公用 letterbox 坐标反变换.
template <typename Detection, typename LetterboxInfo>
void Rescale(std::vector<Detection>* detections, const LetterboxInfo& info) {
    for (Detection& detection : *detections) {
        detection.x1 = std::clamp((detection.x1 - info.left) / info.scale, 0.0F,
                                  static_cast<float>(info.original_width));
        detection.x2 = std::clamp((detection.x2 - info.left) / info.scale, 0.0F,
                                  static_cast<float>(info.original_width));
        detection.y1 = std::clamp((detection.y1 - info.top) / info.scale, 0.0F,
                                  static_cast<float>(info.original_height));
        detection.y2 = std::clamp((detection.y2 - info.top) / info.scale, 0.0F,
                                  static_cast<float>(info.original_height));
    }
}

// 公用 单图 COCO 预测记录.
template <typename Detection, typename Categories>
void AppendPredictions(std::ofstream* output, int64_t image_id,
                       const std::vector<Detection>& detections,
                       const Categories& categories) {
    *output << std::fixed;
    for (const Detection& detection : detections) {
        *output << "{\"image_id\":" << image_id
                << ",\"category_id\":" << categories[detection.class_id]
                << ",\"bbox\":[" << std::setprecision(3) << detection.x1 << ','
                << detection.y1 << ',' << detection.x2 - detection.x1 << ','
                << detection.y2 - detection.y1
                << "],\"score\":" << std::setprecision(5) << detection.score
                << "}\n";
    }
    output->flush();
}

// 公用 各阶段推理耗时汇总.
template <typename Timing>
void WriteSummary(const fs::path& path, const std::vector<Timing>& timings,
                  size_t total_images) {
    auto summarize = [&timings](auto accessor) {
        std::vector<double> values;
        values.reserve(timings.size());
        for (const Timing& timing : timings) {
            values.push_back(accessor(timing));
        }
        std::sort(values.begin(), values.end());
        const auto percentile = [&values](double ratio) {
            const double position = ratio * (values.size() - 1);
            const size_t lower = static_cast<size_t>(std::floor(position));
            const size_t upper = static_cast<size_t>(std::ceil(position));
            const double weight = position - lower;
            return values[lower] * (1.0 - weight) + values[upper] * weight;
        };
        const double mean =
            std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        return std::array<double, 6>{
            mean,           percentile(0.5), percentile(0.9), percentile(0.95),
            values.front(), values.back()};
    };
    const auto preprocess =
        summarize([](const Timing& value) { return value.preprocess_ms; });
    const auto npu =
        summarize([](const Timing& value) { return value.npu_ms; });
    const auto postprocess =
        summarize([](const Timing& value) { return value.postprocess_ms; });
    const auto e2e =
        summarize([](const Timing& value) { return value.e2e_ms; });
    const auto write_metric = [](std::ofstream* output, const std::string& name,
                                 const std::array<double, 6>& values,
                                 bool trailing_comma) {
        *output << "  \"" << EscapeJson(name) << "\": {\"mean\": " << values[0]
                << ", \"p50\": " << values[1] << ", \"p90\": " << values[2]
                << ", \"p95\": " << values[3] << ", \"min\": " << values[4]
                << ", \"max\": " << values[5] << "}"
                << (trailing_comma ? "," : "") << "\n";
    };

    std::ofstream output(path);
    output << std::fixed << std::setprecision(6);
    const int64_t peak_rss_kb = GetPeakRssKb();
    output << "{\n  \"processed_images\": " << total_images << ",\n";
    output << "  \"peak_rss_kb\": " << peak_rss_kb << ",\n";
    output << "  \"peak_rss_bytes\": " << peak_rss_kb * 1024 << ",\n";
    write_metric(&output, "preprocess_ms", preprocess, true);
    write_metric(&output, "npu_ms", npu, true);
    write_metric(&output, "postprocess_ms", postprocess, true);
    write_metric(&output, "e2e_ms", e2e, false);
    output << "}\n";
}

}  // namespace mtk
#endif  // MTK_MODELS_COMMON_CPP_DETECTION_H_
