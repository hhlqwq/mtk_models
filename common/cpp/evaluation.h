// 与 Python 公共工具一致,默认使用固定小子集.
#ifndef MTK_MODELS_COMMON_CPP_EVALUATION_H_
#define MTK_MODELS_COMMON_CPP_EVALUATION_H_

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace mtk {

// 部署入口固定使用 subset,独立全量评价入口传递 full.
inline bool IsTestEvaluation() {
    const char* value = std::getenv("MTK_EVAL_SCOPE");
    const std::string scope = value == nullptr ? "subset" : value;
    if (scope != "subset" && scope != "full") {
        throw std::runtime_error("MTK_EVAL_SCOPE 必须为 subset 或 full.");
    }
    return scope == "subset";
}

// 与主机一致,固定包含已发布效果示例,再按编号补齐100张.
inline std::set<int64_t> TestCocoIds(const std::vector<int64_t>& values) {
    const std::set<int64_t> all(values.begin(), values.end());
    if (all.size() < 100)
        throw std::runtime_error("COCO快速模式至少需要100张图片.");
    std::set<int64_t> selected;
    for (const int64_t id : {139, 285, 785, 210299, 252219, 481386}) {
        if (all.count(id)) selected.insert(id);
    }
    for (const int64_t id : all) {
        if (selected.size() == 100) break;
        selected.insert(id);
    }
    return selected;
}

// Whisper固定5条音频,保留两位说话人的效果示例.
inline std::set<std::string> TestSpeechIds(
    const std::vector<std::string>& values) {
    const std::set<std::string> all(values.begin(), values.end());
    if (all.size() < 5)
        throw std::runtime_error("语音快速模式至少需要5条音频.");
    std::set<std::string> selected;
    for (const std::string id : {"1089-134686-0000", "1188-133604-0000"}) {
        if (all.count(id)) selected.insert(id);
    }
    for (const auto& id : all) {
        if (selected.size() == 5) break;
        selected.insert(id);
    }
    return selected;
}

}  // namespace mtk
#endif  // MTK_MODELS_COMMON_CPP_EVALUATION_H_
