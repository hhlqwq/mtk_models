// 不依赖模型结构与芯片 SDK 的框计算和稳定 Sigmoid.
#ifndef MTK_MODELS_COMMON_CPP_BOXES_H_
#define MTK_MODELS_COMMON_CPP_BOXES_H_

#include <algorithm>
#include <cmath>

namespace mtk {

// 计算 xyxy 坐标框的交并比,退化框面积按零处理.
inline float BoxIou(float ax1, float ay1, float ax2, float ay2, float bx1,
                    float by1, float bx2, float by2) {
    const float intersection =
        std::max(0.0F, std::min(ax2, bx2) - std::max(ax1, bx1)) *
        std::max(0.0F, std::min(ay2, by2) - std::max(ay1, by1));
    const float area_a = std::max(0.0F, ax2 - ax1) * std::max(0.0F, ay2 - ay1);
    const float area_b = std::max(0.0F, bx2 - bx1) * std::max(0.0F, by2 - by1);
    return intersection / std::max(area_a + area_b - intersection, 1.0e-9F);
}

// 接收带 xyxy 字段的模型检测结构,不要求统一模型自身的数据类型.
template <typename Detection>
float BoxIou(const Detection& a, const Detection& b) {
    return BoxIou(a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2);
}

// 限制极端输入,保持现有检测模型的数值范围.
inline float Sigmoid(float value) {
    const float clipped = std::clamp(value, -30.0F, 30.0F);
    return 1.0F / (1.0F + std::exp(-clipped));
}

}  // namespace mtk
#endif  // MTK_MODELS_COMMON_CPP_BOXES_H_
