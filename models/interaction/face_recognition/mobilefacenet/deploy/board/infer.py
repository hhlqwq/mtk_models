"""调度模型 C++ 程序完成预处理、SDK 推理和后处理,不计算精度."""

import json

from full_accuracy_board import load_pairs
from tools.board import context, infer_images


def main():
    """使用统一 JSON 配置,快速与全量结果保存到同一模型的 results 下."""
    ctx = context(__file__, new_run=True)
    rows = load_pairs(ctx.dataset)
    names = sorted({row[field] for row in rows for field in ("image_a", "image_b")})
    images = [(name, ctx.dataset / "images" / name) for name in names]
    metadata = json.loads((ctx.root / "models/metadata.json").read_text(encoding="utf-8"))
    if metadata.get("input_protocol") != "RGB_ImageNet_normalize_aligned_112x112":
        raise ValueError("量化元数据未声明已对齐人脸的 RGB ImageNet 输入协议.")
    infer_images(ctx, "mobilefacenet_board", images, metadata, [1, 3, 112, 112], [1, 128])


if __name__ == "__main__":
    main()
