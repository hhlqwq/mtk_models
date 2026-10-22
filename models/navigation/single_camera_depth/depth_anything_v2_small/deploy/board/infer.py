"""调度模型 C++ 程序完成预处理、SDK 推理和后处理,不计算精度."""

import json

from full_accuracy_board import load_protocol
from tools.board import context, infer_images


def main():
    """使用统一 JSON 配置,快速与全量结果保存到同一模型的 results 下."""
    ctx = context(__file__, new_run=True)
    names = sorted(load_protocol(ctx.dataset))
    images = [(name, ctx.dataset / name) for name in names]
    metadata = json.loads((ctx.root / "models/model_int8.json").read_text(encoding="utf-8"))
    infer_images(ctx, "depth_anything_board", images, metadata, [1, 3, 518, 518], [1, 518, 518])


if __name__ == "__main__":
    main()
