"""读取 FastSAM 板端配置,复用检查点生成三张效果图."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main() -> None:
    """确定本次结果目录并调用已有绘图工具,不执行模型推理."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config.json")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("运行编号只能包含字母、数字、下划线和连字符.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = root / config["board"]["results_dir"] / args.run_id
    output = result / "examples/output"
    if os.environ.get("EVAL_RESUME", "0") == "1" and output.is_dir():
        for index in range(1, 4):
            path = output / f"sample_{index}_segmentation.jpg"
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError(f"续跑示例不完整: {path}.")
        print(f"[示例] 复用本次已保存的三张效果图: {output}", flush=True)
        return
    print(f"[示例] 从本次检查点绘图: {output}", flush=True)
    subprocess.run([sys.executable, str(root / "board/render_examples.py"), "--model", "fastsam",
                    "--input-dir", str(root / "examples/input"), "--output-dir", str(output),
                    "--work-dir", str(result / "work"), "--dataset-root", config["board"]["dataset_dir"],
                    "--models-dir", str(root / "models")], check=True)


if __name__ == "__main__":
    main()
