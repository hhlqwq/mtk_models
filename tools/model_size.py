"""只读取模型文件大小,以 MB 补充既有结果,不运行模型或评测."""

import argparse
from datetime import date
import json
from pathlib import Path

from tools.board import configure_parser


def collect_model_size(models_dir: Path, includes: list[str] | None = None,
                       source: str = "board_file_stat") -> dict:
    """统计明确选定的模型文件,排除历史目录与评测输入输出.

    Args:
        models_dir: 当前部署模型目录.
        includes: 相对目录的文件模式; 每个显式模式必须匹配非空文件.
        source: 文件采集位置的说明,区分板端与交付包.

    Returns:
        两位小数的 MB 汇总、文件明细、采集日期与范围.
    """
    root = models_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"模型目录无效: {root}")
    patterns = includes or ["*.dla", "embedding*.bin", "*.onnx",
                            "*int8*.tflite", "*w8*.tflite", "*.pt", "*.pth",
                            "*.safetensors", "*.h5"]
    files = set()
    for pattern in patterns:
        matches = [path for path in root.glob(pattern) if path.is_file()]
        if includes and not matches:
            raise FileNotFoundError(f"缺少声明的模型文件: {root}/{pattern}")
        for path in matches:
            path.resolve(strict=True).relative_to(root)
            if path.stat().st_size <= 0:
                raise ValueError(f"模型文件为空: {path}")
            files.add(path)
    totals = {key: 0 for key in ("source_weights", "fp32_onnx",
                                "quantized_tflite", "compiled_dla",
                                "auxiliary_weights")}
    records = []
    for path in sorted(files):
        if path.suffix == ".dla":
            role = "compiled_dla"
        elif path.suffix == ".bin" and path.name.startswith("embedding"):
            role = "auxiliary_weights"
        elif path.suffix == ".onnx":
            role = "fp32_onnx"
        elif path.suffix == ".tflite" and any(
                name in path.name for name in ("int8", "w8")):
            role = "quantized_tflite"
        elif path.suffix in (".pt", ".pth", ".safetensors", ".h5"):
            role = "source_weights"
        else:
            raise ValueError(f"模型文件角色不明确: {path}")
        size = path.stat().st_size
        totals[role] += size
        records.append({"file": path.relative_to(root).as_posix(),
                        "role": role, "size_mb": round(size / 1_000_000, 2)})
    if not totals["compiled_dla"]:
        raise FileNotFoundError("未找到实际部署的 DLA,不写入空的模型大小记录.")
    return {
        "unit": "MB",
        **{key: round(value / 1_000_000, 2) if value else None
           for key, value in totals.items()},
        "deployment_total": round((totals["compiled_dla"] +
                                   totals["auxiliary_weights"]) / 1_000_000, 2),
        "files": records, "source": source, "directory": str(root),
        "collected_on": date.today().isoformat(),
        "scope": "current_deployment_files_not_reverified_original_run",
    }


def main() -> None:
    """只补充大小字段,保留已有结果中的精度、性能、状态和运行编号."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--include", action="append")
    parser.add_argument("--source", default="board_file_stat")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--run-id")
    configure_parser(parser, __file__, {
        "include": lambda c: ["model_" + c.config["conversion"]["precision"] + ".dla"]
                   if c.config["name"] == "yamnet" else None,
        "models_dir": lambda c: c.root / "models",
        "summary": lambda c: c.result / ("board_summary.json" if c.config["name"] == "deepseek_r1_distill_qwen_1_5b" else "summary.json"),
    })
    args = parser.parse_args()
    size = collect_model_size(args.models_dir, args.include, args.source)
    if args.summary is None:
        print(json.dumps(size, ensure_ascii=False))
        return
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    summary["model_size"] = size
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(f"[大小记录] {size['deployment_total']:.2f} MB -> {args.summary}", flush=True)


if __name__ == "__main__":
    main()
