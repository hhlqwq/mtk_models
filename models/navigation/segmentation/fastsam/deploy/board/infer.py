"""在开发板执行 FastSAM 推理并保存逐图检查点,不分派评测或汇总任务."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import cv2
import numpy
import pycocotools.mask

from tools.process import run_command

from tools.files import require_file

from tools.evaluation import coco_ids

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def infer_image(root: Path, image: Path, output: Path, protocol: dict,
                *, quiet: bool = False) -> None:
    """调用 C++ 程序完成单图预处理、NPU 推理与分割后处理."""
    require_file(image)
    command = [
        root / "board/fastsam_board", "--model", root / "models/model_int8.dla",
        "--config", root / "models/runtime_config.csv", "--image", image,
        "--output-dir", output, "--confidence", protocol["confidence"],
        "--iou", protocol["nms_iou"], "--max-det", protocol["max_detections"],
    ]
    if quiet:
        command.append("--quiet")
    run_command(command, quiet=quiet)


def single(args: argparse.Namespace, paths: dict, protocol: dict) -> None:
    """执行一张图片并保留掩码、叠加图及原始结果,不计算全量指标."""
    image = paths["dataset"] / args.image
    infer_image(paths["root"], image, paths["result"] / "single", protocol)


def infer(args: argparse.Namespace, paths: dict, protocol: dict) -> None:
    """逐图执行全量 NPU 推理并保存可续跑检查点,每 50 张提示进度."""
    images = sorted((paths["dataset"] / "images").glob("*.jpg"))
    if len(images) != protocol["images"]:
        raise ValueError(f"数据集需要 {protocol['images']} 张图片,实际 {len(images)}.")
    selected = set(coco_ids(int(image.stem) for image in images))
    images = [image for image in images if int(image.stem) in selected]
    require_file(paths["annotations"])
    # 提前检查编码依赖,避免推理后才发现无法保存 RLE 检查点.
    work = paths["work"]
    if args.resume:
        if not work.is_dir():
            raise FileNotFoundError("续跑需要已有 work 目录及相同的 --run-id.")
    elif work.exists():
        raise FileExistsError("本次 work 已存在,请换运行编号或显式使用 --resume.")
    for name in ("raw", "predictions_by_image", "report"):
        (work / name).mkdir(parents=True, exist_ok=True)
    print(f"[推理] FastSAM 共 {len(images)} 张图片.", flush=True)
    for count, image in enumerate(images, 1):
        checkpoint = work / "predictions_by_image" / f"{image.stem}.json"
        if not checkpoint.exists():
            raw = work / "raw" / image.stem
            if raw.exists():
                raise FileExistsError(f"残留单图中间目录: {raw}.请保留现场并换运行编号.")
            try:
                infer_image(paths["root"], image, raw, protocol, quiet=True)
                run_command([sys.executable, paths["root"] / "board/full_accuracy_board.py",
                             "--mode", "encode-one", "--image", image,
                             "--images", paths["dataset"] / "images",
                             "--annotations", paths["annotations"], "--work-dir", work,
                             "--run-id", args.run_id], quiet=True)
            except (subprocess.CalledProcessError, OSError):
                print(f"[失败图片] {image}", file=sys.stderr, flush=True)
                raise
            # 成功保存 RLE 后仅清理本次单图中间文件,保留检查点.
            if raw.is_symlink() or raw.resolve().parent != (work / "raw").resolve():
                raise ValueError(f"中间目录超出本次运行范围: {raw}.")
            shutil.rmtree(raw)
        elif not args.resume:
            raise FileExistsError(f"非续跑模式发现检查点: {checkpoint}.")
        if count % 50 == 0 or count == len(images):
            print(f"[进度] FastSAM {count}/{len(images)} 张.", flush=True)
    print(f"[完成] FastSAM {len(images)}/{len(images)} 张.", flush=True)


def main() -> None:
    """直接读取板端配置,执行单图或全量 NPU 推理."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "config.json")
    parser.add_argument("--run-id", required=True, help="本次运行的独立编号.")
    parser.add_argument("--resume", action="store_true", default=os.environ.get("EVAL_RESUME", "0") == "1",
                        help="续跑已有全量推理检查点.")
    parser.add_argument("--image", help="只推理指定图片,相对路径以板端数据集为基准.")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("运行编号只能包含字母、数字、下划线和连字符.")
    if args.image and args.resume:
        parser.error("单图推理不使用 --resume,重新执行请更换运行编号.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    dataset = Path(config["board"]["dataset_dir"])
    result = root / config["board"]["results_dir"] / args.run_id
    paths = {"root": root, "dataset": dataset, "result": result,
             "work": result / "work", "annotations": dataset / "annotations/instances_val2017.json"}
    protocol = json.loads((root / "board/accuracy_protocol.json").read_text(encoding="utf-8"))
    for name in ("models/model_int8.dla", "models/runtime_config.csv", "board/fastsam_board"):
        require_file(root / name)
    print(f"[推理] 运行编号: {args.run_id}; 结果目录: {result}", flush=True)
    if args.image:
        single(args, paths, protocol)
    else:
        infer(args, paths, protocol)


if __name__ == "__main__":
    main()
