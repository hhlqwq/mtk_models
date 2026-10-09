"""在编译环境中生成示例清单,不下载数据或运行模型."""

import argparse
import csv
import json
from pathlib import Path


COCO_MODELS = ("yolov5s", "yolov8n", "yoloworld_xl", "fastsam", "rtmpose_body2d")
OTHER_MODELS = ("vit_base_patch16_224", "mobilefacenet",
                "depth_anything_v2_small", "whisper_tiny")


def load_json(path: Path):
    """读取数据集已有标注."""
    return json.loads(path.read_text(encoding="utf-8"))


def coco_samples(dataset: Path, model: str) -> dict:
    """从 COCO 标注读取选定三张图片的类别、地址与许可."""
    annotation = load_json(dataset / "annotations/instances_val2017.json")
    images = {item["id"]: item for item in annotation["images"]}
    licenses = {item["id"]: item for item in annotation["licenses"]}
    categories = sorted(annotation["categories"], key=lambda item: item["id"])
    ids = (210299, 481386, 252219) if model == "rtmpose_body2d" else (139, 285, 785)
    samples = []
    for position, image_id in enumerate(ids, 1):
        image = images[image_id]
        relative = "images/" + image["file_name"]
        if not (dataset / relative).is_file():
            raise FileNotFoundError(dataset / relative)
        samples.append({"file": f"sample_{position}.jpg", "image": relative,
                        "image_id": image_id, "url": image.get("flickr_url"),
                        "license": licenses[image["license"]]})
    return {"source": "https://cocodataset.org/#download", "samples": samples,
            "categories": {str(item["id"]): item["name"] for item in categories},
            "class_names": [item["name"] for item in categories]}


def classification_samples(dataset: Path) -> dict:
    """使用 TorchVision 类别顺序与数据集标签生成 ViT 三图清单."""
    from torchvision.models import ViT_B_16_Weights

    names = ViT_B_16_Weights.IMAGENET1K_V1.meta["categories"]
    labels = (dataset / "val_labels_0based.txt").read_text().splitlines()
    images = "val" if (dataset / "val").is_dir() else "ILSVRC2012_img_val"
    samples = []
    for position in range(1, 4):
        relative = f"{images}/ILSVRC2012_val_{position:08d}.JPEG"
        if not (dataset / relative).is_file():
            raise FileNotFoundError(dataset / relative)
        label = int(labels[position - 1])
        samples.append({"file": f"sample_{position}.JPEG", "image": relative,
                        "label_id": label, "label": names[label]})
    return {"source": "https://image-net.org/challenges/LSVRC/2012/",
            "samples": samples, "class_names": names}


def face_samples(dataset: Path) -> dict:
    """读取选定的两组同人和一组不同人 LFW 配对."""
    with (dataset / "pairs.csv").open(encoding="utf-8", newline="") as stream:
        pairs = {row["pair_id"]: row for row in csv.DictReader(stream)}
    samples = []
    for position, pair_id in enumerate(("0", "300", "1"), 1):
        row = pairs[pair_id]
        for key in ("image_a", "image_b"):
            if not (dataset / "images" / row[key]).is_file():
                raise FileNotFoundError(dataset / "images" / row[key])
        samples.append({"file_a": f"sample_{position}_a.jpg",
                        "file_b": f"sample_{position}_b.jpg",
                        "image_a": "images/" + row["image_a"],
                        "image_b": "images/" + row["image_b"],
                        "pair_id": pair_id, "is_same": int(row["is_same"])})
    return {"source": "https://huggingface.co/datasets/marcelohaps/lfw",
            "samples": samples}


def depth_samples(dataset: Path) -> dict:
    """核对选定室内、街景与玻璃建筑在 DA-2K 中的对应关系."""
    paths = ("images/indoor/1009402064_7c74b0b819_k.jpg",
             "images/outdoor/17943623232_5f974cad30_k.jpg",
             "images/transparent_reflective/49126232263_30c5bf5038_k.jpg")
    annotations = load_json(dataset / "annotations.json")
    samples = []
    for position, relative in enumerate(paths, 1):
        if relative not in annotations or not (dataset / relative).is_file():
            raise ValueError(f"缺少 DA-2K 示例或标注: {relative}.")
        samples.append({"file": f"sample_{position}.jpg", "image": relative,
                        "scene": Path(relative).parts[1]})
    return {"source": "https://github.com/DepthAnything/Depth-Anything-V2/tree/main/DA-2K",
            "samples": samples}


def speech_samples(dataset: Path) -> dict:
    """从 LibriSpeech 的转录文件读取两位说话人的参考文本."""
    samples = []
    for position, sample_id in enumerate(("1089-134686-0000", "1188-133604-0000"), 1):
        matches = list(dataset.rglob(f"{sample_id}.flac"))
        if len(matches) != 1:
            raise ValueError(f"LibriSpeech 示例需唯一存在: {sample_id}.")
        audio = matches[0]
        transcript = audio.parent / ("-".join(sample_id.split("-")[:2]) + ".trans.txt")
        references = dict(line.split(" ", 1) for line in
                          transcript.read_text(encoding="utf-8").splitlines() if line.strip())
        samples.append({"file": f"sample_{position}.wav", "sample_id": sample_id,
                        "audio": audio.relative_to(dataset).as_posix(),
                        "reference": references[sample_id]})
    return {"source": "https://www.openslr.org/12/", "samples": samples}


def main() -> None:
    """根据配置的数据集生成运行清单,仅写入指定构建目录."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=COCO_MODELS + OTHER_MODELS, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"[示例] 从数据集生成 {args.model} 清单.", flush=True)
    adapters = {"vit_base_patch16_224": classification_samples,
                "mobilefacenet": face_samples, "depth_anything_v2_small": depth_samples,
                "whisper_tiny": speech_samples}
    if args.model in COCO_MODELS:
        manifest = coco_samples(args.dataset_root, args.model)
    else:
        manifest = adapters[args.model](args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"[OK] 示例清单已生成: {args.output}.", flush=True)


if __name__ == "__main__":
    main()
