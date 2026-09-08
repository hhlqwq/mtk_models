"""生成与常用 ImageNet 模型输出顺序一致的 ILSVRC2012 验证标签."""

import argparse
import collections
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from scipy.io import loadmat


DEVKIT_URL = (
    "https://image-net.org/data/ILSVRC/2012/"
    "ILSVRC2012_devkit_t12.tar.gz"
)
CLASS_INDEX_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "imagenet_class_index.json"
)
EXPECTED_DEVKIT_SHA256 = (
    "b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953"
)
EXPECTED_CLASS_INDEX_SHA256 = (
    "a1e7a966a1f601d39e4b43e119b3e7dd4a2ad3ea08cf69847cbaf021013767bc"
)
EXPECTED_IMAGES = 50000
EXPECTED_CLASSES = 1000
EXPECTED_IMAGES_PER_CLASS = 50


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """计算内存数据 SHA-256."""
    return hashlib.sha256(data).hexdigest()


def read_unique_member(archive: tarfile.TarFile, suffix: str) -> bytes:
    """从 tar 归档读取唯一匹配成员, 不向文件系统解压."""
    members = [member for member in archive.getmembers()
               if member.isfile() and member.name.endswith(suffix)]
    if len(members) != 1:
        raise ValueError(
            f"归档成员数量错误: suffix={suffix}, count={len(members)}")
    stream = archive.extractfile(members[0])
    if stream is None:
        raise ValueError(f"无法读取归档成员: {members[0].name}")
    return stream.read()


def load_devkit(devkit_path: Path) -> tuple[list[int], dict[int, str], dict]:
    """读取官方 ground truth 和 ILSVRC2012_ID 到 WNID 的映射."""
    with tarfile.open(devkit_path, "r:gz") as archive:
        ground_truth_bytes = read_unique_member(
            archive, "/data/ILSVRC2012_validation_ground_truth.txt")
        meta_bytes = read_unique_member(archive, "/data/meta.mat")
    ground_truth = [int(value) for value in
                    ground_truth_bytes.decode("utf-8").splitlines()
                    if value.strip()]
    synsets = loadmat(
        io.BytesIO(meta_bytes), squeeze_me=True,
        struct_as_record=False)["synsets"]
    id_to_wnid = {}
    for synset in synsets:
        class_id = int(synset.ILSVRC2012_ID)
        wnid = str(synset.WNID)
        if 1 <= class_id <= EXPECTED_CLASSES:
            if class_id in id_to_wnid:
                raise ValueError(f"ILSVRC2012_ID 重复: {class_id}")
            id_to_wnid[class_id] = wnid
    hashes = {
        "ground_truth_sha256": sha256_bytes(ground_truth_bytes),
        "meta_mat_sha256": sha256_bytes(meta_bytes),
    }
    return ground_truth, id_to_wnid, hashes


def load_class_index(path: Path) -> tuple[dict[str, int], list[str]]:
    """读取 Keras/TensorFlow 输出索引并返回 WNID 反向映射."""
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {str(index) for index in range(EXPECTED_CLASSES)}
    if set(raw) != expected_keys:
        raise ValueError("类别索引必须完整包含字符串键 0 到 999.")
    wnid_to_index = {}
    labels = []
    for index in range(EXPECTED_CLASSES):
        wnid, label = raw[str(index)]
        if wnid in wnid_to_index:
            raise ValueError(f"类别索引 WNID 重复: {wnid}")
        wnid_to_index[str(wnid)] = index
        labels.append(str(label))
    return wnid_to_index, labels


def normalize_label(value: str) -> str:
    """移除类别名称中的空格和符号, 用于辅助核对显示名称."""
    return "".join(character for character in value.casefold()
                   if character.isalnum())


def verify_display_labels(path: Path, reference: list[str]) -> list[dict]:
    """核对 Qualcomm 显示标签与 Keras 类别名称顺序."""
    labels = [line.strip() for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != EXPECTED_CLASSES:
        raise ValueError(
            f"Qualcomm 标签数量错误: expected=1000, actual={len(labels)}")
    mismatches = []
    for index, (actual, expected) in enumerate(zip(labels, reference)):
        if normalize_label(actual) != normalize_label(expected):
            mismatches.append({
                "index": index,
                "qualcomm": actual,
                "class_index": expected,
            })
    return mismatches


def build_labels(ground_truth: list[int], id_to_wnid: dict[int, str],
                 wnid_to_index: dict[str, int]) -> list[int]:
    """将官方 1-based 类别 ID 转换为模型 0-based 输出索引."""
    if len(ground_truth) != EXPECTED_IMAGES:
        raise ValueError(
            f"ground truth 数量错误: expected=50000, "
            f"actual={len(ground_truth)}")
    if len(id_to_wnid) != EXPECTED_CLASSES:
        raise ValueError(
            f"devkit 类别映射数量错误: expected=1000, "
            f"actual={len(id_to_wnid)}")
    labels = []
    for image_index, class_id in enumerate(ground_truth, start=1):
        if class_id not in id_to_wnid:
            raise ValueError(
                f"未知 ILSVRC2012_ID: image={image_index}, id={class_id}")
        wnid = id_to_wnid[class_id]
        if wnid not in wnid_to_index:
            raise ValueError(
                f"WNID 不在模型类别索引中: image={image_index}, wnid={wnid}")
        labels.append(wnid_to_index[wnid])
    histogram = collections.Counter(labels)
    if set(histogram) != set(range(EXPECTED_CLASSES)):
        raise ValueError("生成标签没有完整覆盖 0 到 999.")
    invalid_counts = {key: value for key, value in histogram.items()
                      if value != EXPECTED_IMAGES_PER_CLASS}
    if invalid_counts:
        raise ValueError(f"类别样本数不是每类 50 张: {invalid_counts}")
    return labels


def run(args: argparse.Namespace) -> None:
    """执行来源校验、类别映射、输出生成和清单记录."""
    print("[1/4] 校验官方资源 SHA-256.")
    devkit_sha256 = sha256_file(args.devkit)
    class_index_sha256 = sha256_file(args.class_index)
    if devkit_sha256 != EXPECTED_DEVKIT_SHA256:
        raise ValueError(f"devkit SHA-256 不匹配: {devkit_sha256}")
    if class_index_sha256 != EXPECTED_CLASS_INDEX_SHA256:
        raise ValueError(
            f"class index SHA-256 不匹配: {class_index_sha256}")

    print("[2/4] 读取 devkit 与模型类别索引.")
    ground_truth, id_to_wnid, member_hashes = load_devkit(args.devkit)
    wnid_to_index, class_names = load_class_index(args.class_index)
    mismatches = verify_display_labels(args.qualcomm_labels, class_names)

    print("[3/4] 生成并验证 50,000 行 0-based 标签.")
    labels = build_labels(ground_truth, id_to_wnid, wnid_to_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{label}\n" for label in labels), encoding="utf-8")

    print("[4/4] 写入可追溯清单.")
    manifest = {
        "schema_version": 1,
        "sources": {
            "devkit_url": DEVKIT_URL,
            "devkit_sha256": devkit_sha256,
            "class_index_url": CLASS_INDEX_URL,
            "class_index_sha256": class_index_sha256,
            "qualcomm_labels_sha256": sha256_file(args.qualcomm_labels),
            **member_hashes,
        },
        "mapping": "ILSVRC2012_ID -> WNID -> model_output_index_0based",
        "images": len(labels),
        "classes": len(set(labels)),
        "images_per_class": EXPECTED_IMAGES_PER_CLASS,
        "display_label_mismatch_count": len(mismatches),
        "display_label_mismatches": mismatches,
        "output_sha256": sha256_file(args.output),
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"[OK] 标签: {args.output}")
    print(f"[OK] 清单: {args.manifest}")
    print(f"[INFO] 显示名称差异: {len(mismatches)}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devkit", type=Path, required=True)
    parser.add_argument("--class-index", type=Path, required=True)
    parser.add_argument("--qualcomm-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
