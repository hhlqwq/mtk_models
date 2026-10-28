"""全量与快速流程验证的固定子集规则,供主机和板端共同使用."""

from collections import defaultdict
import hashlib
import json
import math
import os


LIMITS = {"fastsam": 300, "yolov5s": 300, "yolov8n": 300,
          "resnet101": 1000, "rtmpose_body2d": 500, "vit_base_patch16_224": 1000,
          "mobilefacenet": 500, "depth_anything_v2_small": 100,
          "yamnet": 250, "whisper_tiny": 50, "mit_bevfusion": 100}
COCO_EXAMPLES = (139, 285, 785, 210299, 252219, 481386)
IMAGENET_EXAMPLES = tuple(f"ILSVRC2012_val_{index:08d}.JPEG" for index in range(1, 6))
SPEECH_EXAMPLES = ("1089-134686-0000", "1188-133604-0000")


def is_test():
    """默认选择固定小子集,全量范围由独立评价入口传递."""
    scope = os.environ.get("MTK_EVAL_SCOPE", "subset")
    if scope not in ("subset", "full"):
        raise ValueError("MTK_EVAL_SCOPE 必须为 subset 或 full.")
    return scope == "subset"


def result_status(status="complete"):
    """仅将成功的快速结果标为 smoke_test,不掩盖 incomplete 或失败."""
    return "smoke_test" if is_test() and status in ("complete", "board_verified") else status


def scope_metadata():
    """记录评测范围与已选择样本的指纹,用于校验参考基准."""
    result = {"evaluation_scope": "subset" if is_test() else "full"}
    if is_test():
        result["purpose"] = "固定子集任务精度检查,不代表全量精度"
        digest = os.environ.get("MTK_EVAL_SAMPLE_SHA256")
        if digest:
            result["sample_ids_sha256"] = digest
    return result


def record_selection(values):
    """按排序后的唯一标识记录实际样本集合,便于两端比对."""
    identifiers = sorted(str(value) for value in values)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("评测子集必须非空且没有重复标识.")
    digest = hashlib.sha256(json.dumps(identifiers, ensure_ascii=False).encode()).hexdigest()
    os.environ["MTK_EVAL_SAMPLE_SHA256"] = digest
    return digest


def fixed_subset(items, count, *, key=lambda value: value, required=()):
    """保留必要示例,其余名额在完整有序样本中等间隔抽取."""
    items = list(items)
    if len(items) < count:
        raise ValueError(f"快速模式需要至少 {count} 个样本,实际 {len(items)}.")
    mapping = {key(item): item for item in items}
    if len(mapping) != len(items):
        raise ValueError("样本标识重复.")
    wanted = {value for value in required if value in mapping}
    if len(wanted) > count:
        raise ValueError("固定示例数超过快速评测样本数.")
    remaining = [value for value in sorted(mapping) if value not in wanted]
    slots = count - len(wanted)
    step = max(1, len(remaining) // slots) if slots else 1
    for index in range(slots):
        wanted.add(remaining[index * step])
    selected = [mapping[value] for value in sorted(wanted)]
    record_selection(key(item) for item in selected)
    return selected


def coco_ids(ids):
    """检测和分割固定 300 张,保留 README 示例并覆盖整个验证集."""
    values = sorted(int(value) for value in ids)
    if not is_test():
        record_selection(values)
        return values
    return fixed_subset(values, 300, required=COCO_EXAMPLES)


def pose_image_ids(detections):
    """按整图保留全部人体框,均匀取图直到至少覆盖 500 个人体实例."""
    groups = defaultdict(int)
    for row in detections:
        groups[int(row["image_id"])] += 1
    keys = sorted(groups)
    if not is_test():
        record_selection(keys)
        return keys
    if not keys:
        raise ValueError("姿态检测框列表为空.")
    target = LIMITS["rtmpose_body2d"]
    selected = set(COCO_EXAMPLES) & set(keys)
    total = sum(groups[key] for key in selected)
    remaining = [key for key in keys if key not in selected]
    average = len(detections) / len(keys)
    images = max(1, math.ceil(max(0, target - total) / average))
    step = max(1, len(remaining) // images)
    # 先跨整个划分均匀取图,不足时补齐;始终保留每张图的全部检测框.
    candidates = remaining[::step]
    sampled = set(candidates)
    candidates.extend(key for key in remaining if key not in sampled)
    for key in candidates:
        if total >= target:
            break
        selected.add(key)
        total += groups[key]
    if total < target:
        raise ValueError(f"姿态子集需要至少 {target} 个实例,实际 {total}.")
    selected = sorted(selected)
    record_selection(selected)
    print(f"[姿态子集] {len(selected)} 张完整图像, {total} 个人体实例,目标至少 {target}.", flush=True)
    return selected


def select_samples(items, model, *, key=lambda value: value, required=()):
    """仅在评测调用处选择样本,校准数据的加载规则不受影响."""
    values = list(items)
    if not is_test():
        record_selection(key(item) for item in values)
        return values
    if model in ("resnet101", "vit_base_patch16_224"):
        required = (*required, *IMAGENET_EXAMPLES)
    return fixed_subset(values, min(LIMITS[model], len(values)), key=key, required=required)


def lfw_pairs(rows):
    """每折均匀选 25 对同人和 25 对异人,十折共 500 对."""
    if not is_test():
        record_selection(row["pair_id"] for row in rows)
        return rows
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["fold_id"]), int(row["is_same"]))].append(row)
    selected = []
    for fold in range(1, 11):
        for label in (0, 1):
            selected.extend(fixed_subset(groups[(fold, label)], 25,
                key=lambda row: int(row["pair_id"]), required=(0, 1, 300)))
    selected.sort(key=lambda row: int(row["pair_id"]))
    record_selection(row["pair_id"] for row in selected)
    return selected


def audio_records(rows):
    """ESC-50 每个类别、每一折选择一条,共 250 条."""
    if not is_test():
        record_selection(row["filename"] for row in rows)
        return rows
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["target"]), int(row["fold"]))].append(row)
    if len(groups) != 250:
        raise ValueError("ESC-50 必须覆盖 50 类和 5 折.")
    selected = [min(group, key=lambda row: row["filename"]) for group in groups.values()]
    selected.sort(key=lambda row: row["filename"])
    record_selection(row["filename"] for row in selected)
    return selected


def matching_reference(report):
    """只接受与当前范围和子集指纹一致的浮点参考报告."""
    if not is_test():
        return report.get("evaluation_scope", "full") == "full"
    return (report.get("evaluation_scope") == "subset" and
            report.get("sample_ids_sha256") == os.environ.get("MTK_EVAL_SAMPLE_SHA256") and
            bool(report.get("sample_ids_sha256")))


def accuracy_result(model, backend, values, count, unit, *, selection_id=None):
    """构造两端共同的任务精度接口,指标比例与样本单位必须明确."""
    if not isinstance(model, str) or not model or backend not in ("onnx", "pytorch", "tensorflow", "npu"):
        raise ValueError("任务精度必须声明模型和实际推理来源.")
    if type(count) is not int or count <= 0 or not isinstance(unit, str) or not unit:
        raise ValueError("任务精度必须有正整数样本数及明确单位.")
    if not isinstance(values, dict) or not values:
        raise ValueError("任务精度指标不能为空.")
    metrics = {}
    for name, value in values.items():
        if (not isinstance(name, str) or not name or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (name != "WER" and value > 1 + 1e-12)):
            raise ValueError(f"任务精度数值无效: {name}.")
        metrics[name] = {"value": float(value), "unit": "ratio",
                         "higher_is_better": name != "WER"}
    selection = selection_id or os.environ.get("MTK_EVAL_SAMPLE_SHA256")
    if not isinstance(selection, str) or not selection:
        raise ValueError("任务精度缺少实际评价样本集合标识.")
    return {"model": model, "backend": backend,
            "scope": "subset" if is_test() else "full",
            "samples": {"count": count, "unit": unit, "selection_id": selection},
            "metrics": metrics}


def read_accuracy(report, model, backend):
    """严格读取统一接口,不通过模型名称或旧字段猜测报告格式."""
    try:
        result = report["evaluation"]
        if result["model"] != model or result["backend"] != backend:
            raise ValueError("任务精度的模型或推理来源不匹配.")
        expected_scope = "subset" if is_test() else "full"
        if result["scope"] != expected_scope:
            raise ValueError("任务精度范围不匹配,不能混用子集与全量.")
        samples = result["samples"]
        values = {name: item["value"] for name, item in result["metrics"].items()}
        checked = accuracy_result(model, backend, values, samples["count"],
                                  samples["unit"], selection_id=samples["selection_id"])
        if checked != result:
            raise ValueError("任务精度接口字段或指标单位、方向不符合统一约定.")
        return result
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"统一任务精度接口不完整: {error}.") from error


def compare_accuracy(reference, actual):
    """只比较同模型同范围同样本同单位的任务指标,正损失表示变差."""
    for field in ("model", "scope", "samples"):
        if reference[field] != actual[field]:
            raise ValueError(f"浮点与 NPU 精度覆盖不一致: {field}.")
    if reference["backend"] not in ("onnx", "pytorch") or actual["backend"] != "npu":
        raise ValueError("任务精度对照必须来自浮点模型和真实 NPU.")
    if reference["metrics"].keys() != actual["metrics"].keys():
        raise ValueError("浮点与 NPU 指标名称不一致.")
    result = {}
    for name, baseline in reference["metrics"].items():
        measured = actual["metrics"][name]
        if any(baseline[field] != measured[field] for field in ("unit", "higher_is_better")):
            raise ValueError(f"浮点与 NPU 指标单位或方向不一致: {name}.")
        difference = baseline["value"] - measured["value"]
        if not baseline["higher_is_better"]:
            difference = -difference
        result[name] = {"fp32_accuracy": baseline["value"], "npu_accuracy": measured["value"],
                        "higher_is_better": baseline["higher_is_better"],
                        "accuracy_loss_percentage_points": difference * 100}
    return result


def publish_result(root, report, filename):
    """只发布两种最终报告,成功后清除结果目录中的旧格式文件."""
    from pathlib import Path
    import shutil

    if filename not in ("summary_test.json", "summary.json"):
        raise ValueError("结果文件只能是 summary_test.json 或 summary.json.")
    root = Path(root).resolve()
    directory = root / "results"
    if directory.is_symlink():
        raise ValueError("结果目录不能是符号链接.")
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    previous = list(directory.iterdir())
    for path in previous:
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("拒绝修改包含链接或越界文件的结果目录.")
    destination = directory / filename
    destination.write_text(text, encoding="utf-8")
    for path in previous:
        if path.name in ("summary_test.json", "summary.json"):
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    return destination
