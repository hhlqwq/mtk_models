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
    return fixed_subset(values, 300, required=COCO_EXAMPLES) if is_test() else values


def pose_image_ids(detections):
    """按整图保留全部人体框,均匀取图直到至少覆盖 500 个人体实例."""
    groups = defaultdict(int)
    for row in detections:
        groups[int(row["image_id"])] += 1
    keys = sorted(groups)
    if not is_test():
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
        return values
    if model in ("resnet101", "vit_base_patch16_224"):
        required = (*required, *IMAGENET_EXAMPLES)
    return fixed_subset(values, min(LIMITS[model], len(values)), key=key, required=required)


def lfw_pairs(rows):
    """每折均匀选 25 对同人和 25 对异人,十折共 500 对."""
    if not is_test():
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
