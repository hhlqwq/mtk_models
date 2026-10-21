"""全量与快速流程验证的固定子集规则,供主机和板端共同使用."""

from collections import defaultdict
import hashlib
import json
import os


LIMITS = {"fastsam": 100, "yolov5s": 100, "yolov8n": 100,
          "rtmpose_body2d": 100, "vit_base_patch16_224": 200,
          "mobilefacenet": 100, "depth_anything_v2_small": 100,
          "yamnet": 250, "whisper_tiny": 5}
COCO_EXAMPLES = (139, 285, 785, 210299, 252219, 481386)
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
        result["purpose"] = "流程验证,不代表全量精度"
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
    """保留固定示例后按键补足数量,再按键排序,不随机抽样."""
    items = list(items)
    if len(items) < count:
        raise ValueError(f"快速模式需要至少 {count} 个样本,实际 {len(items)}.")
    mapping = {key(item): item for item in items}
    if len(mapping) != len(items):
        raise ValueError("样本标识重复.")
    wanted = {value for value in required if value in mapping}
    if len(wanted) > count:
        raise ValueError("固定示例数超过快速评测样本数.")
    for value in sorted(mapping):
        if len(wanted) >= count:
            break
        wanted.add(value)
    selected = [mapping[value] for value in sorted(wanted)]
    record_selection(key(item) for item in selected)
    return selected


def coco_ids(ids):
    """COCO 快速模式固定 100 张,纳入检测、分割和姿态示例."""
    values = sorted(int(value) for value in ids)
    return fixed_subset(values, 100, required=COCO_EXAMPLES) if is_test() else values


def select_samples(items, model, *, key=lambda value: value, required=()):
    """仅在评测调用处选择样本,校准数据的加载规则不受影响."""
    values = list(items)
    if not is_test():
        return values
    return fixed_subset(values, LIMITS[model], key=key, required=required)


def lfw_pairs(rows):
    """每折分别选 5 对同人和 5 对异人,保持十折且覆盖已有示例."""
    if not is_test():
        return rows
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["fold_id"]), int(row["is_same"]))].append(row)
    selected = []
    for fold in range(1, 11):
        for label in (0, 1):
            selected.extend(fixed_subset(groups[(fold, label)], 5,
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
