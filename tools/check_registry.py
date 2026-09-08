"""检查模型注册表与交付目录的一致性."""

import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "registry" / "models.yaml"
REQUIRED_FILES = (
    "README.md",
    "model_card.md",
    "LICENSE",
    "model.yaml",
    "original/source_url.txt",
    "models/README.md",
    "deploy/convert.sh",
    "deploy/build.sh",
    "docs/accuracy.md",
    "docs/benchmark.md",
)
VALID_STATUSES = {
    "not_started",
    "environment_ready",
    "converted",
    "board_verified",
    "complete",
    "unsupported",
}


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 文件.

    Args:
        path: YAML 文件路径.

    Returns:
        YAML 顶层字典.

    Raises:
        ValueError: YAML 顶层不是字典.
    """
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须是字典: {path}")
    return data


def check_model(entry: dict[str, Any]) -> list[str]:
    """检查单个注册模型的路径、元数据和状态.

    Args:
        entry: `registry/models.yaml` 中的单模型条目.

    Returns:
        发现的问题列表；空列表表示通过.
    """
    errors: list[str] = []
    model_id = str(entry.get("id", "<missing-id>"))
    relative_path = Path(str(entry.get("path", "")))
    model_root = PROJECT_ROOT / relative_path
    if not model_root.is_dir():
        return [f"{model_id}: 模型目录不存在: {relative_path}"]

    for required_file in REQUIRED_FILES:
        if not (model_root / required_file).is_file():
            errors.append(f"{model_id}: 缺少文件: {required_file}")

    model_yaml_path = model_root / "model.yaml"
    if model_yaml_path.is_file():
        model_yaml = load_yaml(model_yaml_path)
        if model_yaml.get("id") != model_id:
            errors.append(
                f"{model_id}: model.yaml id 不一致: {model_yaml.get('id')}")
        if model_yaml.get("scenario") != entry.get("scenario"):
            errors.append(f"{model_id}: model.yaml scenario 与注册表不一致.")
        if model_yaml.get("category") != entry.get("category"):
            errors.append(f"{model_id}: model.yaml category 与注册表不一致.")

    platforms = entry.get("platforms", {})
    if not isinstance(platforms, dict):
        errors.append(f"{model_id}: platforms 必须是字典.")
    else:
        for platform, status in platforms.items():
            if status not in VALID_STATUSES:
                errors.append(
                    f"{model_id}: {platform} 使用非法状态: {status}")
    return errors


def main() -> int:
    """检查整个注册表并输出关键节点信息.

    Returns:
        无错误时返回 0,否则返回 1.
    """
    registry = load_yaml(REGISTRY_PATH)
    models = registry.get("models", [])
    if not isinstance(models, list):
        print("[ERROR] registry.models 必须是列表.")
        return 1

    print(f"[CHECK] 开始检查 {len(models)} 个注册模型.")
    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(models, start=1):
        if not isinstance(entry, dict):
            errors.append(f"第 {index} 个模型条目不是字典.")
            continue
        model_id = str(entry.get("id", "<missing-id>"))
        print(f"[CHECK] {index}/{len(models)}: {model_id}")
        if model_id in seen_ids:
            errors.append(f"注册表存在重复 id: {model_id}")
        seen_ids.add(model_id)
        errors.extend(check_model(entry))

    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        print(f"[FAIL] 共发现 {len(errors)} 个问题.")
        return 1
    print("[OK] 模型注册表与目录结构一致.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
