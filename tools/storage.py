"""读取不提交 Git 的资源路径配置,代码目录与模型资源目录分离."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def reuse_weight(source, destination, expected_sha256):
    """复用官方权重,共享盘不支持软链接时使用经过哈希校验的副本."""
    source = Path(source).resolve(strict=True)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() != source:
        # 只更新引用,不删除引用所指向的原始权重.
        destination.unlink()
    if not destination.exists():
        try:
            destination.symlink_to(source)
            print(f"[权重] 已引用官方资源: {source.name}", flush=True)
        except OSError:
            print(f"[权重] 共享存储不支持软链接,复制: {source.name}", flush=True)
            temporary = destination.with_name(destination.name + ".copy-part")
            copied = 0
            with source.open("rb") as incoming, temporary.open("wb") as outgoing:
                for block in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
                    outgoing.write(block)
                    copied += len(block)
                    if copied % (512 * 1024 * 1024) == 0:
                        print(f"[权重复制] {copied // 1024**2} MiB", flush=True)
            temporary.replace(destination)
    digest = hashlib.sha256()
    processed = 0
    with destination.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            processed += len(block)
            if processed % (512 * 1024 * 1024) == 0:
                print(f"[权重校验] {processed // 1024**2} MiB", flush=True)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"共享存储权重副本哈希不匹配: {destination.name}")


def storage_roots():
    """读取本地配置或环境变量,返回模型和数据集的绝对路径."""
    config_path = Path(os.environ.get(
        "MTK_STORAGE_CONFIG", REPO_ROOT / "configs/storage.local.json"))
    config = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    roots = {}
    for key, variable in (("models_root", "MTK_MODEL_STORAGE_ROOT"),
                          ("dataset_root", "MTK_DATASET_ROOT")):
        value = os.environ.get(variable) or config.get(key)
        if not value or not Path(value).is_absolute():
            raise ValueError(
                f"请在 configs/storage.local.json 配置 {key} 的绝对路径,"
                f"或设置 {variable}.")
        roots[key] = Path(value)
    return roots


def model_storage(model_directory):
    """按模型的场景、类别、名称映射到外部资源目录."""
    relative = Path(model_directory).resolve().relative_to(REPO_ROOT / "models")
    if len(relative.parts) != 3:
        raise ValueError("模型代码目录必须为 models/场景/类别/模型名.")
    return storage_roots()["models_root"] / relative


def load_model_config(config_path):
    """读取 FastSAM YAML,将资源相对路径映射到本地存储配置."""

    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    roots = storage_roots()
    os.environ["MTK_MODEL_STORAGE_ROOT"] = str(roots["models_root"])
    os.environ["MTK_DATASET_ROOT"] = str(roots["dataset_root"])
    model_root = model_storage(config_path.parent)
    for key in ("weights", "output_dir"):
        value = Path(config["model"][key])
        if not value.is_absolute():
            config["model"][key] = str(model_root / value)
    dataset = Path(config["data"]["dataset_dir"])
    if not dataset.is_absolute():
        config["data"]["dataset_dir"] = str(roots["dataset_root"] / dataset)
    return config


def platform_environment():
    """读取 G720 平台配置,生成进入容器时使用的公共环境变量."""
    config = yaml.safe_load(
        (REPO_ROOT / "configs/platforms/genio_720.yaml").read_text(encoding="utf-8"))
    toolchain = config["toolchain"]
    sdk = Path(os.environ.get("MTK_NP_ROOT") or toolchain["sdk_root"])
    ncc = Path(os.environ.get("NCC_ROOT") or sdk / "neuron_sdk/host")
    defaults = {
        "MTK_CONTAINER_NAME": toolchain["container_name"],
        "MTK_HOST_SDK_ROOT": toolchain["host_sdk_root"],
        "MTK_NP_ROOT": sdk,
        "NCC_ROOT": ncc,
        "MTK_NCC_BIN": ncc / "bin",
        "MTK_NCC_LIB": ncc / "lib",
        "MTK_NEURON_INCLUDE": ncc / "include",
        "MTK_G720_CPP_TOOLCHAIN_ROOT": toolchain["cpp_toolchain_root"],
        "CROSS_CXX": toolchain["cross_cxx"],
        "MTK_BOARD_HOST": config["board_host"],
        "MTK_BOARD_ROOT": config["board_root"],
        "MTK_BOARD_OPEN_MODELS_ROOT": config["board_open_models_root"],
        "MTK_BOARD_DATASETS_ROOT": config["board_datasets_root"],
        "MTK_BOARD_MODELZOO_ROOT": config["board_modelzoo_root"],
    }
    return {name: os.environ.get(name) or value for name, value in defaults.items()}


def main():
    """输出经过 Shell 转义的变量,供 Bash 入口加载."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--print-build-dir", type=Path, metavar="MODEL_CONFIG",
                        help="只输出模型配置对应的构建目录.")
    args = parser.parse_args()
    if args.print_build_dir:
        config = load_model_config(args.print_build_dir)
        print(Path(config["model"]["output_dir"]) / "build")
        return
    roots = storage_roots()
    values = {"MTK_MODEL_STORAGE_ROOT": roots["models_root"],
              "MTK_DATASET_ROOT": roots["dataset_root"]}
    values.update(platform_environment())
    config_path = Path(os.environ.get(
        "MTK_STORAGE_CONFIG", REPO_ROOT / "configs/storage.local.json"))
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("container_image"):
            values["MTK_G720_IMAGE"] = config["container_image"]
    if args.model:
        values["MODEL_ASSETS_DIR"] = model_storage(args.model)
    if args.model_config:
        model_config = load_model_config(args.model_config)
        values["MODEL_BUILD_DIR"] = Path(model_config["model"]["output_dir"]) / "build"
    for name, value in values.items():
        print(f"export {name}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()
