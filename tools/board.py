"""板端步骤读取 JSON 配置快照,不依赖主机或 PyYAML."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys

from tools.evaluation import add_test_argument, is_test
from tools.process import run_command


def context(entry):
    """板端沿用原部署目录,快速结果仅追加运行编号的 _test 后缀."""
    root = Path(entry).resolve().parent.parent
    parser = argparse.ArgumentParser(description="执行当前板端独立步骤.")
    parser.add_argument("--config", type=Path, default=root / "config.json")
    parser.add_argument("--run-id", default=os.environ.get("EVAL_RUN_ID"))
    add_test_argument(parser)
    args = parser.parse_args()
    return paths(root, args.config, args.run_id)


def paths(root, config_path, run_id=None):
    """从上传的 JSON 快照计算本次结果位置."""
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_test() and not run_id.endswith("_test"):
        run_id += "_test"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("运行编号只允许字母、数字、下划线和连字符.")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    result = root / config["board"].get("results_dir", "results") / run_id
    return argparse.Namespace(root=root, config=config, run_id=run_id,
        dataset=Path(config["board"]["dataset_dir"]), result=result, work=result / "work")


def command(ctx, filename, *arguments, **kwargs):
    """执行板端程序或 Python 脚本,失败时保留原始错误."""
    path = ctx.root / "board" / filename
    prefix = [sys.executable] if path.suffix == ".py" else []
    run_command([*prefix, path, *arguments], **kwargs)


def configure_parser(parser, entry, bindings):
    """为公共指标和绘图工具提供板端配置默认值,保留原命令兼容性."""
    if not any(action.dest == "test" for action in parser._actions):
        add_test_argument(parser)
    if not any(action.dest == "config" for action in parser._actions):
        parser.add_argument("--config", type=Path)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    bootstrap.add_argument("--run-id", default=os.environ.get("EVAL_RUN_ID"))
    options, _ = bootstrap.parse_known_args()
    if options.config is None:
        return
    ctx = paths(Path(entry).resolve().parent.parent, options.config, options.run_id)
    for action in parser._actions:
        if action.dest in bindings:
            action.default = bindings[action.dest](ctx)
            action.required = False
