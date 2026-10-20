"""验证板端 JSON 配置与快速结果路径,不运行模型推理."""

import argparse
from datetime import date
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.board import configure_parser, paths


class BoardConfigTest(unittest.TestCase):
    """验证结果隔离不改变模型部署位置,并保持原显式参数兼容."""

    def setUp(self):
        """仅在日期临时目录写入测试 JSON."""
        root = Path("/tmp/hailongcodex") / date.today().isoformat() / "board_config_tests"
        root.mkdir(parents=True, exist_ok=True)
        self.folder = Path(tempfile.mkdtemp(dir=root))
        self.config = self.folder / "config.json"
        self.config.write_text(json.dumps({
            "name": "fastsam",
            "board": {"dataset_dir": "/opt/mtk/datasets/coco/val2017",
                      "deploy_dir": "/opt/mtk/open_models/fastsam",
                      "results_dir": "results"}}))
        self.root = Path("/opt/mtk/open_models/fastsam")
        self.environment = mock.patch.dict(os.environ, {}, clear=True)
        self.environment.start()

    def tearDown(self):
        """恢复调用方环境,保留临时测试文件供核查."""
        self.environment.stop()

    def test_full_and_quick_results_share_deployment(self):
        """快速模式只给 results 内的运行编号添加后缀."""
        with mock.patch("sys.argv", ["infer.py"]):
            full = paths(self.root, self.config, "example")
            self.assertEqual(full.result, self.root / "results/example")
            os.environ["MTK_EVAL_TEST"] = "1"
            quick = paths(self.root, self.config, "example")
            self.assertEqual(quick.result, self.root / "results/example_test")
            self.assertEqual(quick.config["board"]["deploy_dir"], str(self.root))
            self.assertEqual(paths(self.root, self.config, "example_test").run_id, "example_test")

    def test_unsafe_run_id_rejected(self):
        """拒绝路径越界、空格和 Shell 字符."""
        for run_id in ("../other", "a/b", "a b", "a;command"):
            with self.assertRaises(ValueError):
                paths(self.root, self.config, run_id)

    def test_json_defaults_and_explicit_override(self):
        """读取快照补足必填参数,显式参数仍覆盖默认值."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=Path, required=True)
        with mock.patch("sys.argv", ["render.py", "--config", str(self.config),
                                     "--dataset", "/tmp/explicit"]):
            configure_parser(parser, self.root / "board/render.py",
                             {"dataset": lambda ctx: ctx.dataset})
            self.assertEqual(parser.parse_args().dataset, Path("/tmp/explicit"))

    def test_legacy_explicit_arguments(self):
        """没有配置快照时保留原命令行接口."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=Path, required=True)
        with mock.patch("sys.argv", ["evaluate.py", "--dataset", "/tmp/data"]):
            configure_parser(parser, self.root / "board/evaluate.py",
                             {"dataset": lambda ctx: ctx.dataset})
            self.assertEqual(parser.parse_args().dataset, Path("/tmp/data"))


if __name__ == "__main__":
    unittest.main()
