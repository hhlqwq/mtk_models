"""验证快速子集、默认全量行为及参考报告的范围隔离."""

import argparse
import ast
import os
from pathlib import Path
import unittest
from unittest import mock

from tools import evaluation


class EvaluationTest(unittest.TestCase):
    """不加载模型,覆盖主机与板端共用的样本规则."""

    def setUp(self):
        """隔离每项测试的模式及样本指纹."""
        self.environment = mock.patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.arguments = mock.patch("sys.argv", ["test_evaluation.py"])
        self.arguments.start()

    def tearDown(self):
        """恢复调用方环境与参数."""
        self.arguments.stop()
        self.environment.stop()

    def test_default_full(self):
        """无参数保留输入全集及原有成功状态."""
        values = list(range(5000))
        self.assertEqual(evaluation.coco_ids(values), values)
        self.assertEqual(evaluation.result_status(), "complete")
        self.assertEqual(evaluation.scope_metadata(), {"evaluation_scope": "full"})

    def test_argument_and_failure_status(self):
        """独立入口识别快速参数,失败状态不被改为成功."""
        with mock.patch("sys.argv", ["evaluate.py", "-test"]):
            parser = argparse.ArgumentParser()
            evaluation.add_test_argument(parser)
            self.assertTrue(parser.parse_args().test)
            self.assertEqual(evaluation.result_status(), "smoke_test")
            self.assertEqual(evaluation.result_status("incomplete"), "incomplete")

    def test_coco_subset_deterministic(self):
        """固定 100 张包含示例,输入顺序不影响两端指纹."""
        os.environ["MTK_EVAL_TEST"] = "1"
        values = list(range(5000)) + [210299, 252219, 481386]
        selected = evaluation.coco_ids(values)
        digest = evaluation.scope_metadata()["sample_ids_sha256"]
        self.assertEqual(len(selected), 100)
        self.assertTrue(set(evaluation.COCO_EXAMPLES).issubset(selected))
        self.assertEqual(selected, evaluation.coco_ids(reversed(values)))
        self.assertEqual(digest, evaluation.scope_metadata()["sample_ids_sha256"])

    def test_lfw_balanced_folds(self):
        """十折每折各选五个同人和异人配对."""
        os.environ["MTK_EVAL_TEST"] = "1"
        rows = [{"fold_id": fold, "is_same": label,
                 "pair_id": (fold - 1) * 600 + label * 300 + index}
                for fold in range(1, 11) for label in (0, 1)
                for index in range(300)]
        selected = evaluation.lfw_pairs(rows)
        self.assertEqual(len(selected), 100)
        for fold in range(1, 11):
            for label in (0, 1):
                self.assertEqual(sum(int(row["fold_id"]) == fold and
                                     int(row["is_same"]) == label for row in selected), 5)

    def test_audio_coverage(self):
        """ESC-50 子集每类每折各一条,且固定按文件名排序."""
        os.environ["MTK_EVAL_TEST"] = "1"
        rows = [{"target": target, "fold": fold,
                 "filename": f"{fold}-{target:02d}-{index}.wav"}
                for fold in range(1, 6) for target in range(50) for index in range(8)]
        selected = evaluation.audio_records(rows)
        self.assertEqual(len(selected), 250)
        self.assertEqual(len({(row["target"], row["fold"]) for row in selected}), 250)

    def test_reference_scope(self):
        """拒绝全量参考、其他子集参考及缺少指纹的快速报告."""
        os.environ["MTK_EVAL_TEST"] = "1"
        digest = evaluation.record_selection(["a", "b"])
        self.assertFalse(evaluation.matching_reference({"evaluation_scope": "full"}))
        self.assertFalse(evaluation.matching_reference({"evaluation_scope": "subset"}))
        self.assertFalse(evaluation.matching_reference(
            {"evaluation_scope": "subset", "sample_ids_sha256": "wrong"}))
        self.assertTrue(evaluation.matching_reference(
            {"evaluation_scope": "subset", "sample_ids_sha256": digest}))

    def test_llm_cli_limits(self):
        """只调用真实参数解析函数,验证 LLM 诊断上限而不加载模型."""
        root = Path(__file__).resolve().parents[1]
        cases = (
            ("evaluate_mmlu.py", ["prepare", "--output", "/tmp/output"],
             "diagnostic_questions", 0, 0),
            ("evaluate_quality.py", ["--model", "/tmp/model", "--corpus", "/tmp/corpus",
             "--output", "/tmp/output", "--backend", "pytorch"],
             "diagnostic_blocks", 0, 4),
            ("evaluate_board_quality.py", ["--package", "/tmp/package",
             "--quality-data", "/tmp/data", "--library", "/tmp/library",
             "--output", "/tmp/output"], "diagnostic_blocks", 0, 4),
        )
        for filename, arguments, field, full_value, quick_value in cases:
            tree = ast.parse((root / "tools/llm" / filename).read_text())
            main = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "main")
            for quick in (False, True):
                captured = []
                namespace = {"__doc__": "参数检查", "argparse": argparse, "Path": Path,
                             "is_test": evaluation.is_test,
                             "add_test_argument": evaluation.add_test_argument,
                             "evaluate": captured.append, "prepare": captured.append}
                exec(compile(ast.Module(body=[main], type_ignores=[]), filename, "exec"),
                     namespace)
                argv = [filename, *arguments, *(["-test"] if quick else [])]
                with mock.patch("sys.argv", argv):
                    namespace["main"]()
                self.assertEqual(len(captured), 1)
                args = captured[0]
                self.assertEqual(getattr(args, field), quick_value if quick else full_value)
                self.assertEqual(args.output, Path("/tmp/output/test" if quick else "/tmp/output"))
                if filename == "evaluate_mmlu.py":
                    self.assertEqual(args.limit, 16 if quick else 1000)
                if filename == "evaluate_quality.py" and quick:
                    self.assertTrue(args.skip_demo)
                    self.assertFalse(args.full_test)

    def test_small_or_duplicate_dataset_rejected(self):
        """不把缺少样本或重复标识的数据集当作完整快速评测."""
        with self.assertRaises(ValueError):
            evaluation.fixed_subset([1, 2], 3)
        with self.assertRaises(ValueError):
            evaluation.fixed_subset([1, 1, 2], 2)


if __name__ == "__main__":
    unittest.main()
