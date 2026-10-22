"""公共工具回归测试,不下载资源、不加载权重、不执行全量评测."""

import argparse
import contextlib
from datetime import date
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from tools.boxes import box_iou
from tools.files import copy_public_tools, read_json, require_file, sha256, single_file
from tools.process import run_command
from tools.quantization import tensor_metadata


class CommonTest(unittest.TestCase):
    """验证文件、命令、数值计算和量化契约的正常及异常路径."""

    def setUp(self):
        """每项测试使用独立临时目录,保留结果便于检查."""
        default_root = Path("/tmp/hailongcodex") / date.today().isoformat() / "common_tests"
        root = Path(os.environ.get("MTK_TEST_ROOT", default_root))
        root.mkdir(parents=True, exist_ok=True)
        self.work = Path(tempfile.mkdtemp(dir=root))

    def test_hash_empty_and_multiblock(self):
        """空文件与跨分块文件的哈希均等于标准 SHA256."""
        for content in (b"", b"abc", b"x" * (8 * 1024 * 1024 + 17)):
            path = self.work / "data.bin"
            path.write_bytes(content)
            self.assertEqual(sha256(path), hashlib.sha256(content).hexdigest())

    def test_json_and_file_checks(self):
        """UTF-8 JSON 可读取,空文件、目录和多个匹配项被拒绝."""
        path = self.work / "input.json"
        path.write_text('{"名称": 1}', encoding="utf-8")
        self.assertEqual(read_json(path), {"名称": 1})
        self.assertEqual(require_file(path), path)
        self.assertEqual(single_file(self.work, "*.json"), path)
        (self.work / "other.json").write_text("{}")
        with self.assertRaises(ValueError):
            single_file(self.work, "*.json")
        for missing in (self.work / "missing", self.work):
            with self.assertRaises(FileNotFoundError):
                require_file(missing)
        empty = self.work / "empty"
        empty.touch()
        with self.assertRaises(FileNotFoundError):
            require_file(empty)

    def test_public_tools_copy(self):
        """部署副本包含包入口和指定模块,拒绝上级路径."""
        copy_public_tools(self.work / "tools", ("files", "process"))
        self.assertEqual({p.name for p in (self.work / "tools").iterdir()},
                         {"__init__.py", "files.py", "process.py"})
        with self.assertRaises(ValueError):
            copy_public_tools(self.work / "tools", ("../storage",))

    def test_command_success_and_environment(self):
        """命令正确接收空格参数和环境变量,静默仅关闭回显."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = run_command([sys.executable, "-c", "import os; print(os.environ['VALUE'])"],
                                 quiet=True, env={**os.environ, "VALUE": "a b"},
                                 text=True, capture_output=True)
        self.assertEqual(result.stdout.strip(), "a b")
        self.assertEqual(output.getvalue(), "")

    def test_command_failure_is_not_hidden(self):
        """失败命令继续抛异常,原始子进程诊断不丢失."""
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                run_command([sys.executable, "-c", "import sys; print('failure', file=sys.stderr); sys.exit(7)"],
                            quiet=True, text=True, capture_output=True)
        self.assertEqual(caught.exception.returncode, 7)
        self.assertIn("failure", caught.exception.stderr)
        self.assertIn("[失败命令]", output.getvalue())

    def test_box_iou_geometry(self):
        """覆盖相同框、相离框、部分重叠、退化框与空候选."""
        boxes = np.array([[0, 0, 2, 2], [3, 3, 4, 4], [1, 0, 3, 2], [0, 0, 0, 0]], dtype=np.float32)
        np.testing.assert_allclose(box_iou(boxes[0], boxes), [1, 0, 1 / 3, 0])
        self.assertEqual(box_iou(boxes[0], boxes[:0]).shape, (0,))

    def test_quantization_types_and_legacy_shape(self):
        """INT8 和 INT16 均保留原模型所需的元数据字段."""
        for dtype in ("int8", "int16"):
            detail = {"name": "input", "shape": [1, 3, 2, 2], "type": dtype.upper(),
                      "quantization": {"scales": [0.125], "zero_points": [-1]}}
            value = tensor_metadata(detail, dtype)
            self.assertEqual(value, {"name": "input", "shape": [1, 3, 2, 2],
                                     "dtype": dtype, "scale": 0.125, "zero_point": -1})
            self.assertNotIn("name", tensor_metadata(detail, dtype, include_name=False))

    def test_quantization_invalid_contract(self):
        """拒绝无效尺度、无类型和逐通道 IO,不猜测量化参数."""
        detail = {"name": "x", "shape": [1], "type": "int8",
                  "quantization": {"scales": [0.1], "zero_points": [0]}}
        for scale in (0, -1, float("nan"), float("inf")):
            invalid = {**detail, "quantization": {"scales": [scale], "zero_points": [0]}}
            with self.assertRaises(ValueError):
                tensor_metadata(invalid)
        for dtype in ("uint8", "float32", ""):
            with self.assertRaises(ValueError):
                tensor_metadata({**detail, "type": dtype})
        with self.assertRaises(ValueError):
            tensor_metadata({**detail, "quantization": {"scales": [0.1, 0.2], "zero_points": [0, 0]}})

    @mock.patch.dict(os.environ, {"MTK_EVAL_SCOPE": "full"})
    def test_coco_model_status_policies(self):
        """用替身评测器验证两模型的状态、精度差和报告字段,不跑全量评测."""
        from tools import coco
        ids = set(range(5000))
        processed = self.work / "ids.txt"
        processed.write_text("\n".join(map(str, sorted(ids))))
        prediction = self.work / "prediction.json"
        prediction.write_text('[{"image_id": 0}]')
        timing = self.work / "timing.json"
        timing.write_text(json.dumps({"processed_images": 5000, "npu_ms": {"mean": 10},
                                      "peak_rss_kb": 2048}))
        annotation = mock.Mock()
        annotation.getImgIds.return_value = sorted(ids)
        evaluator = mock.Mock()
        evaluator.stats = [0.5]
        for model, fallback in (("yolov5s", "complete"), ("yolov8n", "board_verified")):
            for reference in (None, 0.6):
                args = argparse.Namespace(annotations=self.work / "ann.json", predictions=prediction,
                                          processed_ids=processed, timings=timing, fp32_map=reference,
                                          fp32_source="test", metrics=self.work / "metrics.json", run_id="test")
                with mock.patch.object(coco, "COCO", return_value=annotation), \
                     mock.patch.object(coco, "COCOeval", return_value=evaluator), \
                     contextlib.redirect_stdout(io.StringIO()):
                    coco.evaluate_coco(args, model=model, status_without_reference=fallback)
                metrics = read_json(args.metrics)
                self.assertEqual(metrics["model"], model)
                self.assertEqual(metrics["status"], fallback if reference is None else "complete")
                self.assertEqual(metrics["board_accuracy"], 0.5)
                if reference is not None:
                    self.assertAlmostEqual(metrics["accuracy_change_percentage_points"], -10)
        processed.write_text("0\n")
        with mock.patch.object(coco, "COCO", return_value=annotation), \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            coco.evaluate_coco(args, model="yolov8n", status_without_reference="board_verified")

    def test_llm_contract(self):
        """原生张量契约拒绝非 INT16 与非法尺度."""
        from tools.llm import package
        quant = SimpleNamespace(ScaleLength=lambda: 1, Scale=lambda index: 0.5)
        tensor = SimpleNamespace(Quantization=lambda: quant, Type=lambda: package.TensorType.INT16,
                                 Name=lambda: b"input", ShapeAsNumpy=lambda: np.array([1, 128]))
        graph = SimpleNamespace(TensorsLength=lambda: 1, Tensors=lambda index: tensor,
                                InputsLength=lambda: 1, OutputsLength=lambda: 1,
                                Inputs=lambda index: 0, Outputs=lambda index: 0)
        model = self.work / "model.tflite"
        model.write_bytes(b"fixture")
        with mock.patch.object(package.Model, "GetRootAsModel", return_value=SimpleNamespace(Subgraphs=lambda index: graph)):
            result = package.graph_contract(model)
            self.assertEqual(result["inputs"][0]["dtype"], "INT16")
            quant.Scale = lambda index: float("nan")
            with self.assertRaises(ValueError):
                package.graph_contract(model)


if __name__ == "__main__":
    unittest.main()
