"""验证全部模型的真实精度出口、报告、数值误差和总入口,不运行模型."""

import ast
import copy
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from tools import evaluation
from tools.board import paths
from tools.compile_accuracy import save_subset, subset_report
from tools.output_metrics import Moments, OutputComparison, ReferenceWriter


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "resnet101": ("Top-1", "image"),
    "vit_base_patch16_224": ("Top-1", "image"),
    "yolov5s": ("mAP@0.5:0.95", "image"),
    "yolov8n": ("mAP@0.5:0.95", "image"),
    "mobilefacenet": ("LFW 验证准确率", "face_pair"),
    "rtmpose_body2d": ("WholeBody AP", "person_instance"),
    "depth_anything_v2_small": ("DA-2K 点对准确率", "annotated_point_pair"),
    "fastsam": ("segm AR@100", "image"),
    "yamnet": ("ESC-50 macro AP (folds 2-5)", "audio_clip"),
    "whisper_tiny": ("WER", "utterance"),
    "mit_bevfusion": ("mAP", "frame"),
    "minicpm5_2b": ("MMLU Accuracy", "question"),
    "deepseek_r1_distill_qwen_1_5b": ("MMLU Accuracy", "question"),
}


class DeployFlowTest(unittest.TestCase):
    """只使用小数组、临时文件和命令替身检查接口与控制流程."""

    def setUp(self):
        """隔离范围、样本标识及临时目录,不访问正式模型或数据集."""
        self.environment = mock.patch.dict(os.environ, {"MTK_EVAL_SCOPE": "subset"})
        self.environment.start()
        base = Path("/tmp/hailongcodex") / date.today().isoformat() / "deploy_flow_tests"
        base.mkdir(parents=True, exist_ok=True)
        self.work = Path(tempfile.mkdtemp(dir=base))
        evaluation.record_selection(["sample"])

    def tearDown(self):
        """恢复调用方环境,测试文件留在日期临时目录."""
        self.environment.stop()

    def test_all_models_both_scopes(self):
        """全部模型两端使用相同接口,损失方向、单位和范围逐项校验."""
        for scope in ("subset", "full"):
            os.environ["MTK_EVAL_SCOPE"] = scope
            for model, (metric, unit) in TASKS.items():
                with self.subTest(scope=scope, model=model):
                    backend = "pytorch" if unit == "question" else "onnx"
                    baseline = evaluation.accuracy_result(model, backend, {metric: 0.8}, 1, unit)
                    actual = evaluation.accuracy_result(model, "npu", {metric: 0.7}, 1, unit)
                    self.assertEqual(evaluation.read_accuracy({"evaluation": baseline}, model, backend), baseline)
                    loss = evaluation.compare_accuracy(baseline, actual)[metric]
                    self.assertAlmostEqual(loss["accuracy_loss_percentage_points"], -10 if metric == "WER" else 10)
                    broken = copy.deepcopy(actual)
                    broken["samples"]["unit"] = "wrong"
                    with self.assertRaises(ValueError):
                        evaluation.compare_accuracy(baseline, broken)
                    broken = copy.deepcopy(actual)
                    broken["samples"]["selection_id"] = "other"
                    with self.assertRaises(ValueError):
                        evaluation.compare_accuracy(baseline, broken)

    def test_real_producer_expressions(self):
        """执行真实源码中的接口构造表达式,发现字段名和变量衔接错误."""
        entry = (ROOT / "models/deploy/compile.sh").read_text(encoding="utf-8")
        models = re.search(r"MODELS=\(\n(.*?)\n\)", entry, re.S).group(1).split()
        generic = {"vit_base_patch16_224", "mobilefacenet", "rtmpose_body2d",
                   "depth_anything_v2_small", "fastsam", "whisper_tiny"}
        for model_path in models:
            model = model_path.rsplit("/", 1)[-1]
            deploy = ROOT / "models" / model_path / "deploy"
            for backend in ("host", "board"):
                if model in ("minicpm5_2b", "deepseek_r1_distill_qwen_1_5b"):
                    source = ROOT / "tools/llm/evaluate_mmlu.py"
                elif backend == "host" and model in generic:
                    source = ROOT / "tools/accuracy/evaluate_onnx.py"
                elif backend == "host" and model == "yolov5s":
                    source = ROOT / "tools/accuracy/yolov5s_val_coco.py"
                elif backend == "board" and model in ("yolov5s", "yolov8n"):
                    source = ROOT / "tools/coco.py"
                elif model == "mit_bevfusion":
                    source = deploy / backend / ("reference.py" if backend == "host" else "evaluate.py")
                elif model == "resnet101":
                    source = deploy / backend / ("evaluate_onnx.py" if backend == "host" else "evaluate.py")
                else:
                    source = deploy / backend / ("evaluate_onnx.py" if backend == "host" else "full_accuracy_board.py")
                tree = ast.parse(source.read_text(encoding="utf-8"))
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name) and node.func.id == "accuracy_result"]
                self.assertTrue(calls, str(source))
                metric, unit = TASKS[model]
                report = {"accuracy": 0.8, "map_50_95": 0.8, "npu_top1": 0.8,
                          "pairwise_accuracy": 0.8, "AR_100": 0.8,
                          "metrics": {"held_out": {"map": 0.8}}}
                namespace = dict(accuracy_result=evaluation.accuracy_result, record_selection=evaluation.record_selection,
                                 evaluation_ids=lambda records, indices: ["sample"],
                                 MODEL=model, model=model, name=model, metric=metric, unit=unit,
                                 report=report, summary=report, units={name: value[1] for name, value in TASKS.items()},
                                 args=SimpleNamespace(model=model, model_name=model,
                                                      backend="pytorch" if backend == "host" else "npu",
                                                      mode="onnx", onnx=True),
                                 value=0.8, accuracy=0.8, primary=0.8, board_map=0.8,
                                 samples=1, expected=1, total_pairs=1, image_ids=[1], ids=[1],
                                 paths=[1], images=[1], rows=[1], records=[1], done=[1], successful={"a": 1},
                                 tokens=[1], indices=[], expected_ids=[1],
                                 verification={"verification_accuracy": 0.8},
                                 metrics={"wholebody": {"AP": 0.8}}, grouped={1: [1]},
                                 backend="onnxruntime" if backend == "host" else "python_neuron_runtime_hw")
                if model == "whisper_tiny" and backend == "board":
                    namespace["accuracy"] = {"value": 0.8}
                if model == "mit_bevfusion":
                    namespace["accuracy"] = {"mAP": 0.8, "NDS": 0.7}
                with self.subTest(model=model, side=backend, source=str(source)):
                    result = eval(compile(ast.Expression(calls[0]), str(source), "eval"), namespace)
                    self.assertEqual(result["model"], model)
                    self.assertEqual(result["samples"]["unit"], unit)
                    expected_backend = "pytorch" if backend == "host" and unit == "question" else "onnx" if backend == "host" else "npu"
                    self.assertEqual(result["backend"], expected_backend)

    def test_invalid_contract(self):
        """缺失、非有限数、范围和方向错误在汇总前拒绝."""
        with self.assertRaisesRegex(ValueError, "接口不完整"):
            evaluation.read_accuracy({"accuracy": 0.8}, "resnet101", "onnx")
        for value in (float("nan"), float("inf"), -1, True, 1.1):
            with self.assertRaises(ValueError):
                evaluation.accuracy_result("resnet101", "onnx", {"Top-1": value}, 1, "image")
        result = evaluation.accuracy_result("resnet101", "onnx", {"Top-1": 0.8}, 1, "image")
        result["metrics"]["Top-1"]["higher_is_better"] = False
        with self.assertRaises(ValueError):
            evaluation.read_accuracy({"evaluation": result}, "resnet101", "onnx")
        result = evaluation.accuracy_result("resnet101", "onnx", {"Top-1": 0.8}, 1, "image")
        os.environ["MTK_EVAL_SCOPE"] = "full"
        with self.assertRaises(ValueError):
            evaluation.read_accuracy({"evaluation": result}, "resnet101", "onnx")

    def test_official_score_roundoff(self):
        """官方 nuScenes 完全匹配评分可略高于 1,保留原值而不误判非法."""
        score = {"mAP": 1.0000000000000004, "NDS": 1.0000000000000002}
        report = evaluation.accuracy_result("mit_bevfusion", "onnx", score, 1, "frame")
        self.assertEqual(report["metrics"]["mAP"]["value"], score["mAP"])
        self.assertEqual(evaluation.read_accuracy({"evaluation": report}, "mit_bevfusion", "onnx"), report)

    def test_numeric_reference_and_report(self):
        """真实 NPZ 读写、误差累加和子集报告形成闭环,全量不读数值参考."""
        os.environ["MTK_OUTPUT_METRICS"] = "1"
        models = self.work / "models"
        models.mkdir()
        dla = models / "model.dla"
        dla.write_bytes(b"test-dla")
        os.environ["MTK_OUTPUT_DLA_FILES"] = json.dumps({"model.dla": str(dla)})
        source = self.work / "model.onnx"
        source.write_bytes(b"test-onnx")
        reference = self.work / "build/reference/output_reference.npz"
        reference.parent.mkdir(parents=True)
        writer = ReferenceWriter(reference)
        tensor = np.array([[1, 0]], np.float32)
        writer.add(source, [tensor], {"output": tensor})
        writer.finish()
        ctx = SimpleNamespace(root=self.work, models=models, work=self.work / "build/work")
        comparison = OutputComparison(ctx)
        comparison.update([tensor], {"output": np.array([[0, 1]], np.float32)})
        comparison.finish()
        errors = json.loads((ctx.work / "output_metrics.json").read_text())
        values = errors["outputs"]["output"]
        self.assertAlmostEqual(values["cosine_similarity"], 0)
        self.assertAlmostEqual(values["rmse"], 1)
        self.assertAlmostEqual(values["nrmse_percent"], 100 * 2 ** 0.5)
        evaluation.record_selection(["sample"])
        normal = evaluation.accuracy_result("resnet101", "onnx", {"Top-1": 0.8}, 1, "image")
        baseline = {"model": "resnet101", "metrics": {"Top-1": 0.8}, "samples": 1,
                    "evaluation": normal, "dla_artifacts": {}}
        report = subset_report(baseline, {"Top-1": 0.7})
        report["output_errors"] = errors
        save_subset(self.work, report)
        result = json.loads((self.work / "results/summary_test.json").read_text())
        self.assertEqual(result["output_errors"], errors)
        os.environ["MTK_EVAL_SCOPE"] = "full"
        with mock.patch("tools.output_metrics.np.load", side_effect=AssertionError("全量不能读取张量参考")):
            full = OutputComparison(ctx)
            full.update([], {})
            full.finish()

    def test_report_isolation_and_invalid_write(self):
        """只保留两份报告,无效新结果不损坏旧报告."""
        evaluation.publish_result(self.work, {"test": 1}, "summary_test.json")
        directory = self.work / "results"
        (directory / "summary_sub.json").write_text("old")
        evaluation.publish_result(self.work, {"full": 1}, "summary.json")
        self.assertEqual({p.name for p in directory.iterdir()}, {"summary_test.json", "summary.json"})
        with self.assertRaises(ValueError):
            evaluation.publish_result(self.work, {"bad": float("nan")}, "summary.json")
        self.assertEqual(json.loads((directory / "summary.json").read_text()), {"full": 1})

    def test_preflight_before_work_cleanup(self):
        """错误 DLA 在推理前拒绝,不能先删除上次运行现场."""
        root = self.work
        work = root / "build/work"
        work.mkdir(parents=True)
        (work / "old").write_text("keep")
        config = {"name": "resnet101", "deployment": {"model_directory": "."},
                  "board": {"dataset_dir": str(root / "dataset")}}
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config))
        evaluation.record_selection(["sample"])
        reference = {"evaluation": evaluation.accuracy_result("resnet101", "onnx", {"Top-1": 0.8}, 1, "image"),
                     "dla_artifacts": {"model.dla": "wrong"}}
        folder = root / "build/reference"
        folder.mkdir()
        (folder / "subset.json").write_text(json.dumps(reference))
        (root / "model.dla").write_bytes(b"different")
        with self.assertRaises(ValueError):
            paths(root, config_path, "test", new_run=True)
        self.assertTrue((work / "old").is_file())

    def make_shell_environment(self, host=False):
        """构造真实总入口的运行环境,所有远程命令和模型入口均为替身."""
        root = self.work / "flow"
        deploy = root / "models/deploy" if host else root / "deploy"
        deploy.mkdir(parents=True)
        for name in ("compile.sh", "evaluate.sh"):
            (deploy / name).write_bytes((ROOT / "models/deploy" / name).read_bytes())
        entry = (deploy / "compile.sh").read_text()
        models = re.search(r"MODELS=\(\n(.*?)\n\)", entry, re.S).group(1).split()
        log = self.work / "calls.txt"
        commands = self.work / "commands"
        commands.mkdir()
        for name in ("ssh", "scp", "rsync"):
            script = commands / name
            script.write_text('#!/bin/bash\nif [[ "${1:-}" == -V ]]; then echo OpenSSH >&2; exit 0; fi\nprintf "' + name + ':%s\\n" "$*" >> "$FLOW_LOG"\n')
            script.chmod(0o755)
        key = self.work / "key"
        key.write_text("test-only")
        local = deploy / "platform.local.sh"
        local.write_text(f'MTK_RESULTS_HOST=test-only\nMTK_RESULTS_ROOT={root}/returned\nMTK_RESULTS_KEY={key}\n')
        environment = dict(os.environ, FLOW_LOG=str(log), PATH=str(commands) + ":" + os.environ["PATH"])
        for model in models:
            folder = root / "models" / model / "deploy" if host else root / model.rsplit("/", 1)[-1]
            folder.mkdir(parents=True)
            for phase in ("compile", "evaluate"):
                (folder / f"{phase}.sh").write_text(
                    '#!/bin/bash\nprintf "' + phase + ':' + model + '\\n" >> "$FLOW_LOG"\n'
                    + '[[ "${FAIL_MODEL:-}" != "' + model + '" ]] || exit 7\n')
            if not host:
                (folder / "examples/output").mkdir(parents=True)
                (folder / "examples/output/sample.txt").write_text("example")
                (folder / "results").mkdir()
                for report in ("summary.json", "summary_test.json"):
                    (folder / "results" / report).write_text("{}")
        if host:
            (root / "tools").mkdir()
            (root / "tools/storage.py").write_text("# 入口检测占位,不会执行.\n")
            python = commands / "python"
            python.write_text('#!/bin/bash\nprintf "%s\\n" "$FLOW_CONFIG"\n')
            python.chmod(0o755)
            environment.update(MTK_MODELS_ROOT=str(root), MTK_BOARD_HOST="test-only",
                               MTK_BOARD_OPEN_MODELS_ROOT="/test-only", FLOW_CONFIG=str(local))
        return deploy, environment, log, models

    def test_board_compile_and_full_return(self):
        """真实板端总入口顺序执行 13 模型,全量末尾回传不再次触发 compile."""
        deploy, environment, log, models = self.make_shell_environment()
        subprocess.run(["bash", str(deploy / "compile.sh")], env=environment,
                       capture_output=True, text=True, check=True)
        rows = log.read_text().splitlines()
        self.assertEqual([row for row in rows if row.startswith("compile:")],
                         ["compile:" + model for model in models])
        transfers = [row for row in rows if row.startswith("rsync:")]
        self.assertEqual(len(transfers), 26)
        self.assertTrue(all("summary.json" not in row and ".dla" not in row for row in transfers))
        log.write_text("")
        subprocess.run(["bash", str(deploy / "evaluate.sh")], env=environment,
                       capture_output=True, text=True, check=True)
        rows = log.read_text().splitlines()
        self.assertFalse(any(row.startswith("compile:") for row in rows))
        self.assertEqual([row for row in rows if row.startswith("evaluate:")],
                         ["evaluate:" + model for model in models])
        self.assertEqual(len([row for row in rows if row.startswith("rsync:")]), 39)

    def test_host_order_and_stop_on_failure(self):
        """主机顺序调用原模型入口,失败时停止且不触发板端推理."""
        deploy, environment, log, models = self.make_shell_environment(host=True)
        environment["FAIL_MODEL"] = models[1]
        result = subprocess.run(["bash", str(deploy / "compile.sh")], env=environment,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 7)
        rows = log.read_text().splitlines()
        self.assertEqual([row for row in rows if row.startswith("compile:")],
                         ["compile:" + model for model in models[:2]])
        self.assertFalse(any(row.startswith("rsync:") for row in rows))

    def test_full_entrypoints_upload_without_compile(self):
        """所有单模型全量入口自动上传参考,没有量化编译或数值参考采集命令."""
        for path in (ROOT / "models").rglob("deploy/evaluate.sh"):
            if path == ROOT / "models/deploy/evaluate.sh":
                continue
            source = path.read_text(encoding="utf-8")
            host = source.split("\nelse\n", 1)[1]
            with self.subTest(model=str(path)):
                self.assertIn("--upload-reference", host)
                self.assertNotIn("prepare_output_reference.py", host)
                self.assertNotRegex(host, r'python[^\n]*host/(compile|convert|export)')

    def test_all_full_reference_uploads(self):
        """全部模型的真实全量上传函数只发送必要参考,绑定实际 DLA."""
        from tools import deploy
        from tools.files import sha256

        os.environ["MTK_EVAL_SCOPE"] = "full"
        for model, (metric, unit) in TASKS.items():
            with self.subTest(model=model):
                folder = self.work / model
                work = folder / "work"
                output = folder / "assets"
                work.mkdir(parents=True)
                output.mkdir()
                language = unit == "question"
                package = output / "models" if language or model == "mit_bevfusion" else output
                package.mkdir(exist_ok=True)
                (package / "model.dla").write_bytes(b"compiled-test")
                backend = "pytorch" if language else "onnx"
                report = {"evaluation": evaluation.accuracy_result(model, backend, {metric: 0.8}, 1, unit)}
                if language:
                    mode = "mmlu1000" if model == "minicpm5_2b" else "full"
                    data = work / "evaluation" / mode
                    for name in ("mmlu/protocol.json", "mmlu/cases.json", "mmlu/mmlu_inputs.npz",
                                 "mmlu/board_input_contract.json", "quality/board_quality_protocol.json",
                                 "quality/board_inputs.npz", "quality/board_input_contract.json",
                                 "reference/mmlu/mmlu_accuracy.json", "quality/pytorch_quality.json"):
                        path = data / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(report))
                elif model == "yamnet":
                    path = work / "evaluation/full/onnx/summary.json"
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps(report))
                    (work / "fp32_reference.json").write_text("{}")
                else:
                    (work / "onnx_reference.json").write_text(json.dumps(report))
                    if model == "mit_bevfusion":
                        (work / "samples.json").write_text("{}")
                        (work / "ground_truth.json").write_text("{}")
                ctx = SimpleNamespace(work=work, output=output,
                                      config={"name": model, "board": {"deploy_dir": "/test-only"}})
                with mock.patch.object(deploy, "context", return_value=ctx), \
                        mock.patch.object(deploy, "transfer_files") as transfer, \
                        mock.patch.object(deploy, "deployment_artifacts", return_value={"model.dla": package / "model.dla"}):
                    deploy.upload_evaluation(folder / "deploy")
                transfer.assert_called_once()
                files = transfer.call_args.args[1]
                self.assertIn("build/reference/full.json", files)
                self.assertTrue(all(name.startswith("build/reference/") for name in files))
                self.assertFalse(any(path.suffix == ".dla" for path in files.values()))
                saved = json.loads(files["build/reference/full.json"].read_text())
                self.assertEqual(saved["dla_artifacts"], {"model.dla": sha256(package / "model.dla")})

    def test_full_publish_all_models(self):
        """真实全量发布器统一 13 个模型的精度区,保留子集报告且不混入数值误差."""
        from tools.summary import publish

        os.environ["MTK_EVAL_SCOPE"] = "full"
        for model, (metric, unit) in TASKS.items():
            with self.subTest(model=model):
                root = self.work / model
                reference_dir = root / "build/reference"
                reference_dir.mkdir(parents=True)
                backend = "pytorch" if unit == "question" else "onnx"
                baseline = evaluation.accuracy_result(model, backend, {metric: 0.8}, 1, unit)
                actual = evaluation.accuracy_result(model, "npu", {metric: 0.7}, 1, unit)
                (reference_dir / "full.json").write_text(json.dumps({"evaluation": baseline}))
                evaluation.publish_result(root, {"subset": True}, "summary_test.json")
                report = {"evaluation": actual, "accuracy": {"wikitext": {"fp32_ppl": 2, "npu_ppl": 3}}}
                ctx = SimpleNamespace(root=root, result=root / "build/work", config={"name": model})
                publish(ctx, report)
                saved = json.loads((root / "results/summary.json").read_text())
                self.assertEqual(saved["sample_unit"], unit)
                self.assertNotIn("output_errors", saved)
                self.assertEqual(saved["accuracy"][metric]["fp32_accuracy"], 0.8)
                self.assertEqual({path.name for path in (root / "results").iterdir()},
                                 {"summary.json", "summary_test.json"})

    def test_zero_vector_metrics(self):
        """零向量不能伪造 CosSim 或 NRMSE,有效 RMSE 仍正常记录."""
        metric = Moments()
        metric.update(np.zeros(2), np.ones(2))
        result = metric.result()
        self.assertIsNone(result["cosine_similarity"])
        self.assertIsNone(result["nrmse_percent"])
        self.assertAlmostEqual(result["rmse"], 1)

    def test_artifact_contracts_ignore_history(self):
        """全部模型只绑定契约内 DLA,历史文件不能污染参考或引发板端缺文件."""
        from tools.deploy import deployment_artifacts

        for model in TASKS:
            with self.subTest(model=model):
                output = self.work / model
                output.mkdir()
                config = {"name": model, "conversion": {"precision": "w8a16"}}
                if model in ("minicpm5_2b", "deepseek_r1_distill_qwen_1_5b"):
                    package = output / "models"
                    package.mkdir()
                    names = ["prefill.dla", "decode.dla"]
                    (package / "manifest.json").write_text(json.dumps({
                        "context": 256, "model_files": dict(zip(("prompt", "decode"), names))}))
                elif model == "mit_bevfusion":
                    package = output / "models"
                    package.mkdir()
                    names = ["fusion.dla"]
                    (package / "contract.json").write_text(json.dumps({"stages": {
                        "fusion": {"backend": "mtk_npu", "file": "fusion.dla"},
                        "camera": {"backend": "onnxruntime_cpu", "file": "camera.onnx"}}}))
                elif model == "whisper_tiny":
                    package = output
                    names = ["encoder.dla", "decoder.dla"]
                    (output / "build").mkdir()
                    (output / "build/runtime_contract.json").write_text(json.dumps({
                        "encoder": {"file": names[0]}, "decoder_step": {"file": names[1]}}))
                else:
                    package = output
                    names = ["model_w8a16.dla" if model == "yamnet" else "model_int8.dla"]
                for name in [*names, "obsolete.dla"]:
                    (package / name).write_bytes(b"test-artifact")
                result = deployment_artifacts(SimpleNamespace(output=output, config=config))
                self.assertEqual(set(result), set(names))

    def test_runtime_cleanup_boundaries(self):
        """真实清理脚本只删除 models 内旧文件,不碰结果、参考或链接外目标."""
        from tools import deploy

        root = self.work
        models = root / "models"
        models.mkdir()
        (models / "current.dla").write_bytes(b"current")
        (models / "old").mkdir()
        (models / "old/unused.dla").write_bytes(b"old")
        outside = root / "important.txt"
        outside.write_text("keep")
        ctx = SimpleNamespace(config={"board": {"deploy_dir": str(root)}})
        with mock.patch.dict(os.environ, {"MTK_BOARD_HOST": "test-only"}), \
                mock.patch.object(deploy, "run_command") as command:
            deploy.prune_models(ctx, ["current.dla"])
        arguments = shlex.split(command.call_args.args[0][-1])
        subprocess.run([sys.executable, *arguments[1:]], capture_output=True, check=True)
        self.assertFalse((models / "old").exists())
        self.assertTrue(outside.is_file())
        (models / "outside-link").symlink_to(outside)
        result = subprocess.run([sys.executable, *arguments[1:]], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(outside.is_file())

    def test_host_subset_stage_order_all_models(self):
        """真实子集编排对全部模型先完成浮点评分,再上传模型和参考."""
        from tools import compile_accuracy, deploy, process

        with mock.patch.dict(os.environ, {"MTK_BOARD_HOST": "test-only"}):
            for model, (metric, unit) in TASKS.items():
                with self.subTest(model=model):
                    folder = self.work / model
                    work = folder / "work"
                    work.mkdir(parents=True)
                    dla = folder / "model.dla"
                    dla.write_bytes(b"current")
                    backend = "pytorch" if unit == "question" else "onnx"
                    report = {"evaluation": evaluation.accuracy_result(model, backend, {metric: 0.8}, 1, unit),
                              "sample_ids_sha256": os.environ["MTK_EVAL_SAMPLE_SHA256"],
                              "cases_sha256": "cases", "identity": "identity", "sample_tokens": ["sample"]}
                    source = work / ("onnx_reference.json" if model == "mit_bevfusion" else "onnx_reference_test.json")
                    if model == "yamnet":
                        source = work / "evaluation/test/onnx/summary.json"
                        source.parent.mkdir(parents=True)
                    source.write_text(json.dumps(report))
                    quality = work / "quality.json"
                    quality.write_text(json.dumps({"perplexity": 2, "input_ids": [1], "scored_tokens": 1}))
                    ctx = SimpleNamespace(work=work, output=folder, repo=ROOT,
                                          config={"name": model, "board": {"deploy_dir": "/test-only"}})
                    events = []

                    def step(path, *arguments):
                        """记录独立步骤,不执行模型、量化或数据准备."""
                        events.append("numeric" if Path(path).name == "prepare_output_reference.py" else "float")

                    def command(arguments, **kwargs):
                        """记录上传顺序和工作目录环境,不连接远端."""
                        if len(arguments) > 1 and Path(str(arguments[1])).name == "upload.py":
                            events.append("upload")
                            self.assertEqual(kwargs["env"]["MTK_EVALUATION_WORK"], "0")
                        else:
                            events.append("reference")

                    with mock.patch.object(deploy, "context", return_value=ctx), \
                            mock.patch.object(deploy, "deployment_artifacts", return_value={"model.dla": dla}), \
                            mock.patch.object(compile_accuracy, "language", return_value=(source, quality, work)), \
                            mock.patch.object(compile_accuracy, "run", side_effect=step), \
                            mock.patch.object(process, "run_command", side_effect=command):
                        if unit == "question":
                            events.append("float")
                        compile_accuracy.host_reference(folder / "deploy")
                    self.assertLess(events.index("float"), events.index("upload"))
                    self.assertLess(events.index("upload"), events.index("reference"))
                    if model != "mit_bevfusion":
                        self.assertLess(events.index("upload"), events.index("numeric"))
                        self.assertLess(events.index("numeric"), events.index("reference"))

    def test_yamnet_actual_metric_coverage(self):
        """实际音频评分函数与统一接口覆盖相同的折和可映射类别."""
        source = ROOT / "models/audio/ambient_sound/yamnet/deploy/board/audio_utils.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("project_scores", "metrics", "evaluation_ids")]
        namespace = {"np": np}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        records = [{"target": target, "fold": fold, "filename": f"{fold}-{target}-{index}.wav"}
                   for target in range(50) for fold in range(1, 6) for index in range(8)]
        indices = [[] if target < 3 else [target] for target in range(50)]
        for scope, expected in (("subset", 188), ("full", 1504)):
            os.environ["MTK_EVAL_SCOPE"] = scope
            chosen = evaluation.audio_records(records)
            identifiers = namespace["evaluation_ids"](chosen, indices)
            actual = namespace["metrics"](np.zeros((len(chosen), 521), np.float32), chosen, indices)
            self.assertEqual(len(identifiers), expected)
            self.assertEqual(actual["held_out"]["samples"], expected)

    def test_bev_identity_ignores_unused_graphs(self):
        """真实模型标识只包含契约所用图和协议,目录残留不能破坏两端匹配."""
        source = ROOT / "models/pointcloud/three_d_fusion/mit_bevfusion/deploy/board/evaluate.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "identity")
        namespace = {"Path": Path, "hashlib": hashlib, "json": json, "__file__": str(source)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        (self.work / "contract.json").write_text(json.dumps({"stages": {
            "camera": {"backend": "onnxruntime_cpu", "file": "bevfusion-det_camera.onnx"}}}))
        for name in ("bevfusion-det_camera.onnx", "sparse_weights.npz", "geometry.npz"):
            (self.work / name).write_bytes(b"current")
        baseline = namespace["identity"](self.work)
        (self.work / "obsolete.onnx").write_bytes(b"old")
        self.assertEqual(namespace["identity"](self.work), baseline)


if __name__ == "__main__":
    unittest.main()
