"""compile 两端评价同一子集,精度损失写入独立 summary_test.json."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from tools.evaluation import compare_accuracy, is_test, publish_result, read_accuracy
from tools.files import read_json, sha256


LLMS = ("minicpm5_2b", "deepseek_r1_distill_qwen_1_5b")


def subset_report(reference, actual=None, quality=None):
    """统一子集参考和板端报告,正的损失始终表示相对浮点变差."""
    accuracy = {}
    for name, baseline in reference["metrics"].items():
        value = None if actual is None else actual[name]
        if not math.isfinite(baseline) or (value is not None and not math.isfinite(value)):
            raise ValueError("子集精度包含无效数值.")
        higher_is_better = name != "WER"
        loss = None if value is None else (
            baseline - value if higher_is_better else value - baseline) * 100
        accuracy[name] = {"fp32_accuracy": baseline, "npu_accuracy": value,
                          "higher_is_better": higher_is_better,
                          "accuracy_loss_percentage_points": loss}
    result = {"model": reference["model"], "evaluation_scope": "subset",
              "status": "float_reference_ready" if actual is None else "complete",
              "samples": reference["samples"], "accuracy": accuracy,
              "sample_unit": reference["evaluation"]["samples"]["unit"],
              "sampling": "fixed_task_subset_v2",
              "sample_ids_sha256": reference.get("sample_ids_sha256"),
              "dla_artifacts": reference["dla_artifacts"],
              "loss_definition": "正值表示 NPU 相对浮点精度下降,负值表示改善"}
    for key in ("cases_sha256", "sample_tokens", "identity"):
        if key in reference:
            result[key] = reference[key]
    if "quality" in reference:
        baseline = reference["quality"]["perplexity"]
        value = None if quality is None else quality["perplexity"]
        if not math.isfinite(baseline) or baseline <= 0 or (
                value is not None and (not math.isfinite(value) or value <= 0)):
            raise ValueError("子集 PPL 包含无效数值.")
        result["wikitext"] = {"scored_tokens": reference["quality"]["scored_tokens"],
            "fp32_perplexity": baseline, "npu_perplexity": value,
            "perplexity_increase": None if value is None else value - baseline,
            "perplexity_increase_percent": None if value is None else (value / baseline - 1) * 100}
    return result


def save_subset(root, report):
    """只覆盖本模型的子集摘要,不改写或清理正式全量摘要."""
    destination = publish_result(root, report, "summary_test.json")
    print(f"[子集报告] {destination}", flush=True)


def run(path, *arguments):
    """直接运行已有独立步骤,保留原始进度与错误."""
    subprocess.run([sys.executable, str(path), *map(str, arguments)], check=True)


def language(ctx, host):
    """复用同协议 MMLU 和 WikiText2 实现,仅准备和评价固定子集."""
    from tools.llm import evaluate_mmlu

    name = ctx.config["name"]
    if host:
        from tools.files import single_file
        from tools.llm import evaluate_quality
        from tools.llm.native import settings

        native = settings(ctx, python_executable=sys.executable)
        graph = single_file(native.quantized.with_name(
            f"{native.prefix}_{native.prefill}t{native.context}c"), "*.tflite")
        manifest = read_json(ctx.output / "models/manifest.json")
        if sha256(graph) != manifest["graph_contracts"]["prompt"]["tflite_sha256"]:
            raise ValueError("子集参考与已编译 LLM 静态图不匹配.")
        data = ctx.work / "evaluation/compile"
        evaluate_mmlu.prepare(SimpleNamespace(
            dataset=ctx.dataset / ctx.config["data"]["mmlu_test"], model=native.model,
            source_model=native.source, graph=graph, context=native.context,
            limit=ctx.config["evaluation"]["mmlu_questions"],
            seed=ctx.config["evaluation"]["mmlu_seed"],
            test_questions=ctx.config["evaluation"].get("test_mmlu_questions", 100),
            output=data / "mmlu"))
        evaluate_mmlu.evaluate(SimpleNamespace(
            model_name=name, backend="pytorch", data=data / "mmlu", model=native.model,
            output=data / "reference/mmlu", diagnostic_questions=0,
            device=ctx.config["evaluation"]["reference_device"]))
        options = SimpleNamespace(
            model=native.model, corpus=ctx.dataset / ctx.config["data"]["wikitext_test"],
            output=data / "quality", backend="pytorch", tflite=graph.parent,
            embedding=native.quantized / "embedding_int16.bin",
            decode_tflite=None, text_rows=128, full_test=False, blocks=None,
            diagnostic_blocks=ctx.config["evaluation"].get("test_wikitext_blocks", 16),
            decode_steps=0, skip_demo=True, device=ctx.config["evaluation"]["reference_device"],
            cpu_threads=8)
        evaluate_quality.evaluate(options)
        options.backend = "prepare_board"
        evaluate_quality.evaluate(options)
        return data / "reference/mmlu/mmlu_accuracy.json", data / "quality/pytorch_quality.json", data
    from tools.llm import evaluate_board_quality

    data = ctx.root / "build/reference/subset"
    output = ctx.work / "compile_accuracy"
    evaluate_mmlu.evaluate(SimpleNamespace(
        model_name=name, backend="npu", data=data / "mmlu", package=ctx.models,
        output=output / "mmlu", diagnostic_questions=0))
    evaluate_board_quality.evaluate(SimpleNamespace(
        package=ctx.models, quality_data=data / "quality", output=output / "quality",
        diagnostic_blocks=0, decode_steps=0))
    return output / "mmlu/mmlu_accuracy.json", output / "quality/npu_quality.json", data


def board_report(ctx):
    """复用已有板端评分步骤,所有产物均留在本模型 work."""
    name = ctx.config["name"]
    board = ctx.root / "board"
    if name in LLMS:
        report, quality, _ = language(ctx, False)
        return read_json(report), read_json(quality)
    if name == "mit_bevfusion":
        run(board / "evaluate.py")
        run(board / "check.py")
        return read_json(ctx.work / "compile_accuracy.json"), None
    run(board / ("evaluate.py" if name == "resnet101" else "full_accuracy_board.py"))
    report = (ctx.work / "coco_wholebody_metrics.json" if name == "rtmpose_body2d"
              else ctx.work / "summary.json" if name in ("resnet101", "yolov5s", "yolov8n", "yamnet")
              else ctx.work / "report/summary.json")
    return read_json(report), None


def host_reference(deploy):
    """在主机本地计算子集浮点任务精度并上传必要参考,不写 NAS."""
    from tools.deploy import context, deployment_artifacts, reference_path
    from tools.process import run_command

    os.environ.update(MTK_EVALUATION_WORK="1", MTK_COMPILE_ACCURACY="1",
                      MTK_PREPARE_OUTPUT_REFERENCE="0")
    ctx = context(deploy / "host/evaluate_onnx.py")
    name = ctx.config["name"]
    quality, data = None, None
    if name in LLMS:
        path, quality_path, data = language(ctx, True)
        report, quality = read_json(path), read_json(quality_path)
    elif name == "mit_bevfusion":
        run(deploy / "host/prepare.py", "--evaluate")
        run(deploy / "host/reference.py")
        report = read_json(ctx.work / "onnx_reference.json")
    else:
        arguments = ["--onnx", ctx.output / "yolov8n.onnx"] if name == "yolov8n" else []
        run(deploy / "host/evaluate_onnx.py", *arguments)
        path = (ctx.work / "evaluation/test/onnx/summary.json" if name == "yamnet"
                else reference_path(ctx))
        report = read_json(path)
    backend = "pytorch" if name in LLMS else "onnx"
    evaluation = read_accuracy(report, name, backend)
    values = {key: item["value"] for key, item in evaluation["metrics"].items()}
    count = evaluation["samples"]["count"]
    artifacts = deployment_artifacts(ctx)
    reference = {"evaluation": evaluation, "model": name, "evaluation_scope": "subset", "samples": count,
                 "metrics": values, "sample_ids_sha256": report.get("sample_ids_sha256"),
                 "dla_artifacts": {name: sha256(path) for name, path in artifacts.items()}}
    if quality is not None:
        reference["quality"] = {key: quality[key] for key in ("perplexity", "input_ids", "scored_tokens")}
        reference["cases_sha256"] = report["cases_sha256"]
    if name == "mit_bevfusion":
        reference.update(identity=report["identity"], sample_tokens=report["sample_tokens"])
    if not reference["dla_artifacts"]:
        raise FileNotFoundError("子集参考缺少已编译 DLA.")
    print(f"[浮点子集完成] {name}: {count} 个评价单位, {values}.", flush=True)
    # 浮点评分和契约核对通过后才上传部署包,不让半完成模型进入下一阶段.
    upload_environment = dict(os.environ, MTK_EVALUATION_WORK="0",
                              MTK_COMPILE_ACCURACY="0", MTK_PREPARE_OUTPUT_REFERENCE="0")
    run_command([sys.executable, deploy / "host/upload.py"], env=upload_environment)
    if name != "mit_bevfusion":
        run(ctx.repo / "tools/prepare_output_reference.py", deploy)
    destination = ctx.config["board"]["deploy_dir"]
    host = os.environ["MTK_BOARD_HOST"]
    target = ctx.work / "compile_accuracy_reference.json"
    target.write_text(json.dumps(reference, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    run_command(["scp", "-o", "BatchMode=yes", target, f"{host}:{destination}/build/reference/subset.json"])
    if name == "mit_bevfusion":
        for source, filename in ((ctx.work / "samples.json", "compile_samples.json"),
                                 (ctx.work / "ground_truth.json", "compile_ground_truth.json")):
            run_command(["scp", "-o", "BatchMode=yes", source,
                         f"{host}:{destination}/build/reference/{filename}"])
    if data is not None:
        import shlex

        for folder, filenames in (
            ("mmlu", ("protocol.json", "cases.json", "mmlu_inputs.npz", "board_input_contract.json")),
            ("quality", ("board_quality_protocol.json", "board_inputs.npz", "board_input_contract.json")),
        ):
            target_folder = destination + "/build/reference/subset/" + folder
            run_command(["ssh", "-o", "BatchMode=yes", host,
                         "mkdir -p " + shlex.quote(target_folder)])
            run_command(["scp", "-o", "BatchMode=yes", *[data / folder / name for name in filenames],
                         f"{host}:{target_folder}/"])
    print(f"[子集参考完成] {name}: {count} 个评价单位, {values}; 全量报告未改写.", flush=True)


def main():
    """主机准备子集参考,板端计算精度损失并保存独立子集报告."""
    if not is_test():
        raise ValueError("compile 子集评分不能从全量入口调用.")
    if len(sys.argv) > 1:
        deploy = Path(sys.argv[1]).resolve()
        sys.argv = [sys.argv[0]]
        host_reference(deploy)
        return
    from tools.board import context

    ctx = context(__file__)
    name = ctx.config["name"]
    reference = read_json(ctx.root / "build/reference/subset.json")
    if reference["model"] != name or reference["evaluation_scope"] != "subset":
        raise ValueError("子集精度参考的模型或范围不匹配.")
    for filename, digest in reference["dla_artifacts"].items():
        if sha256(ctx.models / filename) != digest:
            raise ValueError("子集任务精度参考与板端 DLA 不匹配.")
    report, quality = board_report(ctx)
    evaluation = read_accuracy(report, name, "npu")
    actual = {key: item["value"] for key, item in evaluation["metrics"].items()}
    count = evaluation["samples"]["count"]
    if evaluation["samples"] != reference["evaluation"]["samples"]:
        raise ValueError("浮点与 NPU 的实际样本集合、数量或单位不一致.")
    if count != reference["samples"] or actual.keys() != reference["metrics"].keys():
        raise ValueError("浮点与 NPU 子集评价的覆盖或指标不一致.")
    if name in LLMS:
        if report["cases_sha256"] != reference["cases_sha256"]:
            raise ValueError("MMLU 子集题目不一致.")
        if any(quality[key] != reference["quality"][key] for key in ("input_ids", "scored_tokens")):
            raise ValueError("WikiText2 子集 Token 覆盖不一致.")
    elif name == "mit_bevfusion":
        if any(report[key] != reference[key] for key in ("identity", "sample_tokens")):
            raise ValueError("BEVFusion 子集模型或帧列表不一致.")
    elif (not reference.get("sample_ids_sha256")
          or report.get("sample_ids_sha256") != reference["sample_ids_sha256"]):
        raise ValueError("浮点与 NPU 子集样本不一致,不能计算精度差值.")
    for metric, value in actual.items():
        baseline = reference["metrics"][metric]
        if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in (value, baseline)):
            raise ValueError("子集精度含无效数值.")
        loss = (value - baseline if metric == "WER" else baseline - value) * 100
        print(f"[子集精度] {name} {count} 个评价单位, {metric}: "
              f"浮点 {baseline:.6f}, NPU {value:.6f}, "
              f"精度损失 {loss:+.4f} 个百分点(正值变差,负值改善).", flush=True)
    if quality is not None:
        if not all(math.isfinite(value) and value > 0 for value in (
                quality["perplexity"], reference["quality"]["perplexity"])):
            raise ValueError("子集 PPL 含无效数值.")
        print(f"[子集精度] WikiText2 {quality['scored_tokens']} 个 Token, PPL: "
              f"浮点 {reference['quality']['perplexity']:.6f}, NPU {quality['perplexity']:.6f}, "
              f"变化(NPU - 浮点) {quality['perplexity'] - reference['quality']['perplexity']:+.6f}.", flush=True)
    result = subset_report(reference, actual, quality)
    result["accuracy"] = compare_accuracy(reference["evaluation"], evaluation)
    result["sample_ids_sha256"] = evaluation["samples"]["selection_id"]
    result["run_id"] = ctx.run_id
    result["output_errors"] = read_json(ctx.work / "output_metrics.json")
    save_subset(ctx.root, result)
    print("[子集精度完成] 已保存 summary_test.json,未改写全量 summary.json.", flush=True)


if __name__ == "__main__":
    main()
