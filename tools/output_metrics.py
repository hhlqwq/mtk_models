"""流式保存浮点输出参考,按样本标识逐输出统计 CosSim、RMSE 和 NRMSE."""

import json
import math
import os
from pathlib import Path

import numpy as np

from tools.evaluation import is_test
from tools.files import sha256


def input_contract(values):
    """核对输入形状、类型和有限值,不比较跨平台预处理的逐字节结果."""
    result = []
    for value in values:
        array = np.asarray(value)
        if not array.size or not np.isfinite(array).all():
            raise ValueError("输出比较的输入为空或包含非有限数值.")
        result.append({"shape": list(array.shape), "dtype": array.dtype.str})
    if not result:
        raise ValueError("输出比较必须提供输入张量.")
    return result


def sample_files_identity(paths, cache):
    """仅对选定样本核对原始文件内容,同一文件在当前步骤只读取一次."""
    if not paths:
        raise ValueError("数值对比必须提供原始样本文件.")
    result = []
    for value in paths:
        path = Path(value)
        if path not in cache:
            cache[path] = {"file": path.name, "sha256": sha256(path)}
        result.append(cache[path])
    return result


class Moments:
    """按输出头累加充分统计量,不保存 NPU 全量张量或逐样本指标."""

    def __init__(self):
        """使用 FP64 累加点积、能量与平方误差."""
        self.count = self.calls = 0
        self.error = self.dot = self.reference = self.actual = 0.0

    def update(self, reference, actual):
        """核对形状和有限值后更新,允许零向量但不伪造未定义指标."""
        reference = np.asarray(reference, dtype=np.float64)
        actual = np.asarray(actual, dtype=np.float64)
        if (reference.shape != actual.shape or not reference.size
                or not np.isfinite(reference).all() or not np.isfinite(actual).all()):
            raise ValueError("输出比较形状不一致、为空或包含非有限数值.")
        reference, actual = reference.ravel(), actual.ravel()
        difference = actual - reference
        self.error += float(np.dot(difference, difference))
        self.dot += float(np.dot(reference, actual))
        self.reference += float(np.dot(reference, reference))
        self.actual += float(np.dot(actual, actual))
        self.count += reference.size
        self.calls += 1

    def result(self):
        """按该输出的全部元素计算指标,NRMSE 使用参考 RMS 作分母."""
        if not self.count:
            raise ValueError("没有可统计的输出元素.")
        if not all(math.isfinite(value) for value in (
                self.error, self.dot, self.reference, self.actual)):
            raise ValueError("输出统计量溢出.")
        cosine = (max(-1.0, min(1.0, self.dot / math.sqrt(self.reference)
                               / math.sqrt(self.actual)))
                  if self.reference > 0 and self.actual > 0 else None)
        return {"cosine_similarity": cosine, "rmse": math.sqrt(self.error / self.count),
                "nrmse_percent": (100 * math.sqrt(self.error / self.reference)
                                  if self.reference > 0 else None),
                "elements": self.count, "calls": self.calls,
                "unavailable": (["CosSim: 参考或实际输出为零向量"] if cosine is None else [])
                + (["NRMSE: 参考输出 RMS 为零"] if self.reference == 0 else [])}


class ReferenceComplete(Exception):
    """固定小样本参考已收集完成,立即停止复用的浮点评价循环."""


_writer = None


class ReferenceWriter:
    """将少量浮点输出保存到单个 NPZ,禁止在全量评价时收集."""

    def __init__(self, path, limit=5):
        """只在主机本地过程目录创建小样本参考,不访问数据库."""
        if not is_test() or not 1 <= limit <= 5:
            raise ValueError("数值参考仅用于 compile 小样本,最多 5 次输入.")
        self.path = Path(path)
        self.limit = limit
        self.arrays, self.samples, self.sources = {}, {}, {}
        self.calls = 0
        self.file_cache = {}
        self.artifacts = {}
        files = json.loads(os.environ["MTK_OUTPUT_DLA_FILES"])
        if not files:
            raise ValueError("数值参考必须绑定当前部署契约中的 DLA.")
        self.artifacts = {name: sha256(Path(path)) for name, path in files.items()}

    def add(self, source, inputs, outputs, *, sample_id, sample_files):
        """按明确样本标识记录输出,保留原图及张量契约检查."""
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("数值参考必须提供非空样本标识.")
        if sample_id in self.samples:
            raise ValueError(f"数值参考样本标识重复: {sample_id}.")
        source = Path(source)
        if str(source) not in self.sources:
            self.sources[str(source)] = {"file": source.name, "sha256": sha256(source)}
        if len(outputs) == 1:
            outputs = {"output": next(iter(outputs.values()))}
        if not outputs:
            raise ValueError("浮点参考输出不能为空.")
        entry = {"calls": 1, "inputs": input_contract(inputs),
                 "files": sample_files_identity(sample_files, self.file_cache), "outputs": {}}
        for name, value in outputs.items():
            array = np.asarray(value, dtype="<f4")
            if not array.size or not np.isfinite(array).all():
                raise ValueError("浮点参考输出为空或包含非有限值.")
            key = f"tensor_{len(self.arrays)}"
            self.arrays[key] = array
            entry["outputs"][name] = key
        self.samples[sample_id] = entry
        self.calls += 1
        print(f"[小样本数值参考] {self.calls}/{self.limit}", flush=True)
        if self.calls >= self.limit:
            raise ReferenceComplete()

    def finish(self):
        """仅发布完整的小样本数组,不发布任务精度报告."""
        if not self.calls:
            raise ValueError("小样本浮点参考为空.")
        metadata = {"schema_version": 2, "status": "complete", "evaluation_scope": "compile_subset",
                    "calls": self.calls, "samples": self.samples,
                    "sources": list(self.sources.values()), "dla_artifacts": self.artifacts}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, metadata=np.array(json.dumps(metadata)), **self.arrays)
        print(f"[小样本参考] {self.calls} 次输入: {self.path}", flush=True)


def record_reference(source, inputs, outputs, *, sample_id, sample_files):
    """仅由 compile 的独立小样本参考步骤启用,全量评价立即返回."""
    global _writer
    if not is_test() or os.environ.get("MTK_PREPARE_OUTPUT_REFERENCE") != "1":
        return
    if _writer is None:
        _writer = ReferenceWriter(os.environ["MTK_OUTPUT_REFERENCE"])
    _writer.add(source, inputs, outputs, sample_id=sample_id, sample_files=sample_files)


def finish_reference():
    """小样本参考成功结束后关闭收集,全量评价不产生任何张量参考."""
    global _writer
    if _writer is not None:
        _writer.finish()
        _writer = None


class OutputComparison:
    """只在 compile 固定小样本中对照,全量评价不加载参考或计算误差."""

    def __init__(self, ctx, reference=None, scope="compile_fixed_subset"):
        """绑定本次小样本参考与 DLA,缺失时明确提示未完成."""
        self.ctx, self.scope = ctx, scope
        self.file_cache = {}
        self.enabled = is_test() and os.environ.get("MTK_OUTPUT_METRICS") == "1"
        self.metrics, self.seen, self.metadata, self.archive = {}, {}, None, None
        if not self.enabled:
            return
        self.path = Path(reference or ctx.root / "build/reference/output_reference.npz")
        if not self.path.is_file():
            raise FileNotFoundError("缺少 compile 数值参考,请先完成主机 compile.")
        self.archive = np.load(self.path, allow_pickle=False)
        self.metadata = json.loads(str(self.archive["metadata"]))
        if self.metadata.get("schema_version") != 2:
            self.archive.close()
            raise ValueError("数值参考为旧格式,请用当前主机脚本重新准备参考并上传.")
        if self.metadata.get("evaluation_scope") != "compile_subset":
            self.archive.close()
            raise ValueError("只接受 compile 固定小样本参考.")
        for name, digest in self.metadata["dla_artifacts"].items():
            artifact = ctx.models / name
            if not artifact.is_file() or sha256(artifact) != digest:
                raise ValueError("小样本参考与板端 DLA 不匹配.")

    def update(self, inputs, outputs, *, sample_id, sample_files):
        """按样本标识配对,允许预处理舍入差异,严格核对原始文件与张量契约."""
        if not self.enabled or self.archive is None:
            return
        entry = self.metadata["samples"].get(sample_id)
        if entry is None:
            return
        if sample_id in self.seen:
            raise ValueError(f"数值比较样本重复: {sample_id}.")
        if input_contract(inputs) != entry["inputs"]:
            raise ValueError(f"数值比较输入张量契约不符: {sample_id}.")
        if sample_files_identity(sample_files, self.file_cache) != entry["files"]:
            raise ValueError(f"数值比较原始样本文件不一致: {sample_id}.")
        actual = dict(outputs)
        if len(actual) == 1:
            actual = {"output": next(iter(actual.values()))}
        expected = entry["outputs"]
        if len(actual) != len(expected):
            raise ValueError("浮点与 NPU 输出数量不一致.")
        used = set()
        for name, value in actual.items():
            candidates = [name] if name in expected else [key for key, item in expected.items()
                if self.archive[item].shape == np.shape(value)]
            if len(candidates) != 1 or candidates[0] in used:
                raise ValueError(f"无法唯一匹配小样本输出: {name}.")
            key = candidates[0]
            used.add(key)
            self.metrics.setdefault(key, Moments()).update(self.archive[expected[key]], value)
        self.seen[sample_id] = 1

    def finish(self):
        """在终端显示实际覆盖和误差,不写入正式全量评价报告."""
        if not self.enabled:
            return
        if self.archive is None:
            raise FileNotFoundError("缺少本次 compile 的数值参考.")
        try:
            expected = {key: entry["calls"] for key, entry in self.metadata["samples"].items()}
            if self.seen != expected:
                missing = sorted(set(expected) - set(self.seen))
                raise ValueError(f"compile 数值检查缺少选定样本: {missing}.")
            print(f"[输出误差] {self.scope}, {sum(self.seen.values())} 次输入; "
                  "NRMSE=100*RMSE/RMS(reference); 包含两端预处理及模型执行差异.", flush=True)
            result = {"scope": self.scope, "compared_calls": sum(self.seen.values()),
                      "comparison_basis": "deployment_output_including_preprocessing",
                      "sample_ids": list(self.seen),
                      "nrmse_definition": "100 * RMSE / RMS(reference)",
                      "outputs": {name: metric.result() for name, metric in self.metrics.items()}}
            self.ctx.work.mkdir(parents=True, exist_ok=True)
            (self.ctx.work / "output_metrics.json").write_text(
                json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            for name, values in result["outputs"].items():
                print(f"[输出误差] {name}: {json.dumps(values, ensure_ascii=False)}", flush=True)
        finally:
            self.archive.close()
