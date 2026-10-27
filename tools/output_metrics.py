"""流式保存浮点输出参考,按同输入逐输出统计 CosSim、RMSE 和 NRMSE."""

import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import zlib

import numpy as np

from tools.evaluation import is_test
from tools.files import sha256


def input_signature(values):
    """按逻辑形状和 FP32 输入内容匹配样本,不使用文件遍历顺序配对."""
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value, dtype="<f4")
        if not np.isfinite(array).all():
            raise ValueError("输出比较的输入包含非有限数值.")
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


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


_writer = None


class ReferenceWriter:
    """用一个 SQLite 文件保存必要浮点输出,重复输入只保留一份张量."""

    def __init__(self, path):
        """创建新的参考库,未完整结束的库不能用于正式比较."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError("输出参考不能为符号链接.")
        self.db = sqlite3.connect(self.path)
        self.db.executescript("DROP TABLE IF EXISTS tensors; DROP TABLE IF EXISTS metadata;"
                             "CREATE TABLE metadata (value TEXT NOT NULL);"
                             "CREATE TABLE tensors (signature TEXT PRIMARY KEY, payload BLOB, calls INTEGER);")
        self.sources = {}
        self.artifacts = {}
        for directory in json.loads(os.environ.get("MTK_OUTPUT_MODEL_DIRS", "[]")):
            for artifact in sorted(Path(directory).glob("*.dla")):
                # 正式产物目录优先,不以 build 中可能残留的旧 DLA 覆盖它.
                if artifact.name not in self.artifacts:
                    self.artifacts[artifact.name] = sha256(artifact)
        self.calls = 0
        self.db.commit()

    def add(self, source, inputs, outputs):
        """记录同输入的后处理前浮点输出,保留来源图和实际调用数."""
        source = Path(source)
        if str(source) not in self.sources:
            self.sources[str(source)] = {"file": source.name, "sha256": sha256(source)}
        signature = input_signature(inputs)
        arrays = {str(name): np.asarray(value, dtype="<f4") for name, value in outputs.items()}
        if not arrays or any(not value.size or not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("浮点参考输出为空或含非有限数值.")
        stream = io.BytesIO()
        np.savez(stream, **arrays)
        payload = zlib.compress(stream.getvalue(), level=1)
        row = self.db.execute("SELECT payload FROM tensors WHERE signature=?", (signature,)).fetchone()
        if row is not None and row[0] != payload:
            raise ValueError("相同输入对应不同浮点输出,不能合并参考.")
        self.db.execute("INSERT INTO tensors VALUES (?, ?, 1) ON CONFLICT(signature) "
                        "DO UPDATE SET calls=calls+1", (signature, payload))
        self.calls += 1
        if self.calls % 32 == 0:
            self.db.commit()

    def finish(self):
        """仅成功完成参考生成后标记完整,并打印需要上传的唯一张量资源."""
        if not self.calls:
            raise ValueError("浮点输出参考为空.")
        self.db.execute("INSERT INTO metadata VALUES (?)", (json.dumps({
            "status": "complete", "calls": self.calls, "sources": list(self.sources.values()),
            "dla_artifacts": self.artifacts,
            "evaluation_scope": "full", "input_matching": "shape_and_fp32_content"}),))
        self.db.commit()
        self.db.close()
        print(f"[输出参考] {self.path}; {self.calls} 次输入,随同本次评价参考放入板端对应资源目录.", flush=True)


def record_reference(source, inputs, outputs):
    """仅由 evaluate.sh 启用参考记录,编译和独立推理不额外保存张量."""
    global _writer
    if os.environ.get("MTK_OUTPUT_METRICS") != "1" or is_test():
        return
    if _writer is None:
        _writer = ReferenceWriter(os.environ["MTK_OUTPUT_REFERENCE"])
    record = dict(outputs)
    if len(record) == 1:
        record = {"output": next(iter(record.values()))}
    _writer.add(source, inputs, record)


def finish_reference():
    """由成功的主机评价流程显式完成,异常退出不发布完整标志."""
    global _writer
    if _writer is not None:
        _writer.finish()
        _writer = None


class OutputComparison:
    """在已有 NPU 推理后比较浮点参考,统计时间不纳入纯 NPU 耗时."""

    def __init__(self, ctx, reference=None, scope="all_evaluation_inputs"):
        """缺参考如实记录,已有但损坏或不完整的参考直接拒绝."""
        self.ctx, self.scope = ctx, scope
        self.db, self.metadata = None, None
        self.metrics, self.seen = {}, {}
        self.enabled = os.environ.get("MTK_OUTPUT_METRICS") == "1" and not is_test()
        if not self.enabled:
            return
        path = Path(reference or ctx.root / "models/output_reference.sqlite")
        self.reference_path = path
        if path.is_file():
            self.db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            rows = self.db.execute("SELECT value FROM metadata").fetchall()
            if len(rows) != 1:
                raise ValueError("浮点输出参考未完成,请重新准备.")
            self.metadata = json.loads(rows[0][0])
            if self.metadata.get("status") != "complete" or self.metadata.get("evaluation_scope") != "full":
                raise ValueError("不能使用子集或未完成的浮点输出参考.")
            # 已部署源图可用时核对其版本,其余模型由既有参考协议核对来源.
            for source in self.metadata["sources"]:
                candidate = ctx.root / "models" / source["file"]
                if candidate.is_file() and sha256(candidate) != source["sha256"]:
                    raise ValueError("输出参考与已部署源图版本不一致.")
            artifacts = self.metadata.get("dla_artifacts", {})
            matched = False
            for name, digest in artifacts.items():
                candidate = ctx.root / "models" / name
                if candidate.is_file():
                    if sha256(candidate) != digest:
                        raise ValueError("输出参考与已部署 DLA 版本不一致.")
                    matched = True
            if reference is None and not matched:
                raise ValueError("输出参考缺少与当前 DLA 的关联,请使用已有编译产物重新准备参考.")

    def update(self, inputs, outputs):
        """以量化前输入内容配对,按语义名称或唯一形状匹配各输出头."""
        if not self.enabled or self.db is None:
            return
        signature = input_signature(inputs)
        row = self.db.execute("SELECT payload,calls FROM tensors WHERE signature=?", (signature,)).fetchone()
        if row is None:
            raise ValueError("找不到同输入浮点输出参考,请核对预处理、模型和参考范围.")
        self.seen[signature] = self.seen.get(signature, 0) + 1
        if self.seen[signature] > row[1]:
            raise ValueError("NPU 输入重复次数超过浮点参考.")
        actual = dict(outputs)
        if len(actual) == 1:
            actual = {"output": next(iter(actual.values()))}
        with np.load(io.BytesIO(zlib.decompress(row[0])), allow_pickle=False) as archive:
            if len(actual) != len(archive.files):
                raise ValueError("浮点与 NPU 输出数量不一致.")
            used = set()
            for name, value in actual.items():
                candidates = ([name] if name in archive.files else
                              [key for key in archive.files if archive[key].shape == np.shape(value)])
                if len(candidates) != 1 or candidates[0] in used:
                    raise ValueError(f"无法唯一匹配输出头: {name}.")
                key = candidates[0]
                used.add(key)
                self.metrics.setdefault(key, Moments()).update(archive[key], value)

    def finish(self):
        """核对完整输入覆盖,将汇总写本模型 work 供唯一正式报告读取."""
        if not self.enabled:
            return
        if self.db is None:
            result = {"status": "unavailable", "scope": self.scope,
                      "reason": f"缺少同输入浮点输出参考 {self.reference_path.name}", "outputs": {}}
        else:
            expected = dict(self.db.execute("SELECT signature,calls FROM tensors"))
            if self.seen != expected or sum(self.seen.values()) != self.metadata["calls"]:
                raise ValueError("输出误差比较未覆盖完整参考输入,拒绝发布部分统计.")
            result = {"status": "complete", "scope": self.scope,
                      "reference": self.metadata, "outputs": {
                          name: metric.result() for name, metric in self.metrics.items()}}
            self.db.close()
        result["nrmse_definition"] = "100 * RMSE / RMS(reference), percent"
        result["aggregation"] = "per_output_global_element_moments; no_cross_output_average"
        self.ctx.work.mkdir(parents=True, exist_ok=True)
        (self.ctx.work / "output_metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"[输出误差] {json.dumps(result['outputs'], ensure_ascii=False)}; {result['status']}", flush=True)


def report_metrics(ctx):
    """发布报告时只读取当前运行的指标,缺失项不沿用历史数值."""
    path = ctx.work / "output_metrics.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"status": "unavailable", "outputs": {}, "reason": "本次未计算输出误差"}
