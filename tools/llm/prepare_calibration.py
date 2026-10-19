"""扩展独立训练集校准覆盖,保留官方模型与测试数据不变."""

import argparse
import json
from pathlib import Path
import random

import pyarrow.parquet as parquet
from transformers import AutoTokenizer

from tools.files import sha256


TRAIN_SHA256 = "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"


def prepare(work, model, train, segments=64, tokens=512, seed=17):
    """按固定种子采样训练窗口,更新校准来源记录并作废旧完成标记."""
    if segments < 1 or tokens < 128:
        raise ValueError("校准段数必须为正,每段至少 128 Token.")
    if sha256(train) != TRAIN_SHA256:
        raise ValueError("WikiText2 训练集哈希不符.")
    config = json.loads((work / model / "config.json").read_text())
    if tokens > config["max_position_embeddings"]:
        raise ValueError("校准窗口超过转换配置的上下文长度.")
    print("[校准输入] 读取已校验的 WikiText2 train.", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        work / model, local_files_only=True)
    table = parquet.read_table(train)
    text = "\n\n".join(table.column("text").to_pylist())
    train_tokens = tokenizer.encode(text, add_special_tokens=False)
    starts = range(0, len(train_tokens) - tokens + 1, tokens)
    if len(starts) < segments:
        raise ValueError("训练集完整窗口不足.")
    selected = random.Random(seed).sample(starts, segments)
    calibration = work / "calibration.jsonl"
    original = [json.loads(line) for line in calibration.read_text().splitlines()
                if line.strip()]
    # 保留原准备脚本生成的短对话,排除此前已混入的 256 Token 训练段.
    authored = [record for record in original
                if len(record["tokens"].split()) < 256]
    records = [{"tokens": " ".join(map(str, train_tokens[start:start + tokens]))}
               for start in selected]
    records.extend(authored)
    backup = work / "calibration_before_expansion.jsonl"
    if backup.exists():
        raise ValueError("该目录已扩展过校准输入,请使用独立候选目录.")
    manifest_path = work / "native_prepare.json"
    manifest = json.loads(manifest_path.read_text())
    backup.write_bytes(calibration.read_bytes())
    (work / "calibration_complete.txt").unlink(missing_ok=True)
    calibration.write_text("".join(json.dumps(record) + "\n"
                                   for record in records))
    source = {"revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
              "split": "train", "sha256": TRAIN_SHA256, "seed": seed,
              "segments": segments, "segment_tokens": tokens,
              "start_token_offsets": selected,
              "authored_records": len(authored)}
    manifest.update(calibration_training_source=source,
                    calibration_scope="wikitext2_train_sampled_and_authored",
                    calibration_samples=len(records),
                    calibration_sha256=sha256(calibration))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False,
                                        indent=2) + "\n")
    print(f"[校准输入完成] {segments} 段训练文本 x {tokens} Token,"
          f"保留 {len(authored)} 条短对话.", flush=True)


def main():
    """解析独立候选目录与已有训练集路径."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--segments", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    prepare(args.work, args.model, args.train, args.segments, args.tokens,
            args.seed)


if __name__ == "__main__":
    main()
