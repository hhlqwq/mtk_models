"""从 Google 原始权重导出固定窗口 ONNX,完成同协议浮点评测."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from audio_utils import load_records, load_waveform, mapping, metrics, preprocess, stream_patches


def original_models(models_dir):
    """加载锁定的 Google Keras 实现,构造原始音频和固定窗口模型."""
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    import tensorflow as tf
    import tf_keras
    sys.path.insert(0, str(models_dir / "upstream"))
    import params
    import yamnet
    parameters = params.Params()
    frames = yamnet.yamnet_frames_model(parameters)
    frames.load_weights(str(models_dir / "yamnet.h5"))
    inputs = tf_keras.layers.Input(shape=(1, 96, 64), batch_size=1, name="audio")
    features = tf_keras.layers.Permute((2, 3, 1))(inputs)
    scores, _ = yamnet.yamnet(features, parameters)
    patches = tf_keras.Model(inputs, scores)
    patches.load_weights(str(models_dir / "yamnet.h5"))
    return frames, patches


def export_model(args):
    """比较原始前向、NumPy 前处理和分块窗口,再导出固定 ONNX."""
    import tensorflow as tf
    import tf2onnx
    frames, model = original_models(args.models_dir)
    records = load_records(args.dataset)
    maximum_feature_error = 0.0
    maximum_score_error = 0.0
    for row in records[:10]:
        waveform = load_waveform(args.dataset / "audio" / row["filename"])
        expected, _, mel = frames(waveform)
        patches = preprocess(waveform)
        chunks = [waveform[start:start + 1600] for start in range(0, len(waveform), 1600)]
        streamed = np.stack(list(stream_patches(chunks)))
        np.testing.assert_allclose(streamed, patches, atol=2e-5, rtol=2e-5)
        reference = np.stack([mel.numpy()[start:start + 96]
                              for start in range(0, len(mel) - 95, 48)])[:, None]
        np.testing.assert_allclose(patches, reference, atol=2e-4, rtol=2e-4)
        predicted = np.concatenate([model(patch[None]).numpy() for patch in patches])
        np.testing.assert_allclose(predicted, expected.numpy(), atol=2e-5, rtol=2e-4)
        maximum_feature_error = max(maximum_feature_error,
                                    float(np.max(np.abs(patches - reference))))
        maximum_score_error = max(maximum_score_error,
                                  float(np.max(np.abs(predicted - expected.numpy()))))
    args.models_dir.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_keras(
        model, input_signature=[tf.TensorSpec((1, 1, 96, 64), tf.float32, name="audio")],
        opset=13, output_path=str(args.models_dir / "model_fp32.onnx"))
    (args.models_dir / "export_check.json").write_text(json.dumps({
        "samples": 10, "streaming_chunk_samples": 1600,
        "streaming_windows_match": True, "maximum_feature_error": maximum_feature_error,
        "maximum_original_score_error": maximum_score_error}, indent=2))
    print("[OK] 原始模型与固定窗口前向一致,ONNX 已导出.", flush=True)


def evaluate(args):
    """对全量 2000 条音频求窗口平均分数,保存结果和校准折外指标."""
    records = load_records(args.dataset)
    indices = mapping(args.models_dir / "upstream/yamnet_class_map.csv")
    if args.mode == "tensorflow":
        frames, _ = original_models(args.models_dir)
    else:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(args.models_dir / "model_fp32.onnx"),
                                       options, providers=["CPUExecutionProvider"])
    all_scores = []
    for row in tqdm(records, desc=f"YAMNet {args.mode} 全量精度"):
        waveform = load_waveform(args.dataset / "audio" / row["filename"])
        if args.mode == "tensorflow":
            patch_scores = frames(waveform)[0].numpy()
        else:
            patch_scores = np.concatenate([
                session.run(None, {"audio": patch[None]})[0]
                for patch in preprocess(waveform)])
        all_scores.append(patch_scores.mean(axis=0))
    scores = np.stack(all_scores)
    if scores.shape != (2000, 521) or not np.isfinite(scores).all():
        raise ValueError("浮点分数覆盖不完整或含非有限数.")
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "scores.npy", scores)
    summary = {"backend": args.mode, "metric": "ESC-50 projected macro AP",
               "metrics": metrics(scores, records, indices)}
    if args.mode == "onnx" and args.reference:
        reference = np.load(args.reference / "scores.npy")
        np.testing.assert_allclose(scores, reference, atol=3e-5, rtol=3e-4)
        summary["maximum_reference_score_error"] = float(np.max(np.abs(scores-reference)))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[OK] {summary}", flush=True)


def main():
    """解析导出或评测阶段,耗时评测显示逐音频进度."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["export", "tensorflow", "onnx"], required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.mode == "export":
        export_model(args)
    else:
        if args.output is None:
            parser.error("评测必须提供 --output.")
        evaluate(args)


if __name__ == "__main__":
    main()
