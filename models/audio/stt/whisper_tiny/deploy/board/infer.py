"""在 Python 中完成 Whisper 音频前处理与双 DLA 贪心解码."""

import json
import subprocess
import time

import numpy as np

from tools.board import context
from tools.evaluation import is_test, select_samples, SPEECH_EXAMPLES
from tools.runtime import Runtime, peak_rss_kib
from tools.output_metrics import OutputComparison


def load_samples(root):
    """核对 LibriSpeech test-clean 转录和音频,按固定规则选取样本."""
    samples = []
    for transcript in sorted(root.rglob("*.trans.txt")):
        for line in transcript.read_text().splitlines():
            sample_id, text = line.split(" ", 1)
            audio = transcript.parent / f"{sample_id}.flac"
            if not audio.is_file():
                raise FileNotFoundError(audio)
            samples.append({"sample_id": sample_id, "audio_path": str(audio),
                            "reference_text": text})
    if len(samples) != 2620 or len({row["sample_id"] for row in samples}) != 2620:
        raise ValueError("LibriSpeech test-clean 必须包含 2620 条唯一音频.")
    return select_samples(samples, "whisper_tiny", key=lambda row: row["sample_id"],
                          required=SPEECH_EXAMPLES)


def preprocess(path, filters):
    """沿用 PCM16 解码、30 秒截补齐、400 点 Hann 与 80 维 Mel 协议."""
    started = time.perf_counter()
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1",
                          "-ar", "16000", "-f", "s16le", "pipe:1"],
                         check=True, stdout=subprocess.PIPE).stdout
    values = np.frombuffer(pcm, dtype="<i2")[:480000].astype(np.float32) / 32768
    if not values.size:
        raise ValueError(f"音频为空: {path}.")
    duration = len(values) / 16000
    audio = np.pad(values, (0, 480000 - len(values)))
    decoded = time.perf_counter()
    framed = np.lib.stride_tricks.sliding_window_view(np.pad(audio, (200, 200), mode="reflect"), 400)[::160][:3000]
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)).astype(np.float32)
    spectrum = np.fft.rfft(framed * window, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32).T
    mel = np.log10(np.maximum(filters @ power, 1e-10))
    mel = (np.maximum(mel, mel.max() - 8) + 4) / 4
    return mel.astype("<f2"), duration, {"audio_decode_ms": (decoded - started) * 1000,
        "log_mel_ms": (time.perf_counter() - decoded) * 1000,
        "mel_write_ms": 0.0, "cache_hit": False}


class WhisperRuntime:
    """复用 Encoder 与 Decoder,每条音频独立清空 KV Cache."""

    def __init__(self, encoder, decoder, config):
        """绑定既有 FP16 原生接口和固定 200 Token 缓存契约."""
        self.encoder, self.decoder, self.config = encoder, decoder, config
        if tuple(map(len, encoder.sizes)) != (1, 1) or tuple(map(len, decoder.sizes)) != (13, 9):
            raise ValueError("Whisper 双 DLA I/O 数量不符.")
        self.encoder_inputs = [np.zeros(size // 2, dtype="<f2") for size in encoder.sizes[0]]
        self.encoder_outputs = [np.zeros(size // 2, dtype="<f2") for size in encoder.sizes[1]]
        self.inputs = [np.zeros(size // 2, dtype="<f2") for size in decoder.sizes[0]]
        self.outputs = [np.zeros(size // 2, dtype="<f2") for size in decoder.sizes[1]]
        expected_inputs = [51872, 1500 * 384, 448, 200, 200] + [6 * 64 * 200] * 8
        expected_outputs = [51872] + [6 * 64 * 200] * 8
        if (encoder.sizes != [[80 * 3000 * 2], [1500 * 384 * 2]]
                or any(len(buffer) < expected or len(buffer) > (expected + 7) // 8 * 8
                       for buffer, expected in zip(self.inputs + self.outputs, expected_inputs + expected_outputs))):
            raise ValueError("Whisper 原生 FP16 缓冲区形状不符.")
        self.initial = [int(value) for value in config["initial"].split(",")]
        self.suppress = [int(value) for value in config["suppress"].split(",") if value]
        self.first = [int(value) for value in config["suppress_first"].split(",") if value]
        self.limit = int(config["max_generated_tokens"])
        if not self.initial or len(self.initial) + self.limit > 200:
            raise ValueError("解码 Token 数超过固定 Cache 上限.")

    def decode(self, mel):
        """逐 Token 执行官方固定过滤规则,只统计实际 SDK 调用耗时."""
        np.copyto(self.encoder_inputs[0], mel.ravel())
        encoder_ms = self.encoder.infer(self.encoder_inputs, self.encoder_outputs)
        np.copyto(self.inputs[1], self.encoder_outputs[0])
        for cache in self.inputs[5:]:
            cache.fill(0)
        current, prompt_index, position = self.initial[0], 0, 0
        tokens, times, reached, first_ms = [], [], False, 0.0
        while position < 200 and len(tokens) < self.limit:
            for index, value in ((0, current), (2, position), (3, position)):
                self.inputs[index].fill(0)
                self.inputs[index][value] = 1
            self.inputs[4].fill(-10000)
            self.inputs[4][:position + 1] = 0
            times.append(self.decoder.infer(self.inputs, self.outputs))
            for index in range(8):
                np.copyto(self.inputs[5 + index], self.outputs[1 + index])
            if prompt_index + 1 < len(self.initial):
                prompt_index += 1
                current = self.initial[prompt_index]
                position += 1
                continue
            logits = self.outputs[0][:51865].astype(np.float32)
            logits[self.suppress] = -np.inf
            if not tokens:
                logits[self.first] = -np.inf
                first_ms = encoder_ms + sum(times)
            token = int(np.argmax(logits))
            if not np.isfinite(logits[token]):
                raise ValueError("解码没有有效 Token.")
            if token == int(self.config["eot"]):
                reached = True
                break
            tokens.append(token)
            current, position = token, position + 1
        row = {"tokens": tokens, "reached_eot": reached, "status": "ok",
               "backend": "python_neuron_runtime_hw"}
        if not is_test():
            total = encoder_ms + sum(times)
            row.update(encoder_ms=encoder_ms, first_token_ms=first_ms,
                decoder_calls=len(times), decoder_total_ms=sum(times), npu_total_ms=total,
                tokens_per_second=len(tokens) * 1000 / total, peak_rss_kb=peak_rss_kib())
        return row


def main():
    """音频逐条在内存准备,保存唯一预测流及评分需要的来源清单."""
    ctx = context(__file__, new_run=True)
    if ctx.result.exists():
        raise FileExistsError(ctx.result)
    samples = load_samples(ctx.dataset)
    filters = np.fromfile(ctx.root / "models/mel_filters_f32.bin", dtype="<f4")
    if filters.size != 80 * 201:
        raise ValueError("Whisper Mel 滤波器大小不符.")
    filters = filters.reshape(80, 201)
    config = dict(line.split("=", 1) for line in (ctx.root / "models/decode_config.txt").read_text().splitlines() if "=" in line)
    cache = ctx.work / "cache/librispeech"
    cache.mkdir(parents=True)
    preprocessing = []
    comparison = OutputComparison(ctx, scope="encoder_output_all_audio; excludes_autoregressive_decoder")
    with Runtime(ctx.root / "models/encoder_fp32.dla") as encoder, Runtime(ctx.root / "models/decoder_step_fp32.dla") as decoder:
        model = WhisperRuntime(encoder, decoder, config)
        with (cache / "source_manifest.jsonl").open("w") as sources, (ctx.work / "board_predictions.jsonl").open("w") as predictions:
            for count, sample in enumerate(samples, 1):
                mel, duration, timing = preprocess(sample["audio_path"], filters)
                row = model.decode(mel)
                comparison.update([mel[None]], {"output": model.encoder_outputs[0].reshape(1, 1500, 384)})
                row.update(sample_id=sample["sample_id"], duration_seconds=duration)
                if not is_test():
                    row["npu_rtf"] = row["npu_total_ms"] / (duration * 1000)
                predictions.write(json.dumps(row) + "\n")
                sources.write(json.dumps(dict(sample, duration_seconds=duration)) + "\n")
                if not is_test():
                    preprocessing.append(dict(timing, sample_id=sample["sample_id"], duration_seconds=duration))
                if count % 25 == 0 or count == len(samples):
                    print(f"[推理] Whisper {count}/{len(samples)}", flush=True)

    if not is_test():
        (cache / "preprocess_metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in preprocessing))
    comparison.finish()


if __name__ == "__main__":
    main()
