"""按原生 LLM 的 MDLA 5.3 单核参数编译 Prefill 与 Decode."""

import os
from pathlib import Path

from tools.deploy import arguments
from tools.llm.native import settings
from tools.process import run_command


def main():
    """保留两个阶段各自的编译优化参数,禁止 CPU 桥接."""
    native = settings(arguments(__file__))
    ncc = Path(os.environ["NCC_ROOT"])
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(ncc / "lib")
    for shape in (f"{native.prefill}t{native.context}c", f"1t{native.context}c"):
        folder = native.quantized.with_name(native.quantized.name + "_" + shape)
        files = list(folder.glob("*.tflite"))
        if len(files) != 1:
            raise ValueError(f"{folder} 必须只有一个静态模型.")
        source = files[0]
        extra = ["--split-16a4w-conv-oc"] if shape.startswith("1t") else [
            "--broadcast-act-wgt", "--broadcast-flow-distance=63", "--split-large-conv-ic=1536"]
        target = source.with_suffix(".dla")
        run_command([ncc / "bin/ncc-tflite", "--arch=mdla5.3", "--l1-size-kb=256", "--num-mdla=1",
            "--opt=3", "--opt-footprint", "--opt-accuracy", "--stable-linearize",
            "--gno=LTS,Inception", "--gno-exp", "--gno-non-4d-tiling", "--mlo",
            "--disable-apusys", "--fc-to-conv", "--mdla-int16-lut",
            "--suppress-input", "--suppress-output", "--show-memory-summary",
            "--mdla-conv-exp", "--intval-color-legacy", "--disallow-bridge", *extra,
            source, "--dla-file", target], env=environment)
        if not target.is_file() or not target.stat().st_size:
            raise FileNotFoundError(target)
        print(f"[DLA] 最终模型: {target}", flush=True)


if __name__ == "__main__":
    main()
