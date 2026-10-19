"""执行原生 LLM 校准,保留成功标记和批次数校验."""

from tools.deploy import arguments
from tools.llm.native import python, settings


def main():
    """校准开始前清除旧标记,仅成功结束后写入完成批次数."""
    native = settings(arguments(__file__))
    marker = native.work / "calibration_complete.txt"
    marker.unlink(missing_ok=True)
    python(native, native.python.parent / "mtk_make_llm_ptq_calib_dataset", "converter",
           native.model / "config.json", native.work / "calibration.jsonl",
           "-b", str(native.batches), "-m", "128")
    marker.write_text(str(native.batches) + "\n")


if __name__ == "__main__":
    main()
