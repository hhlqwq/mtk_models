"""打包并上传 DeepSeek 双 DLA,自动运行板端通路检查."""

import os
import shlex
import sys

from package import package_model

from tools.deploy import arguments
from tools.llm.native import settings
from tools.process import run_command


def main():
    """直接打包并传输当前模型,由命令退出码报告执行失败."""
    native = settings(arguments(__file__), python_executable=sys.executable)
    package = package_model(native)
    host = os.environ["MTK_BOARD_HOST"]
    target = native.ctx.config["board"]["deploy_dir"]
    run_command(["ssh", "-o", "BatchMode=yes", host,
                 "mkdir -p -- " + shlex.quote(target)])
    run_command([
        "rsync", "-a", "--delete", "--exclude=.run.lock", "--exclude=results/", "--exclude=examples/output/",
        "--rsync-path", f"flock -n {shlex.quote(target + '/.run.lock')} rsync",
        "--info=progress2", "-e", "ssh -o BatchMode=yes",
        str(package) + "/", f"{host}:{shlex.quote(target)}/",
    ])
    print(f"[上传完成] 当前转换资源保留在 {native.work}.", flush=True)
    print("[部署] 生成板端展示示例,不计算正式指标.", flush=True)
    run_command(["ssh", "-o", "BatchMode=yes", host,
                 "bash " + shlex.quote(target + "/compile.sh")])
    examples = native.ctx.model / "examples/output"
    examples.mkdir(parents=True, exist_ok=True)
    run_command(["scp", "-o", "BatchMode=yes",
                 f"{host}:{shlex.quote(target + '/examples/output/zh_demo.txt')}",
                 f"{host}:{shlex.quote(target + '/examples/output/en_demo.txt')}", str(examples)])
    print("[评价入口] 准备全量资源后在部署目录执行 bash evaluate.sh.", flush=True)


if __name__ == "__main__":
    main()
