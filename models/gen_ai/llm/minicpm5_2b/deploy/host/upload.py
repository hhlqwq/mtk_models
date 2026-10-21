"""构建 MiniCPM5 部署包并传输到开发板,不进行环境或文件哈希预检."""

import os
import shlex

from package import package_model

from tools.deploy import arguments
from tools.evaluation import is_test
from tools.llm.native import settings
from tools.process import run_command


def main():
    """直接打包并传输当前模型,由命令退出码报告执行失败."""
    native = settings(arguments(__file__))
    package = package_model(native)
    host = os.environ["MTK_BOARD_HOST"]
    target = native.ctx.config["board"]["deploy_dir"]
    run_command(["ssh", "-o", "BatchMode=yes", host,
                 "mkdir -p -- " + shlex.quote(target)])
    run_command([
        "rsync", "-a", "--delete", "--exclude=.run.lock",
        "--rsync-path", f"flock -n {shlex.quote(target + '/.run.lock')} rsync",
        "--info=progress2", "-e", "ssh -o BatchMode=yes",
        str(package) + "/", f"{host}:{shlex.quote(target)}/",
    ])
    print(f"[上传完成] 当前转换资源保留在 {native.work}.", flush=True)
    flag = " -test" if is_test() else ""
    print(f"[下一步] 在板端执行: cd {shlex.quote(target)} && bash run.sh{flag}", flush=True)


if __name__ == "__main__":
    main()
