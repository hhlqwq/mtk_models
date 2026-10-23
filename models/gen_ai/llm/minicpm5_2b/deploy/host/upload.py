"""构建 MiniCPM5 部署包并传输到开发板,不进行环境或文件哈希预检."""

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
    print("[待板端检查] 上传完成,尚未执行板端推理或生成展示示例.", flush=True)
    print(f"[板端入口] 请登录开发板后执行: cd -- {shlex.quote(target)} && bash compile.sh",
          flush=True)
    print("[评价入口] 独立准备千题 MMLU 和完整 WikiText2 参考,放入板端 evaluation/mmlu1000/ 后执行 bash evaluate.sh.",
          flush=True)



if __name__ == "__main__":
    main()
