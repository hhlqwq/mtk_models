"""上传本次运行包,在同一板端文件锁内检查占用、传输并核对文件."""

import json
import os
from pathlib import Path
import shlex

from tools.deploy import arguments
from tools.evaluation import is_test
from tools.files import require_file, sha256
from tools.llm.native import settings
from tools.process import run_command


def main():
    """核对打包记录,上传后保留当前转换资源供板端问题排查."""
    native = settings(arguments(__file__))
    record = json.loads((native.work / "deployment_package.json").read_text())
    mode = "test" if is_test() else "full"
    if record["mode"] != mode or record["run_id"] != os.environ["EVAL_RUN_ID"]:
        raise ValueError("打包记录不属于当前运行或评价模式.")
    package = Path(record["path"])
    if sha256(require_file(package / "manifest.json")) != record["manifest_sha256"]:
        raise ValueError("打包后清单已变化,拒绝上传.")
    host = os.environ["MTK_BOARD_HOST"]
    target = native.ctx.config["board"]["deploy_dir"]
    options = ["-o", "BatchMode=yes"]
    # 以下代码在板端执行,不依赖主机的 SDK 或仓库 Python 模块.
    guard = """
import os
import sys
from pathlib import Path
target = Path(sys.argv[1]).resolve()
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        if (proc / 'cwd').resolve().is_relative_to(target):
            raise RuntimeError('部署目录正在使用,请先结束该任务后重新上传: ' + proc.name)
        for fd in (proc / 'fd').iterdir():
            path = fd.resolve()
            if path == target / '.run.lock':
                continue
            if path.is_relative_to(target):
                raise RuntimeError('进程打开部署文件,拒绝覆盖: ' + proc.name)
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
"""
    verify = """
import hashlib
import json
import shutil
import sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = root / 'manifest.json'
if hashlib.sha256(manifest.read_bytes()).hexdigest() != sys.argv[2]:
    raise ValueError('部署清单哈希不符')
files = json.loads(manifest.read_text())['files']
for index, row in enumerate(files, 1):
    path = root / row['name']
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('非法清单路径')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if path.stat().st_size != row['bytes'] or digest.hexdigest() != row['sha256']:
        raise ValueError('上传文件校验失败: ' + str(path))
    print('[上传核对] ' + str(index) + '/' + str(len(files)) + ' ' + row['name'], flush=True)
results = root / 'results'
if results.is_symlink() or not results.resolve().is_relative_to(root.resolve()):
    raise ValueError('结果目录越界')
if results.exists():
    shutil.rmtree(results)
results.mkdir()
print('[清理] 新部署包校验完成,移除上一部署包的评价输出.', flush=True)
"""
    run_command(["ssh", *options, host, "mkdir -p -- " + shlex.quote(target)])
    # rsync 的远端包装在同一把锁内完成三个动作,核对信息写 stderr 避免破坏传输协议.
    remote_steps = "\n".join((
        "python3 -c " + shlex.quote(guard) + " " + shlex.quote(target),
        'rsync "$@"',
        "python3 -c " + shlex.quote(verify) + " " + shlex.quote(target)
        + " " + shlex.quote(record["manifest_sha256"]) + " >&2",
    ))
    remote_command = (
        "flock -n " + shlex.quote(target + "/.run.lock")
        + " sh -eu -c " + shlex.quote(remote_steps) + " sh")
    run_command([
        "rsync", "-a", "--delete", "--exclude=results/", "--exclude=.run.lock",
        "--rsync-path", remote_command, "--info=progress2", "-e", "ssh -o BatchMode=yes",
        str(package) + "/", f"{host}:{shlex.quote(target)}/",
    ])
    print(f"[上传完成] 保留当前转换目录用于诊断: {native.work}.", flush=True)
    flag = " -test" if is_test() else ""
    print(f"[下一步] 在板端执行: cd {shlex.quote(target)} && bash run.sh{flag}", flush=True)


if __name__ == "__main__":
    main()
