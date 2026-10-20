"""上传本次完整运行包,保留板端已有结果,不自动执行模型."""

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
    """核对打包记录、模式和运行编号,上传后校验部署文件 SHA256."""
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
    guard = """
import os,sys
from pathlib import Path
target=Path(sys.argv[1]).resolve()
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():
        continue
    try:
        if (proc/'cwd').resolve().is_relative_to(target):
            raise RuntimeError('部署目录正在使用,请先结束该任务后重新上传: '+proc.name)
        for fd in (proc/'fd').iterdir():
            if fd.resolve().is_relative_to(target):
                raise RuntimeError('进程打开部署文件,拒绝覆盖: '+proc.name)
    except (FileNotFoundError,PermissionError,ProcessLookupError):
        pass
target.mkdir(parents=True,exist_ok=True)
"""
    run_command(["ssh", *options, host,
                 "python3 -c " + shlex.quote(guard) + " " + shlex.quote(target)])
    # 不删除板端已有结果; 每次运行使用独立运行编号.
    run_command(["rsync", "-a", "--info=progress2", "-e", "ssh -o BatchMode=yes",
                 str(package) + "/", f"{host}:{shlex.quote(target)}/"])
    verify = """
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
manifest=root/'manifest.json'
if hashlib.sha256(manifest.read_bytes()).hexdigest()!=sys.argv[2]:
    raise ValueError('部署清单哈希不符')
files=json.loads(manifest.read_text())['files']
for index,row in enumerate(files,1):
    path=root/row['name']
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('非法清单路径')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    if path.stat().st_size!=row['bytes'] or digest.hexdigest()!=row['sha256']:
        raise ValueError('上传文件校验失败: '+str(path))
    print('[上传核对] '+str(index)+'/'+str(len(files))+' '+row['name'],flush=True)
"""
    run_command(["ssh", *options, host, "python3 -c " + shlex.quote(verify) + " "
                 + shlex.quote(target) + " " + shlex.quote(record["manifest_sha256"])])
    flag = " -test" if is_test() else ""
    print(f"[下一步] 在板端执行: cd {shlex.quote(target)} && bash run.sh{flag}", flush=True)


if __name__ == "__main__":
    main()
