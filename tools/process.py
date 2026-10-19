"""命令执行与日志控制,静默模式不主动重定向子进程输出."""

import shlex
import subprocess
import sys


def run_command(command, *, quiet=False, **kwargs):
    """显示实际命令,静默模式仅隐藏回显,失败时保留命令与异常."""
    arguments = [str(item) for item in command]
    if not quiet:
        print(f"[执行] {shlex.join(arguments)}", flush=True)
    try:
        return subprocess.run(arguments, check=True, **kwargs)
    except (subprocess.CalledProcessError, OSError):
        print(f"[失败命令] {shlex.join(arguments)}", file=sys.stderr, flush=True)
        raise
