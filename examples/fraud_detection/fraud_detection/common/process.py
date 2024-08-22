""" Process Related Helpers """

import logging
import shlex
import subprocess

logger = logging.getLogger(__name__)


def process_launch_wait(shell_out_cmd: str, cwd: str = ".") -> None:
    """
    Internal function for wrapping process launches [and waiting].

    Parameters
    ----------
    shell_out_cmd: str
        The command to be executed.
    cwd: str
        The `current working directory` of the command.  This is the directory to launch the command from.
    """

    args = shlex.split(shell_out_cmd)

    with subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        for line in iter(process.stdout.readline, b""):
            logger.info(line)

    if process.returncode != 0:
        message: str = f"Subprocess failed with exit code: {process.returncode}"
        raise ChildProcessError(message)
