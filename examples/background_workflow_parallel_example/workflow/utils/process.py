""" Process Related Helpers """

import shlex
import subprocess


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

    with subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE) as process:
        for line in iter(process.stdout.readline, b""):
            print(line)
