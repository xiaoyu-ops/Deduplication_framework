import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence


class ExecutorError(RuntimeError):
    """Raised when executor fails to run a command."""


@dataclass
class ExecResult:
    command: List[str]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float


class BaseExecutor:
    """Abstract executor interface."""

    def run(
        self,
        args: Sequence[str],
        *,
        env_name: Optional[str] = None,
        cwd: Optional[str] = None,
        capture_output: bool = True,
        check: bool = True,
        extra_env: Optional[Mapping[str, str]] = None,
    ) -> ExecResult:
        raise NotImplementedError


class LocalExecutor(BaseExecutor):
    """Runs commands on the current machine (Windows friendly)."""

    def __init__(self, conda_executable: Optional[str] = None) -> None:
        self.conda_exec = conda_executable or "conda"

    def _build_command(self, args: Sequence[str], env_name: Optional[str]) -> List[str]:
        if env_name:
            return [
                self.conda_exec,
                "run",
                "-n",
                env_name,
                *args,
            ]
        return list(args)

    def run(
        self,
        args: Sequence[str],
        *,
        env_name: Optional[str] = None,
        cwd: Optional[str] = None,
        capture_output: bool = True,
        check: bool = True,
        extra_env: Optional[Mapping[str, str]] = None,
    ) -> ExecResult:
        import time

        full_cmd = self._build_command(args, env_name)
        process_env = None
        if extra_env:
            process_env = os.environ.copy()
            process_env.update(extra_env)
        # Ensure child Python processes emit UTF-8 to avoid platform
        # dependent decoding issues; do not override if user explicitly set it.
        if process_env is None:
            process_env = os.environ.copy()
        process_env.setdefault("PYTHONIOENCODING", "utf-8")

        start = time.monotonic()
        try:
            # Capture raw bytes from subprocess and decode ourselves.
            # This lets us try UTF-8 first and fall back to the
            # system encoding (mbcs/cp936 on Windows) when needed,
            # avoiding persistent mojibake when child processes
            # emit output in a different encoding.
            completed = subprocess.run(
                full_cmd,
                cwd=cwd,
                capture_output=capture_output,
                text=False,
                env=process_env,
            )
            # completed.stdout/stderr are bytes when text=False.
            def _decode_bytes(b: Optional[bytes]) -> str:
                if b is None:
                    return ""
                # Try UTF-8 first
                try:
                    s = b.decode("utf-8")
                except Exception:
                    # Fallback to platform (Windows) encoding
                    try:
                        s = b.decode("mbcs")
                    except Exception:
                        # Last resort: decode with replacement
                        s = b.decode("utf-8", errors="replace")
                return s

            # Decode outputs to strings for downstream handling
            stdout_decoded = _decode_bytes(completed.stdout) if getattr(completed, "stdout", None) is not None else ""
            stderr_decoded = _decode_bytes(completed.stderr) if getattr(completed, "stderr", None) is not None else ""
            # Attach decoded strings to completed-like object for compatibility
            class _CompletedShim:
                def __init__(self, proc, out, err):
                    self.returncode = proc.returncode
                    self.stdout = out
                    self.stderr = err
                    # preserve args for error messages
                    self.args = proc.args

            completed = _CompletedShim(completed, stdout_decoded, stderr_decoded)
        except FileNotFoundError as exc:
            missing = full_cmd[0] if full_cmd else "<unknown>"
            raise ExecutorError(f"Executable not found: {missing}") from exc
        elapsed = time.monotonic() - start

        if check and completed.returncode != 0:
            raise ExecutorError(
                f"Command failed with code {completed.returncode}: {' '.join(shlex.quote(x) for x in full_cmd)}\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

        return ExecResult(
            command=list(full_cmd),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            elapsed_seconds=elapsed,
        )


def create_executor(exec_type: str = "local", conda_executable: Optional[str] = None) -> BaseExecutor:
    exec_type = exec_type.lower()
    if exec_type == "local":
        return LocalExecutor(conda_executable=conda_executable)
    raise ValueError(f"Unsupported executor type: {exec_type}")
