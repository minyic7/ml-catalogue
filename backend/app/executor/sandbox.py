"""Sandboxed Python code executor.

Runs user-supplied Python snippets in an isolated subprocess with resource
limits, captures stdout/stderr and any matplotlib charts produced.
"""

import asyncio
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

if platform.system() == "Linux":
    import resource
else:
    resource = None  # type: ignore[assignment]

try:
    from app.executor.models import ExecutionResult
except ImportError:
    # Fallback for when models.py is not yet available (parallel ticket).
    from dataclasses import dataclass, field

    @dataclass
    class ExecutionResult:  # type: ignore[no-redef]
        stdout: str = ""
        stderr: str = ""
        error: str | None = None
        charts: list[str] = field(default_factory=list)
        execution_time_ms: float = 0.0


_PREAMBLE_PATH = Path(__file__).with_name("preamble.py")

# Default memory limit: 1 GB (matplotlib + numpy need substantial address space)
_DEFAULT_MEMORY_LIMIT = 1024 * 1024 * 1024


def _build_preexec(memory_limit: int):
    """Return a preexec_fn that sets resource limits (Linux only)."""
    if platform.system() != "Linux":
        return None

    def _set_limits() -> None:
        # Memory limit (address space)
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        # Disable core dumps
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return _set_limits


async def run_sandboxed(
    code: str,
    timeout: float,
    env: dict[str, str] | None = None,
    memory_limit: int = _DEFAULT_MEMORY_LIMIT,
    mode: str = "quick",
    device: str = "cpu",
) -> ExecutionResult:
    """Execute a Python code snippet in a sandboxed subprocess.

    Args:
        code: Python source code to execute.
        timeout: Maximum wall-clock seconds before the process is killed.
        env: Optional extra environment variables for the subprocess.
        memory_limit: Maximum address-space in bytes (default 1 GB).

    Returns:
        ExecutionResult with stdout, stderr, error, charts, and timing.
    """
    tmp_dir = tempfile.mkdtemp(prefix="sandbox_")
    chart_dir = os.path.join(tmp_dir, "charts")
    os.makedirs(chart_dir)
    chart_manifest = os.path.join(tmp_dir, "chart_manifest.json")

    # Build the script: preamble + user code
    preamble_source = _PREAMBLE_PATH.read_text()
    script = preamble_source + "\n\n# --- user code ---\n" + code

    script_path = os.path.join(tmp_dir, "snippet.py")
    with open(script_path, "w") as f:
        f.write(script)

    # Build environment
    proc_env = os.environ.copy()
    proc_env["_SANDBOX_CHART_DIR"] = chart_dir
    proc_env["_SANDBOX_CHART_MANIFEST"] = chart_manifest
    # Best-effort network restriction: unset proxy vars, set offline hint
    for key in list(proc_env.keys()):
        if "proxy" in key.lower():
            del proc_env[key]
    if env:
        proc_env.update(env)

    def _execute() -> ExecutionResult:
        start = time.perf_counter()
        try:
            try:
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tmp_dir,
                    env=proc_env,
                    preexec_fn=_build_preexec(memory_limit),
                )
            except subprocess.TimeoutExpired:
                elapsed_ms = (time.perf_counter() - start) * 1000
                return ExecutionResult(
                    stdout="",
                    stderr="",
                    error=f"Execution timed out after {timeout}s",
                    charts=[],
                    execution_time_ms=round(elapsed_ms, 2),
                )

            elapsed_ms = (time.perf_counter() - start) * 1000

            stdout = result.stdout or ""
            stderr = result.stderr or ""

            # Detect traceback in stderr
            error: str | None = None
            if stderr and ("Traceback" in stderr or result.returncode != 0):
                error = stderr.strip()

            # Read captured charts
            charts: list[str] = []
            if os.path.isfile(chart_manifest):
                try:
                    with open(chart_manifest) as f:
                        charts = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                error=error,
                charts=charts,
                execution_time_ms=round(elapsed_ms, 2),
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return await asyncio.to_thread(_execute)
