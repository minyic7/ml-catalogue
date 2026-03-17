from app.executor.config import get_device_warning, get_execution_env, get_timeout
from app.executor.models import ExecutionResult
from app.executor.sandbox import run_sandboxed


async def execute_code(code: str, mode: str, device: str) -> ExecutionResult:
    """Execute code in a sandboxed subprocess with mode/device configuration."""
    env = get_execution_env(mode, device)
    timeout = get_timeout(mode)
    warning = get_device_warning(device)

    result = await run_sandboxed(
        code=code,
        timeout=timeout,
        env=env,
        mode=mode,
        device=device,
    )

    if warning:
        result = ExecutionResult(
            stdout=warning + "\n" + result.stdout if result.stdout else warning,
            charts=result.charts,
            error=result.error,
            execution_time_ms=result.execution_time_ms,
        )

    return result
