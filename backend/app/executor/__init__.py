from app.executor.models import ExecutionResult


async def execute_code(code: str, mode: str, device: str) -> ExecutionResult:
    """Execute code and return results.

    This is a stub implementation. The real sandbox executor
    will replace this in a separate ticket.
    """
    return ExecutionResult(
        stdout="Stub: code received",
        charts=[],
        error=None,
        execution_time_ms=0.0,
    )
