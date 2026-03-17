from pydantic import BaseModel


class ExecutionResult(BaseModel):
    stdout: str
    charts: list[str]
    error: str | None
    execution_time_ms: float
