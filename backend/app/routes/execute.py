from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.executor import execute_code
from app.executor.models import ExecutionResult

router = APIRouter(prefix="/api")


class ExecuteRequest(BaseModel):
    code: str = Field(max_length=10_000)
    mode: Literal["quick", "full"]
    device: Literal["cpu", "mps"] = "cpu"


class ExecuteResponse(BaseModel):
    stdout: str
    charts: list[str]
    error: str | None
    execution_time_ms: float


@router.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest) -> ExecuteResponse:
    result: ExecutionResult = await execute_code(
        code=request.code,
        mode=request.mode,
        device=request.device,
    )
    return ExecuteResponse(
        stdout=result.stdout,
        charts=result.charts,
        error=result.error,
        execution_time_ms=result.execution_time_ms,
    )
