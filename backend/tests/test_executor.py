"""Tests for the sandbox executor."""

import pytest

from app.executor.config import get_execution_env, get_timeout
from app.executor.sandbox import run_sandboxed


@pytest.mark.asyncio
async def test_simple_print(hello_snippet):
    """Stdout from a print statement is captured."""
    result = await run_sandboxed(hello_snippet, timeout=10)
    assert result.error is None
    assert "hello sandbox" in result.stdout


@pytest.mark.asyncio
async def test_chart_capture(matplotlib_snippet):
    """A matplotlib snippet produces at least one base64 chart."""
    result = await run_sandboxed(matplotlib_snippet, timeout=30)
    assert result.error is None
    assert len(result.charts) >= 1
    # Charts should be base64-encoded PNGs (starts with iVBOR for PNG header)
    assert result.charts[0].startswith("iVBOR")


@pytest.mark.asyncio
async def test_quick_mode_env():
    """Quick mode sets SAMPLE_SIZE=100 and timeout=30s."""
    env = get_execution_env("quick", "cpu")
    assert env["ML_CATALOGUE_SAMPLE_SIZE"] == "100"
    assert env["ML_CATALOGUE_MODE"] == "quick"
    assert get_timeout("quick") == 30.0


@pytest.mark.asyncio
async def test_full_mode_env():
    """Full mode sets SAMPLE_SIZE=0 and timeout=120s."""
    env = get_execution_env("full", "cpu")
    assert env["ML_CATALOGUE_SAMPLE_SIZE"] == "0"
    assert env["ML_CATALOGUE_MODE"] == "full"
    assert get_timeout("full") == 120.0


@pytest.mark.asyncio
async def test_timeout(infinite_loop_snippet):
    """An infinite loop is killed after the timeout and returns an error."""
    result = await run_sandboxed(infinite_loop_snippet, timeout=2)
    assert result.error is not None
    assert "timed out" in result.error.lower()


@pytest.mark.asyncio
async def test_import_error(import_error_snippet):
    """Importing a nonexistent module returns a clean error."""
    result = await run_sandboxed(import_error_snippet, timeout=10)
    assert result.error is not None
    assert "ModuleNotFoundError" in result.error


@pytest.mark.asyncio
async def test_syntax_error(syntax_error_snippet):
    """Malformed Python returns a syntax error."""
    result = await run_sandboxed(syntax_error_snippet, timeout=10)
    assert result.error is not None
    assert "SyntaxError" in result.error


@pytest.mark.asyncio
async def test_execution_time_recorded():
    """Execution time is recorded and positive."""
    result = await run_sandboxed('print("timing")', timeout=10)
    assert result.execution_time_ms > 0
