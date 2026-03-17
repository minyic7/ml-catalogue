"""Tests for sandbox security hardening.

Verifies that dangerous imports, built-in abuse, and unsafe attribute access
are blocked before execution, and that resource limits are enforced.
"""

import platform

import pytest

from app.executor.sandbox import run_sandboxed
from app.executor.security import check_code_security

# ---------------------------------------------------------------------------
# AST pre-check: blocked imports
# ---------------------------------------------------------------------------


class TestBlockedImports:
    """Dangerous modules must be rejected at the AST level."""

    @pytest.mark.parametrize(
        "module",
        [
            "os",
            "sys",
            "subprocess",
            "shutil",
            "socket",
            "http",
            "urllib",
            "requests",
            "ctypes",
            "importlib",
            "pathlib",
            "pickle",
            "multiprocessing",
        ],
    )
    def test_blocked_import(self, module: str):
        violations = check_code_security(f"import {module}")
        assert len(violations) == 1
        assert "Blocked import" in violations[0]
        assert module in violations[0]

    @pytest.mark.parametrize(
        "module",
        ["os", "sys", "subprocess", "shutil", "socket"],
    )
    def test_blocked_from_import(self, module: str):
        violations = check_code_security(f"from {module} import something")
        assert len(violations) == 1
        assert "Blocked import" in violations[0]

    def test_blocked_dotted_import(self):
        violations = check_code_security("import os.path")
        assert len(violations) == 1
        assert "Blocked import" in violations[0]

    def test_blocked_from_dotted_import(self):
        violations = check_code_security("from http.client import HTTPConnection")
        assert len(violations) == 1
        assert "Blocked import" in violations[0]

    def test_default_deny_unknown_module(self):
        """Modules not in ALLOWED_MODULES are blocked (default-deny)."""
        violations = check_code_security("import antigravity")
        assert len(violations) == 1
        assert "Blocked import" in violations[0]

    def test_builtins_import_blocked(self):
        """builtins module should not be importable by user code."""
        violations = check_code_security("import builtins")
        assert len(violations) == 1
        assert "Blocked import" in violations[0]


# ---------------------------------------------------------------------------
# AST pre-check: allowed imports
# ---------------------------------------------------------------------------


class TestAllowedImports:
    """Safe modules must be allowed through."""

    @pytest.mark.parametrize(
        "module",
        [
            "numpy",
            "pandas",
            "matplotlib",
            "sklearn",
            "scipy",
            "torch",
            "statsmodels",
            "sqlite3",
            "math",
            "random",
            "collections",
            "itertools",
            "functools",
            "json",
            "re",
            "string",
            "datetime",
            "time",
        ],
    )
    def test_allowed_import(self, module: str):
        violations = check_code_security(f"import {module}")
        assert violations == []

    def test_allowed_dotted_import(self):
        violations = check_code_security("import matplotlib.pyplot")
        assert violations == []

    def test_allowed_from_import(self):
        violations = check_code_security(
            "from sklearn.linear_model import LinearRegression"
        )
        assert violations == []


# ---------------------------------------------------------------------------
# AST pre-check: dangerous calls
# ---------------------------------------------------------------------------


class TestDangerousCalls:
    """Built-in abuse and dynamic imports must be blocked."""

    def test_dunder_import_blocked(self):
        violations = check_code_security('__import__("os")')
        assert len(violations) == 1
        assert "__import__" in violations[0]

    def test_eval_blocked(self):
        violations = check_code_security('eval("1+1")')
        assert len(violations) == 1
        assert "eval" in violations[0]

    def test_exec_blocked(self):
        violations = check_code_security('exec("print(1)")')
        assert len(violations) == 1
        assert "exec" in violations[0]

    def test_compile_blocked(self):
        violations = check_code_security('compile("pass", "<>", "exec")')
        assert len(violations) == 1
        assert "compile" in violations[0]

    def test_open_write_blocked(self):
        violations = check_code_security('open("/etc/passwd", "w")')
        assert len(violations) == 1
        assert "write mode" in violations[0]

    def test_open_write_keyword_blocked(self):
        violations = check_code_security('open("/tmp/x", mode="a")')
        assert len(violations) == 1
        assert "write mode" in violations[0]

    def test_open_read_allowed(self):
        violations = check_code_security('open("data.csv", "r")')
        assert violations == []

    def test_open_default_mode_allowed(self):
        """open() with no mode defaults to 'r' which is safe."""
        violations = check_code_security('open("data.csv")')
        assert violations == []

    def test_io_open_write_blocked(self):
        """io.open() with write mode must be caught like built-in open()."""
        violations = check_code_security('import io\nio.open("/tmp/x", "w")')
        assert any("write mode" in v for v in violations)

    def test_method_call_exec_blocked(self):
        """builtins.exec() style calls must be caught."""
        violations = check_code_security('builtins.exec("print(1)")')
        assert any("exec()" in v for v in violations)

    def test_method_call_eval_allowed_on_library(self):
        """pd.eval() and similar library method calls must not be blocked."""
        violations = check_code_security('pd.eval("a + b")')
        assert violations == []

    def test_method_call_compile_allowed_on_library(self):
        """re.compile() and similar library method calls must not be blocked."""
        violations = check_code_security('re.compile(r"\\d+")')
        assert violations == []

    def test_getattr_blocked(self):
        violations = check_code_security('getattr(obj, "__globals__")')
        assert any("getattr()" in v for v in violations)


# ---------------------------------------------------------------------------
# AST pre-check: dangerous attribute access
# ---------------------------------------------------------------------------


class TestDangerousAttributes:
    """Sandbox-escape attributes must be blocked."""

    @pytest.mark.parametrize(
        "attr",
        [
            "__subclasses__",
            "__globals__",
            "__builtins__",
            "__code__",
            "__reduce__",
            "__import__",
        ],
    )
    def test_blocked_attribute(self, attr: str):
        violations = check_code_security(f"x.{attr}")
        assert len(violations) == 1
        assert attr in violations[0]


# ---------------------------------------------------------------------------
# Multiple violations
# ---------------------------------------------------------------------------


class TestMultipleViolations:
    """Code with multiple issues should report all of them."""

    def test_multiple_violations(self):
        code = "import os\nimport subprocess\neval('1')"
        violations = check_code_security(code)
        assert len(violations) == 3

    def test_violation_includes_line_number(self):
        violations = check_code_security("x = 1\nimport os")
        assert len(violations) == 1
        assert "line 2" in violations[0]


# ---------------------------------------------------------------------------
# Integration: sandbox rejects dangerous code
# ---------------------------------------------------------------------------


class TestSandboxRejectsCode:
    """The sandbox executor must block dangerous code and return clear errors."""

    @pytest.mark.asyncio
    async def test_import_os_rejected(self):
        result = await run_sandboxed("import os\nos.system('echo pwned')", timeout=10)
        assert result.error is not None
        assert "Security policy violation" in result.error
        assert "Blocked import" in result.error
        assert result.execution_time_ms == 0.0

    @pytest.mark.asyncio
    async def test_subprocess_rejected(self):
        result = await run_sandboxed(
            "import subprocess\nsubprocess.run(['ls'])",
            timeout=10,
        )
        assert result.error is not None
        assert "Security policy violation" in result.error

    @pytest.mark.asyncio
    async def test_eval_rejected(self):
        result = await run_sandboxed(
            "eval(\"__import__('os').system('id')\")",
            timeout=10,
        )
        assert result.error is not None
        assert "Security policy violation" in result.error

    @pytest.mark.asyncio
    async def test_dunder_subclasses_rejected(self):
        result = await run_sandboxed(
            "print(''.__class__.__subclasses__())",
            timeout=10,
        )
        assert result.error is not None
        assert "Security policy violation" in result.error

    @pytest.mark.asyncio
    async def test_open_write_rejected(self):
        result = await run_sandboxed(
            'open("/tmp/hack.txt", "w").write("pwned")',
            timeout=10,
        )
        assert result.error is not None
        assert "Security policy violation" in result.error


# ---------------------------------------------------------------------------
# Integration: safe code still runs
# ---------------------------------------------------------------------------


class TestSafeCodeStillRuns:
    """Legitimate educational code must continue to work."""

    @pytest.mark.asyncio
    async def test_print_works(self):
        result = await run_sandboxed('print("hello")', timeout=10)
        assert result.error is None
        assert "hello" in result.stdout

    @pytest.mark.asyncio
    async def test_numpy_works(self):
        result = await run_sandboxed(
            "import numpy as np\nprint(np.array([1,2,3]).sum())",
            timeout=15,
        )
        assert result.error is None
        assert "6" in result.stdout

    @pytest.mark.asyncio
    async def test_pandas_works(self):
        result = await run_sandboxed(
            "import pandas as pd\ndf = pd.DataFrame({'a': [1,2,3]})\nprint(df.shape)",
            timeout=15,
        )
        assert result.error is None
        assert "(3, 1)" in result.stdout

    @pytest.mark.asyncio
    async def test_sqlite3_memory_works(self):
        code = (
            "import sqlite3\n"
            "conn = sqlite3.connect(':memory:')\n"
            "conn.execute('CREATE TABLE t (x INTEGER)')\n"
            "conn.execute('INSERT INTO t VALUES (42)')\n"
            "row = conn.execute('SELECT * FROM t').fetchone()\n"
            "print(row[0])\n"
            "conn.close()"
        )
        result = await run_sandboxed(code, timeout=15)
        assert result.error is None
        assert "42" in result.stdout

    @pytest.mark.asyncio
    async def test_math_json_re_work(self):
        code = (
            "import math\nimport json\nimport re\n"
            "print(math.pi)\nprint(json.dumps({'a': 1}))\n"
            "print(re.findall(r'\\d+', 'abc123def456'))"
        )
        result = await run_sandboxed(code, timeout=10)
        assert result.error is None
        assert "3.14" in result.stdout

    @pytest.mark.asyncio
    async def test_matplotlib_chart(self):
        code = (
            "import matplotlib.pyplot as plt\nplt.plot([1,2,3],[4,5,6])\nplt.show()\n"
        )
        result = await run_sandboxed(code, timeout=30)
        assert result.error is None
        assert len(result.charts) >= 1


# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------


class TestResourceLimits:
    """Resource limits are enforced on Linux."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        platform.system() != "Linux",
        reason="Resource limits only enforced on Linux",
    )
    async def test_memory_limit_enforced(self):
        """Attempting to allocate well over 512 MB should fail."""
        code = "x = bytearray(600 * 1024 * 1024)"
        result = await run_sandboxed(code, timeout=10)
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_timeout_still_works(self):
        """Infinite loops are still killed by timeout."""
        result = await run_sandboxed("while True: pass", timeout=2)
        assert result.error is not None
        assert "timed out" in result.error.lower()
