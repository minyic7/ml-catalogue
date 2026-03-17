"""Pre-execution security checks for the sandbox.

Uses Python's ``ast`` module to statically analyse user code before execution.
Blocks dangerous imports, built-in abuse, and unsafe attribute access.
"""

from __future__ import annotations

import ast

# ---------------------------------------------------------------------------
# Blocked / allowed module lists
# ---------------------------------------------------------------------------

BLOCKED_MODULES: frozenset[str] = frozenset(
    {
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
        "shelve",
        "marshal",
        "code",
        "codeop",
        "compileall",
        "py_compile",
        "multiprocessing",
        "threading",
        "signal",
        "_thread",
        "webbrowser",
        "ftplib",
        "smtplib",
        "xmlrpc",
        "telnetlib",
    }
)

ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "numpy",
        "pandas",
        "matplotlib",
        "sklearn",
        "scikit-learn",
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
        "copy",
        "operator",
        "abc",
        "numbers",
        "decimal",
        "fractions",
        "statistics",
        "textwrap",
        "enum",
        "dataclasses",
        "typing",
        "warnings",
        "contextlib",
        "io",
        "csv",
        "pprint",
        "hashlib",
        "hmac",
        "secrets",
        "imblearn",
        "imbalanced-learn",
        # Preamble internals
        "ml_catalogue_runtime",
        "builtins",
        "types",
        "atexit",
        "base64",
    }
)

# Dangerous dunder attributes that enable sandbox escape
BLOCKED_ATTRS: frozenset[str] = frozenset(
    {
        "__subclasses__",
        "__globals__",
        "__builtins__",
        "__code__",
        "__reduce__",
        "__reduce_ex__",
        "__getattr__",
        "__import__",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _top_level_module(name: str) -> str:
    """Return the top-level package from a dotted module path."""
    return name.split(".")[0]


def _is_module_blocked(name: str) -> bool:
    """Check whether *name* (possibly dotted) resolves to a blocked module."""
    top = _top_level_module(name)
    # Explicitly allowed takes precedence
    if top in ALLOWED_MODULES:
        return False
    return top in BLOCKED_MODULES


# ---------------------------------------------------------------------------
# AST visitor
# ---------------------------------------------------------------------------


class _SecurityVisitor(ast.NodeVisitor):
    """Walk the AST and collect security violations."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    # -- import / import-from -------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            if _is_module_blocked(alias.name):
                self.violations.append(
                    f"Blocked import: '{alias.name}' is not allowed in the sandbox "
                    f"(line {node.lineno})"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.module and _is_module_blocked(node.module):
            self.violations.append(
                f"Blocked import: '{node.module}' is not allowed in the sandbox "
                f"(line {node.lineno})"
            )
        self.generic_visit(node)

    # -- __import__() calls ---------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        # __import__("something")
        if isinstance(node.func, ast.Name) and node.func.id == "__import__":
            self.violations.append(
                f"Blocked call: '__import__()' is not allowed in the sandbox "
                f"(line {node.lineno})"
            )

        # eval / exec / compile
        if isinstance(node.func, ast.Name) and node.func.id in {
            "eval",
            "exec",
            "compile",
        }:
            self.violations.append(
                f"Blocked call: '{node.func.id}()' is not allowed in the sandbox "
                f"(line {node.lineno})"
            )

        # open() with write mode
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            self._check_open_write(node)

        self.generic_visit(node)

    def _check_open_write(self, node: ast.Call) -> None:
        """Flag open() calls that use a write mode."""
        write_modes = {"w", "a", "x", "r+", "w+", "a+", "x+", "wb", "ab", "xb", "r+b"}
        # Check positional mode argument (second arg)
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            mode_val = str(node.args[1].value)
            if any(m in mode_val for m in {"w", "a", "x", "+"}):
                self.violations.append(
                    f"Blocked call: 'open()' with write mode '{mode_val}' "
                    f"is not allowed in the sandbox (line {node.lineno})"
                )
                return
        # Check keyword mode argument
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                mode_val = str(kw.value.value)
                if any(m in mode_val for m in {"w", "a", "x", "+"}):
                    self.violations.append(
                        f"Blocked call: 'open()' with write mode '{mode_val}' "
                        f"is not allowed in the sandbox (line {node.lineno})"
                    )
                    return

    # -- dangerous attribute access -------------------------------------

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if node.attr in BLOCKED_ATTRS:
            self.violations.append(
                f"Blocked attribute: '{node.attr}' access is not allowed "
                f"in the sandbox (line {node.lineno})"
            )
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_code_security(code: str) -> list[str]:
    """Statically analyse *code* and return a list of security violations.

    Returns an empty list when the code passes all checks.
    Raises ``SyntaxError`` (via ``ast.parse``) if the code is unparseable.
    """
    tree = ast.parse(code, filename="<user_code>")
    visitor = _SecurityVisitor()
    visitor.visit(tree)
    return visitor.violations
