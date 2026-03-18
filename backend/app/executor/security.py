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
        "xgboost",
        "lightgbm",
        "catboost",
        # Preamble internals (builtins intentionally excluded — user code
        # must not access builtins directly as it enables exec/eval bypass)
        "ml_catalogue_runtime",
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
    """Check whether *name* (possibly dotted) resolves to a blocked module.

    Uses a default-deny policy: any module not explicitly in ALLOWED_MODULES
    is blocked.
    """
    top = _top_level_module(name)
    return top not in ALLOWED_MODULES


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

    # Names blocked as bare calls: eval(), exec(), compile(), __import__()
    _BLOCKED_BARE_CALLS: frozenset[str] = frozenset(
        {"eval", "exec", "compile", "__import__"}
    )

    # Names blocked as method calls — only truly dangerous ones that have no
    # legitimate library usage (e.g. builtins.exec()).  compile/eval are
    # excluded because re.compile(), pd.eval(), etc. are legitimate.
    _BLOCKED_METHOD_CALLS: frozenset[str] = frozenset({"exec", "__import__"})

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func

        # Direct calls: eval(), exec(), compile(), __import__()
        if isinstance(func, ast.Name) and func.id in self._BLOCKED_BARE_CALLS:
            self.violations.append(
                f"Blocked call: '{func.id}()' is not allowed in the sandbox "
                f"(line {node.lineno})"
            )

        # Method-call style: builtins.exec(), obj.__import__(), etc.
        # Only block exec/__import__ — compile/eval have legitimate library
        # uses (re.compile, pd.eval).
        if isinstance(func, ast.Attribute) and func.attr in self._BLOCKED_METHOD_CALLS:
            self.violations.append(
                f"Blocked call: '{func.attr}()' via attribute access is not "
                f"allowed in the sandbox (line {node.lineno})"
            )

        # getattr() — block only when accessing dunder attributes (sandbox escape risk)
        if isinstance(func, ast.Name) and func.id == "getattr":
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                attr_name = node.args[1].value
                if isinstance(attr_name, str) and attr_name.startswith("__"):
                    self.violations.append(
                        f"Blocked call: 'getattr()' with dunder attribute "
                        f"'{attr_name}' is not allowed in the sandbox "
                        f"(line {node.lineno})"
                    )

        # open() / io.open() with write mode
        if isinstance(func, ast.Name) and func.id == "open":
            self._check_open_write(node)
        elif (
            isinstance(func, ast.Attribute)
            and func.attr == "open"
            and isinstance(func.value, ast.Name)
            and func.value.id == "io"
        ):
            self._check_open_write(node)

        self.generic_visit(node)

    def _check_open_write(self, node: ast.Call) -> None:
        """Flag open() calls that use a write mode."""
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
