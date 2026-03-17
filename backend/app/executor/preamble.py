"""Preamble script prepended to every user snippet before execution.

Sets up monkey-patches for matplotlib so that plt.show() and plt.savefig()
capture charts as PNG files in a temp directory. After the snippet runs,
captured charts are base64-encoded and written to a JSON manifest file.

The preamble is designed to be safe when matplotlib is not imported — it
hooks into the import system lazily so there is zero overhead for snippets
that don't use matplotlib.
"""

import atexit
import base64
import builtins
import json
import os

_CHART_DIR = os.environ.get("_SANDBOX_CHART_DIR", "")
_CHART_MANIFEST = os.environ.get("_SANDBOX_CHART_MANIFEST", "")
_captured_charts: list[str] = []
_chart_counter = 0
_matplotlib_patched = False


def _next_chart_path() -> str:
    global _chart_counter
    _chart_counter += 1
    return os.path.join(_CHART_DIR, f"chart_{_chart_counter}.png")


def _patch_matplotlib() -> None:
    """Monkey-patch pyplot.show and pyplot.savefig to capture charts."""
    global _matplotlib_patched
    if _matplotlib_patched:
        return
    _matplotlib_patched = True

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _original_show = _plt.show
    _original_savefig = _plt.savefig

    def _patched_show(*args, **kwargs) -> None:  # noqa: ARG001
        fig = _plt.gcf()
        if fig.get_axes():
            path = _next_chart_path()
            fig.savefig(path, format="png", bbox_inches="tight")
            _captured_charts.append(path)
        _plt.close("all")

    def _patched_savefig(fname, *args, **kwargs) -> None:
        # Save to user-specified location first
        _original_savefig(fname, *args, **kwargs)
        # Also capture a copy for the sandbox output
        try:
            path = _next_chart_path()
            fig = _plt.gcf()
            fig.savefig(path, format="png", bbox_inches="tight")
            _captured_charts.append(path)
        except Exception:
            pass

    _plt.show = _patched_show
    _plt.savefig = _patched_savefig


def _write_chart_manifest() -> None:
    """Write captured charts as base64 JSON to the manifest file."""
    if not _CHART_MANIFEST:
        return
    charts_b64: list[str] = []
    for path in _captured_charts:
        try:
            with open(path, "rb") as f:
                charts_b64.append(base64.b64encode(f.read()).decode("ascii"))
        except OSError:
            pass
    with open(_CHART_MANIFEST, "w") as f:
        json.dump(charts_b64, f)


# Hook into __import__ to lazily patch matplotlib when first imported
if _CHART_DIR and _CHART_MANIFEST:
    _original_import = builtins.__import__

    def _hooked_import(name, *args, **kwargs):  # noqa: N802
        mod = _original_import(name, *args, **kwargs)
        if not _matplotlib_patched and (
            name == "matplotlib.pyplot"
            or (name == "pyplot" and args and "matplotlib" in str(args[0]))
        ):
            _patch_matplotlib()
        return mod

    builtins.__import__ = _hooked_import
    atexit.register(_write_chart_manifest)
