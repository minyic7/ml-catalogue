"""Shared fixtures for backend tests."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    """FastAPI test client."""
    return TestClient(app)


# --- Sample code snippets used across test modules ---


@pytest.fixture()
def hello_snippet():
    return 'print("hello sandbox")'


@pytest.fixture()
def syntax_error_snippet():
    return "def foo(\n"


@pytest.fixture()
def import_error_snippet():
    return "import nonexistent_module_xyz"


@pytest.fixture()
def infinite_loop_snippet():
    return "while True: pass"


@pytest.fixture()
def matplotlib_snippet():
    return (
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1, 2, 3], [4, 5, 6])\n"
        "plt.show()\n"
    )
