"""Dataset mode and device configuration for the execution pipeline.

Provides environment variable generation for Quick/Full dataset modes
and MPS/CPU device selection.
"""

from enum import Enum


class DatasetMode(str, Enum):
    quick = "quick"
    full = "full"


_MODE_CONFIG = {
    DatasetMode.quick: {"sample_size": "100", "timeout": 30.0},
    DatasetMode.full: {"sample_size": "0", "timeout": 120.0},
}


def _detect_mps() -> bool:
    """Check MPS availability at runtime without requiring torch."""
    try:
        import torch

        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False


class DeviceConfig:
    """Resolves the target device, falling back to CPU when MPS is unavailable."""

    def __init__(self, requested: str) -> None:
        self.mps_available = _detect_mps()
        if requested == "mps" and not self.mps_available:
            self.device = "cpu"
            self.warning = (
                "MPS device requested but not available, falling back to CPU"
            )
        else:
            self.device = requested
            self.warning = ""


def get_execution_env(mode: str, device: str) -> dict[str, str]:
    """Return environment variables to inject into the sandbox subprocess.

    Args:
        mode: "quick" or "full".
        device: "cpu" or "mps".

    Returns:
        Dict of ML_CATALOGUE_* environment variables.
    """
    dataset_mode = DatasetMode(mode)
    cfg = _MODE_CONFIG[dataset_mode]
    dev = DeviceConfig(device)

    return {
        "ML_CATALOGUE_MODE": dataset_mode.value,
        "ML_CATALOGUE_SAMPLE_SIZE": cfg["sample_size"],
        "ML_CATALOGUE_DEVICE": dev.device,
        "ML_CATALOGUE_MPS_AVAILABLE": str(dev.mps_available).lower(),
    }


def get_timeout(mode: str) -> float:
    """Return the subprocess timeout in seconds for the given mode."""
    return _MODE_CONFIG[DatasetMode(mode)]["timeout"]


def get_device_warning(device: str) -> str:
    """Return a warning string if the requested device is unavailable."""
    return DeviceConfig(device).warning
