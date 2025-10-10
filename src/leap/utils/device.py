"""Utility functions for PyTorch device detection and management."""

import torch


def get_device(device: str | None = None) -> str:  # noqa: PLR0911
    """Get the appropriate PyTorch device for computation.

    This function provides centralized device detection with support for:
    - CUDA (NVIDIA GPUs)
    - MPS (Apple Silicon M1/M2/M3)
    - CPU (fallback)

    Parameters
    ----------
    device : str | None, optional
        Requested device as a string ("cuda", "mps", or "cpu").
        If None, automatically detects the best available device.
        If specified but not available, falls back to the best available device.

    Returns
    -------
    str
        Device string that can be used with PyTorch operations.
        One of: "cuda", "mps", or "cpu".

    Examples
    --------
    >>> device = get_device()  # Auto-detect best available
    >>> model.to(device)
    >>>
    >>> device = get_device("cuda")  # Request specific device
    >>> tensor = torch.tensor([1, 2, 3]).to(device)

    Notes
    -----
    - Priority order: CUDA > MPS > CPU
    - Returns a string (not torch.device) for simplicity and consistency
    - PyTorch accepts both strings and torch.device objects in .to() methods
    """
    # If no device specified, auto-detect
    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    # Validate requested device
    device_lower = device.lower()

    if device_lower == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        else:
            # Fall back to MPS or CPU
            return get_device(None)

    elif device_lower == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        else:
            # Fall back to CPU
            return "cpu"

    elif device_lower == "cpu":
        return "cpu"

    else:
        # Invalid device specified, auto-detect
        return get_device(None)
