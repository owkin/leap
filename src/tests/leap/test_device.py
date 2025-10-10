"""Tests for device utility functions."""

import torch

from leap.utils.device import get_device


class TestGetDevice:
    """Test suite for get_device function."""

    def test_get_device_returns_string(self):
        """Test that get_device returns a string."""
        device = get_device()
        assert isinstance(device, str)
        assert device in ["cpu", "cuda", "mps"]

    def test_get_device_auto_detect(self):
        """Test automatic device detection."""
        device = get_device()
        # Should return cuda if available, else mps if available, else cpu
        if torch.cuda.is_available():
            assert device == "cuda"
        elif torch.backends.mps.is_available():
            assert device == "mps"
        else:
            assert device == "cpu"

    def test_get_device_cpu_request(self):
        """Test requesting CPU device."""
        device = get_device("cpu")
        assert device == "cpu"

    def test_get_device_cuda_request_when_available(self):
        """Test requesting CUDA when available."""
        device = get_device("cuda")
        if torch.cuda.is_available():
            assert device == "cuda"
        else:
            # Should fallback to mps or cpu
            assert device in ["mps", "cpu"]

    def test_get_device_mps_request_when_available(self):
        """Test requesting MPS when available."""
        device = get_device("mps")
        if torch.backends.mps.is_available():
            assert device == "mps"
        else:
            # Should fallback to cpu
            assert device == "cpu"

    def test_get_device_invalid_request(self):
        """Test that invalid device strings fallback gracefully."""
        device = get_device("invalid")
        assert device in ["cpu", "cuda", "mps"]

    def test_get_device_case_insensitive(self):
        """Test that device strings are case-insensitive."""
        device_upper = get_device("CPU")
        device_lower = get_device("cpu")
        assert device_upper == device_lower == "cpu"

    def test_device_works_with_pytorch(self):
        """Test that returned device string works with PyTorch operations."""
        device = get_device()
        # Should not raise an error
        tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
        assert tensor.device.type == device
