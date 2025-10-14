"""Pytest configuration for LEAP tests.

This module configures pytest to avoid segmentation faults caused by
multi-threaded operations in PyTorch and numpy libraries.
"""

import os


# Set environment variables to prevent threading issues that cause segfaults
# These must be set before importing numpy/torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Import torch after setting environment variables
import torch


# Set torch to use single thread
torch.set_num_threads(1)
