"""LEAP: Layered Ensemble of Autoencoders and Predictors."""

try:
    from leap._version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["__version__"]
