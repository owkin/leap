"""Test version information."""

import leap


def test_version_exists() -> None:
    """Test that version string exists."""
    assert hasattr(leap, "__version__")
    assert isinstance(leap.__version__, str)
    assert len(leap.__version__) > 0


def test_version_format() -> None:
    """Test that version follows semantic versioning format."""
    version = leap.__version__
    # Should be either a semantic version or "unknown" for dev installs
    assert version == "unknown" or "." in version or "dev" in version
