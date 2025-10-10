"""I/O utilities."""

import pickle
from pathlib import Path
from typing import Any


def load_pickle(path: Path) -> Any:
    """Load a pickle file.

    Parameters
    ----------
    path : Path
        Path to the pickle file.

    Returns
    -------
    Any
        The loaded object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(object: Any, path: Path) -> None:
    """Save an object to a pickle file.

    Parameters
    ----------
    object : Any
        Object to save.
    path : Path
        Path to the pickle file.
    """
    with open(path, "wb") as f:
        pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)
