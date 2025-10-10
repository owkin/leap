"""Utils for handling configurations."""

import functools
import importlib
from collections import abc
from copy import deepcopy
from typing import Any

from ml_collections import config_dict


def _get_and_pop(config: config_dict.ConfigDict, key: str) -> Any:
    val = None
    if key in config:
        val = config[key]
        del config[key]
    return val


def get_config_dict_copy(config: config_dict.ConfigDict) -> config_dict.ConfigDict:
    """Return an editable copy of an ml_collections configuration dictionary."""
    return config_dict.ConfigDict(deepcopy(config)).unlock()


def load_module(module_path: str | Any) -> abc.Callable:
    """Load a module from its string representation."""
    if isinstance(module_path, str):
        module_name, class_name = module_path.rsplit(".", 1)
        return getattr(importlib.import_module(module_name), class_name)
    return module_path


def instantiate(config: config_dict.ConfigDict, force_partial: bool = False) -> Any:
    """Process config entries that contain _target_ and _partial_ options."""
    if not isinstance(config, config_dict.ConfigDict):
        return config
    config = get_config_dict_copy(config)

    if config.get("_skip_instantiate_"):
        return config

    target = _get_and_pop(config, "_target_")
    target = load_module(target)
    partial = _get_and_pop(config, "_partial_") or force_partial

    for key in config:
        val = instantiate(config[key])
        del config[key]
        config[key] = val

    if target is not None:
        if not partial:
            try:
                return target(**config)
            except TypeError:
                return target.remote(**config)  # type: ignore
            except Exception as e:
                raise ValueError(f"Failed to instantiate {target} with config {config}") from e
        else:
            return functools.partial(target, **config)
    return config
