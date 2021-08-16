import os
import pathlib
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

import yaml

__all__ = ["read_config"]


def _update_recursive(a: dict, b: dict, depth: int = 0) -> dict:
    """Update a with values from b recursively."""
    for k, v in b.items():
        # special keys that are accumulated in order
        acc_keys = ["model_name_suffix", "custom_objects"]
        if k in acc_keys and depth == 0 and isinstance(v, Iterable):
            if k not in a:
                a[k] = v
            else:
                a[k] += v
        else:
            if k not in a:
                a[k] = v
            elif isinstance(a[k], dict) and isinstance(v, dict):
                a[k] = _update_recursive(a[k], v, depth + 1)
            else:
                a[k] = v
    return a


def read_config(config_paths: Optional[List[str]] = None) -> dict:
    """Build a settings dictionary from a list of configuration files."""
    if config_paths is None:
        config_paths = os.environ.get("config_paths", "").split(",")
    if len(config_paths) == 0:
        raise ValueError("No configuration paths provided")
    config: Dict[str, Any] = dict()
    for config_path_str in config_paths:
        config_path = pathlib.Path(config_path_str)
        if not config_path.exists():
            raise ValueError(f"{config_path_str} does not exist.")
        with open(config_path, "r") as src:
            config = _update_recursive(
                config,
                yaml.load(src, Loader=yaml.Loader),
            )
    return config
