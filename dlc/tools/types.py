"""This module contains common type definitions."""
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Union

from geopandas import GeoDataFrame

GeoDataFrameOrigin = Union[Path, str, GeoDataFrame]

PathSource = Union[Path, str]

CountHeuristicFuntion = Callable[[Any], float]

__all__ = [
    "GeoDataFrameSource",
    "PathSource",
    "CountHeuristicFuntion",
]
