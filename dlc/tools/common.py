"""This module contains common functions used in this package."""
import os
import pathlib
from typing import Optional

import geopandas as gpd

from .types import GeoDataFrameOrigin


def load_geodataframe(
    src: GeoDataFrameOrigin,
    *,
    copy: bool = True,
    driver: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Load a GeoDataframe from a file path or another in-memory dataframe."""
    if not isinstance(src, gpd.GeoDataFrame):
        src = pathlib.Path(src)
        if driver is not None:
            return gpd.read_file(src, driver=driver)
        return gpd.read_file(src)
    return src.copy() if copy else src


def get_n_processes(n_jobs: int, target_n_processes: Optional[int]) -> int:
    n_cpus = os.cpu_count()
    n_processes = min(n_cpus, n_jobs)
    if target_n_processes is not None:
        n_processes = min(n_processes, target_n_processes)
    return n_processes
