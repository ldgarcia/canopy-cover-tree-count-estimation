from typing import Optional

import geopandas as gpd
import shapely.geometry

from dlc.frames.creators.base import FrameDataCreator
from dlc.frames.creators.base import FrameDataCreatorResult
from dlc.frames.creators.data import CoreFrameDataSource
from dlc.tools.cache import BaseArrayCache


class FramePropertiesDataCreator(FrameDataCreator):
    """Compute some scalar information for an area.

    Parameters
    ----------
    data
        The consolidated data source.
    key
        The key for the result.

    Examples
    --------
    >>> creator = FramePropertiesDataCreator(data)
    >>> result = creator.run(area_id)
    >>> result.payload["width"]
    """

    def __init__(
        self,
        data: CoreFrameDataSource,
        *,
        key: Optional[str] = None,
    ):
        self._data = data
        if key is not None:
            self._key = key
        else:
            parts = ["props"]
            self._key = self._compose_key(parts)

    def run(
        self,
        area_id: int,
        tile_id: int,
        *,
        verbose: bool = True,
        overwrite: bool = True,
        cache: Optional[BaseArrayCache] = None,
    ) -> FrameDataCreatorResult:
        area = self._data.get_area(area_id)
        area_data = self._data.get_area_raster_data(area_id, tile_id)
        polygons = self._data.get_polygons(tile_id, area_id)

        return FrameDataCreatorResult(
            tile_id,
            area_id,
            self._key,
            dict(
                n_polygons=len(polygons),
                height=area_data.shape[0],
                width=area_data.shape[1],
                original_split=area.get("split", None),
            ),
            payload_no_prefix_keys=(
                "height",
                "width",
                "n_polygons",
                "original_split",
            ),
        )


class SaharaSahelFramePropertiesDataCreator(FrameDataCreator):
    """Specific properties for frames of the Sahara and Sahel-Sudan dataset."""

    def __init__(
        self,
        data: CoreFrameDataSource,
        *,
        key: str = "sahara-sahel-props",
    ):
        self._data = data
        self._key = key

    def run(
        self,
        area_id: int,
        tile_id: int,
        *,
        verbose: bool = True,
        overwrite: bool = True,
        cache: Optional[BaseArrayCache] = None,
    ) -> FrameDataCreatorResult:
        area = self._data.get_area(area_id)

        return FrameDataCreatorResult(
            tile_id,
            area_id,
            self._key,
            dict(region=area["region"]),
            payload_no_prefix_keys=("region"),
        )
