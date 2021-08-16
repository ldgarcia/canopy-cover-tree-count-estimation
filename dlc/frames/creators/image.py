# References
# [1]
#   Kariryaa, A., et al. (2021). PlanetUNet codebase.
#   Code: https://gitlab.com/rscph/planetunet/-/blob/master/preprocessing.py
# [2]
#   Brandt, M., Tucker, C., Kariryaa, A., et al. (2020).
#   An unexpectedly large count of trees in the West African Sahara and Sahel.
#   Nature, 587(7832), 78–82.
#   DOI:  https://doi.org/10.1038/s41586-020-2824-5
#   Code: https://git.io/Jsivl
import pathlib
from typing import Any
from typing import Optional

import rasterio

from dlc.frames.creators.base import FrameDataCreatorResult
from dlc.frames.creators.base import RasterFrameDataCreator
from dlc.frames.creators.data import CoreFrameDataSource
from dlc.tools.cache import BaseArrayCache
from dlc.tools.types import PathSource
from dlc.tools.vendor import raster_copy


class BaseImageFrameCreator(RasterFrameDataCreator):
    def __init__(
        self,
        data: CoreFrameDataSource,
        output_base_path: PathSource,
        *,
        key: str = "image",
        resolution: Optional[Any] = None,
        warp_mem_limit: int = 0,
        only_filename: bool = False,
    ):
        self._data = data
        self._output_base_path = pathlib.Path(output_base_path)
        self._key = key
        self._resolution = resolution
        self._warp_mem_limit = warp_mem_limit
        self._only_filename = only_filename


class ImageFrameCreator(BaseImageFrameCreator):
    """Create an image frame of the intersection of the tile and area."""

    def run(
        self,
        area_id: int,
        tile_id: int,
        *,
        verbose: bool = True,
        overwrite: bool = True,
        cache: Optional[BaseArrayCache] = None,
    ) -> FrameDataCreatorResult:
        self._check_output_base_path()
        output_path = self.output_path(area_id, tile_id)
        if overwrite or not output_path.exists():
            tile = self._data.get_tile(tile_id)
            area = self._data.get_area(area_id)
            if tile["reproject"]:
                # TODO: Handle correct reprojection of the image in
                # the code below
                raise ValueError("Image requires reprojection.")

            input_path = self._data.get_tile_path(tile_id)
            # See: https://git.io/JcD6q
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True):
                with rasterio.open(input_path, "r") as src:
                    window = rasterio.features.geometry_window(
                        src,
                        [area.geometry],
                    )
                    transform = src.window_transform(window)
                    out_meta = src.meta
                    out_meta.update(
                        {
                            "driver": "GTiff",
                            "height": int(window.height),
                            "width": int(window.width),
                            "transform": transform,
                        }
                    )

                    with rasterio.open(output_path, "w", **out_meta) as dst:
                        for i in range(1, src.count + 1):
                            rasterio.warp.reproject(
                                source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=self._data.crs,
                                resampling=rasterio.warp.Resampling.bilinear,
                                dst_resolution=self._resolution,
                                warp_mem_limit=self._warp_mem_limit,
                            )

        return FrameDataCreatorResult(
            tile_id,
            area_id,
            self._key,
            self.output_filename(output_path),
        )


class AltImageFrameCreator(BaseImageFrameCreator):
    """Create an image frame of the intersection of the tile and area."""

    def run(
        self,
        area_id: int,
        tile_id: int,
        *,
        verbose: bool = True,
        overwrite: bool = True,
        cache: Optional[BaseArrayCache] = None,
    ) -> FrameDataCreatorResult:
        self._check_output_base_path()
        output_path = self.output_path(area_id, tile_id)
        if overwrite or not output_path.exists():
            tile = self._data.get_tile(tile_id)
            area = self._data.get_area(area_id)
            if tile["reproject"]:
                # TODO: Handle correct reprojection of the image in
                # the code below
                raise ValueError("Image requires reprojection.")

            input_path = self._data.get_tile_path(tile_id)
            raster_copy(
                str(output_path),
                str(input_path),
                bounds=area.geometry.bounds,
            )

        return FrameDataCreatorResult(
            tile_id,
            area_id,
            self._key,
            self.output_filename(output_path),
        )
