# References
# [1]
#   Kariryaa, A., et al. (2021). PlanetUNet codebase.
# [2]
#   Brandt, M., Tucker, C., Kariryaa, A., et al. (2020).
#   An unexpectedly large count of trees in the West African Sahara and Sahel.
#   Nature, 587(7832), 78–82.
#   DOI:  https://doi.org/10.1038/s41586-020-2824-5
#   Code: https://git.io/Jsivl
import pathlib
from typing import Optional

import numpy as np
import rasterio.features

from dlc.frames.creators.base import FrameDataCreatorResult
from dlc.frames.creators.base import RasterFrameDataCreator
from dlc.frames.creators.data import CoreFrameDataSource
from dlc.frames.raster import raster_max
from dlc.frames.raster import raster_min
from dlc.frames.raster import raster_write
from dlc.tools.cache import BaseArrayCache
from dlc.tools.types import PathSource
from dlc.tools.vendor import calculate_boundary_weights


class BaseSegmentationMaskFrameCreator(RasterFrameDataCreator):
    """Create the segmentation mask for an area.

    Parameters
    ----------
    data
        The consolidated data source.
    output_base_path
        The path to the folder in which to store the created frames.
    key
        The key for the result.

    Examples
    --------
    >>> creator = SegmentationMaskFrameCreator(data, "./frames")
    >>> result = creator.run(area_id, tile_id)

    Notes
    -----
    Adapted from [1].
    """

    def __init__(
        self,
        data: CoreFrameDataSource,
        output_base_path: PathSource,
        *,
        key: str = "segmentation-mask",
        dtype: str = "float32",
        only_filename: bool = False,
    ):
        self._data = data
        self._output_base_path = pathlib.Path(output_base_path)
        self._key = key
        self._dtype = dtype
        self._only_filename = only_filename


class SegmentationMaskFrameCreator(BaseSegmentationMaskFrameCreator):
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
        area_data = self._data.get_area_raster_data(area_id, tile_id)
        segmentation_mask = np.zeros(area_data.shape, dtype=self._dtype)
        polygons = self._data.get_polygons(tile_id, area_id)
        for polygon_id, _ in polygons.iterrows():
            polygon_data = self._data.get_polygon_raster_data(
                area_id,
                tile_id,
                polygon_id,
            )
            # We add the polygon binary mask to the segmentation mask.
            raster_max(
                polygon_data.raster,
                polygon_data.transform,
                segmentation_mask,
                area_data.transform,
            )
        raster_write(
            output_path,
            segmentation_mask,
            area_data.transform,
            area_data.crs,
        )
        canopy_cover = 0.0
        if segmentation_mask.size > 0.0:
            canopy_cover = np.true_divide(
                np.sum(segmentation_mask), segmentation_mask.size
            )
        if verbose:
            print(
                "Polygon count: {}, Canopy cover: {:.2f}".format(
                    len(polygons),
                    canopy_cover,
                )
            )
        return FrameDataCreatorResult(
            tile_id,
            area_id,
            self._key,
            dict(
                path=self.output_filename(output_path),
                canopy_cover=canopy_cover,
            ),
            payload_main_key="path",
            payload_no_prefix_keys=("canopy_cover",),
        )


class AltSegmentationMaskFrameCreator(BaseSegmentationMaskFrameCreator):
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
            area = self._data.get_area(area_id)
            area_data = self._data.get_area_raster_data(area_id, tile_id)
            polygons = self._data.get_polygons(tile_id, area_id)
            segmentation_mask = np.zeros(area_data.shape, dtype=self._dtype)
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True):
                input_path = self._data.get_tile_path(tile_id)
                with rasterio.open(input_path, "r") as src:
                    window = rasterio.features.geometry_window(
                        src,
                        [area.geometry],
                    )
                    transform = src.window_transform(window)
                    if len(polygons) > 0:
                        segmentation_mask[:] = rasterio.features.geometry_mask(
                            polygons.geometry,
                            out_shape=segmentation_mask.shape,
                            transform=transform,
                            invert=True,
                            all_touched=True,
                        ).astype(self._dtype)
                    raster_write(
                        output_path,
                        segmentation_mask,
                        transform,
                        self._data.crs,
                    )
            canopy_cover = 0.0
            if len(polygons) > 0 and segmentation_mask.size > 0.0:
                canopy_cover = np.sum(segmentation_mask)
                canopy_cover /= segmentation_mask.size
            if verbose:
                print(
                    "Polygon count: {}, Canopy cover: {:.2f}".format(
                        len(polygons),
                        canopy_cover,
                    )
                )
        return FrameDataCreatorResult(
            tile_id,
            area_id,
            self._key,
            dict(
                path=self.output_filename(output_path),
                canopy_cover=canopy_cover,
            ),
            payload_main_key="path",
            payload_no_prefix_keys=("canopy_cover",),
        )


class SegmentationBoundaryWeightsFrameCreator(RasterFrameDataCreator):
    """Create boundary weight masks between close polygons.

    Parameters
    ----------
    data
        The consolidated data source.
    output_base_path
        The path to the folder in which to store the created frames.
    key
        The key for the result.
    scale: optional
        The scale to expand the polygons.

    Examples
    --------
    >>> creator = SegmentationBoundaryWeightsFrameCreator(data, "./frames")
    >>> result = creator.run(area_id, tile_id)

    Notes
    -----
    Adapted from [1].
    """

    def __init__(
        self,
        data: CoreFrameDataSource,
        output_base_path: PathSource,
        *,
        scale: float = 1.5,
        non_boundary_weight: float = 0.0,
        boundary_weight: float = 1.0,
        key: str = "segmentation-boundary-weights",
        dtype: str = "float32",
        only_filename: bool = False,
    ):
        self._data = data
        self._output_base_path = pathlib.Path(output_base_path)
        self._key = key
        self._non_boundary_weight = non_boundary_weight
        self._boundary_weight = boundary_weight
        self._scale = scale
        self._dtype = dtype
        self._only_filename = only_filename

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
            area = self._data.get_area(area_id)
            area_data = self._data.get_area_raster_data(area_id, tile_id)
            polygons = self._data.get_polygons(tile_id, area_id)
            weights_mask = np.zeros(area_data.shape, dtype=self._dtype)
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True):
                input_path = self._data.get_tile_path(tile_id)
                with rasterio.open(input_path, "r") as src:
                    window = rasterio.features.geometry_window(
                        src,
                        [area.geometry],
                    )
                    transform = src.window_transform(window)
                    if len(polygons) > 0:
                        boundary_weights = calculate_boundary_weights(
                            polygons,
                            self._scale,
                        )
                        weights_mask[:] = rasterio.features.geometry_mask(
                            boundary_weights.geometry,
                            out_shape=weights_mask.shape,
                            transform=transform,
                            invert=True,
                            all_touched=True,
                        ).astype(self._dtype)
                        weights_mask[:] = np.where(
                            weights_mask == 1.0,
                            self._boundary_weight,
                            self._non_boundary_weight,
                        )
                    raster_write(
                        output_path,
                        weights_mask,
                        transform,
                        self._data.crs,
                    )

        return FrameDataCreatorResult(
            tile_id,
            area_id,
            self._key,
            self.output_filename(output_path),
        )


class OutlierWeightsMaskFrameCreator(RasterFrameDataCreator):
    def __init__(
        self,
        data: CoreFrameDataSource,
        output_base_path: PathSource,
        *,
        query: str = "is_outlier == True",
        key: str = "outlier-weights",
        dtype: str = "float32",
        only_filename: bool = False,
    ):
        self._data = data
        self._output_base_path = pathlib.Path(output_base_path)
        self._key = key
        self._dtype = dtype
        self._only_filename = only_filename
        self._query = query

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
            area_data = self._data.get_area_raster_data(area_id, tile_id)
            weights_mask = np.ones(area_data.shape, dtype=self._dtype)
            polygons = self._data.get_polygons(tile_id, area_id)
            outlier_polygons = polygons.query(self._query)
            for polygon_id, _ in outlier_polygons.iterrows():
                polygon_data = self._data.get_polygon_raster_data(
                    area_id,
                    tile_id,
                    polygon_id,
                )
                # We invert the polygon binary mask
                weights = np.where(polygon_data.raster > 0.0, 0.0, 1.0)
                raster_min(
                    weights,
                    polygon_data.transform,
                    weights_mask,
                    area_data.transform,
                )
            raster_write(
                output_path,
                weights_mask,
                area_data.transform,
                area_data.crs,
            )
        return FrameDataCreatorResult(
            tile_id,
            area_id,
            self._key,
            self.output_filename(output_path),
        )
