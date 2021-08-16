import abc
import math
import pathlib
from typing import Optional
from typing import Tuple

import affine
import numpy as np
import rasterio
import scipy.signal
import skimage.filters

from dlc.frames.centroids import centroid_mask
from dlc.frames.centroids import energy_centroid
from dlc.frames.centroids import get_min_segment_distance
from dlc.frames.centroids import standard_centroid
from dlc.frames.creators.base import FrameDataCreatorResult
from dlc.frames.creators.base import RasterFrameDataCreator
from dlc.frames.creators.data import CoreFrameDataSource
from dlc.frames.creators.data import PolygonRasterData
from dlc.frames.raster import raster_add
from dlc.frames.raster import raster_write
from dlc.tools.cache import BaseArrayCache
from dlc.tools.cache import DummyCache
from dlc.tools.types import PathSource


def apply_threshold(data: np.ndarray, thresh_z_score: Optional[float]):
    # See: https://en.wikipedia.org/wiki/Standard_score
    if thresh_z_score is not None:
        n_keep = 0
        while n_keep == 0:
            mean = np.mean(data)
            std = np.std(data)
            thresh = mean + (thresh_z_score * std)
            throw_idx = data < thresh
            keep_idx = np.logical_not(throw_idx)
            n_keep = float(len(data[keep_idx]))
            if n_keep > 0.0:
                data[keep_idx] += np.sum(data[throw_idx]) / n_keep
                data[throw_idx] = 0.0
            else:
                thresh_z_score -= 0.01
    return data


def apply_mask(data: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    keep_idx = np.logical_and(binary_mask == 1.0, data != 0.0)
    throw_idx = np.logical_not(keep_idx)
    n_keep = float(len(data[keep_idx]))
    if n_keep > 0.0:
        data[keep_idx] += np.sum(data[throw_idx]) / n_keep
    data[throw_idx] = 0.0
    return data


def apply_normalization(data: np.ndarray, *, copy: bool = True) -> np.ndarray:
    if copy:
        target = data.copy()
    else:
        target = data
    target_sum = np.sum(target)
    if target_sum > 0.0:
        target /= target_sum
    return target


def get_gaussian_filter(filter_size: int, sigma: float) -> np.ndarray:
    filter_1d = scipy.signal.gaussian(filter_size, sigma)
    filter_2d = np.outer(filter_1d, filter_1d)
    return filter_2d


def apply_filter(
    target: np.ndarray,
    filter: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    pad_h = filter.shape[0] - 1
    pad_w = filter.shape[1] - 1
    target_padded = np.pad(target, (pad_h, pad_w))
    target_filtered = scipy.signal.convolve2d(
        filter,
        target_padded,
        mode="valid",
        boundary="fill",
        fillvalue=0.0,
    )
    output = target_filtered[pad_h // 2 : -pad_h // 2, pad_w // 2 : -pad_w // 2]
    if normalize:
        output = apply_normalization(output)
    return output


def apply_alt_gaussian_filter(
    binary_mask: np.ndarray,
    sigma: float,
    thresh_z_score: Optional[float] = None,
    *,
    normalize: bool = True,
) -> np.ndarray:
    result = skimage.filters.gaussian(binary_mask, sigma, mode="nearest")
    result = apply_threshold(result, thresh_z_score)
    if normalize:
        result = apply_normalization(result)
    return result


def edt_transform(
    binary_mask: np.ndarray,
    *,
    pad_width: int = 20,
    thresh_z_score: Optional[float] = None,
) -> np.ndarray:
    if pad_width < 0:
        raise ValueError("Invalid pad_width")
    # We need to pad the image so that each boundary pixel has a
    # background pixel as neighbor, and the transform is computed
    # correctly.
    target_padded = np.pad(binary_mask, pad_width)
    energy_map = scipy.ndimage.distance_transform_edt(target_padded)
    energy_map = energy_map[pad_width:-pad_width, pad_width:-pad_width]
    energy_map = apply_threshold(energy_map, thresh_z_score)
    energy_map = apply_normalization(energy_map)
    return energy_map


def uniform_transform(target: np.ndarray) -> np.ndarray:
    return apply_normalization(target, copy=True)


class DensityFrameCreator(RasterFrameDataCreator):
    _output_base_path: pathlib.Path
    _data: CoreFrameDataSource
    _dtype: str
    _use_padded_bbox: bool

    @abc.abstractmethod
    def _run_polygon(
        self,
        data: PolygonRasterData,
        cache: BaseArrayCache,
    ) -> Tuple[np.ndarray, affine.Affine]:
        pass

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
        min_pixel_value = None
        max_pixel_value = None
        if overwrite or not output_path.exists():
            # The dummy cache save us from some logic duplication
            # when we do not want to use a cache.
            if cache is None:
                cache = DummyCache()
            area_data = self._data.get_area_raster_data(area_id, tile_id)
            density_map = np.zeros(area_data.shape, dtype=self._dtype)
            polygons = self._data.get_polygons(tile_id, area_id)
            for polygon_id, _ in polygons.iterrows():
                polygon_data = self._data.get_polygon_raster_data(
                    area_id,
                    tile_id,
                    polygon_id,
                )
                # We get a density map for each polygon
                # that should sum up to one (unit normalized).
                polygon_density, transform = self._run_polygon(
                    polygon_id,
                    polygon_data,
                    cache,
                )
                if verbose:
                    polygon_sum = np.sum(polygon_density)
                    if not math.isclose(1.0, polygon_sum, rel_tol=1e-3):
                        print(
                            "Warning: Density map sum for polygon ({}) sums up to: {:.4f}".format(
                                polygon_id,
                                polygon_sum,
                            )
                        )
                # We add the polygon density map to the frame density map.
                raster_add(
                    polygon_density,
                    transform,
                    density_map,
                    area_data.transform,
                )
            raster_write(
                output_path,
                density_map,
                area_data.transform,
                area_data.crs,
            )
            min_pixel_value = np.min(density_map)
            max_pixel_value = np.max(density_map)
            if verbose:
                print(
                    "Polygon count: {}, Density map sum: {:.2f}".format(
                        len(polygons),
                        np.sum(density_map),
                    )
                )
                print("Max pixel value: {}".format(np.max(density_map)))

        return FrameDataCreatorResult(
            tile_id,
            area_id,
            self._key,
            payload=self.output_filename(output_path),
            meta=dict(
                min_pixel_value=min_pixel_value,
                max_pixel_value=max_pixel_value,
            ),
        )

    def _get_raster(
        self, data: PolygonRasterData
    ) -> Tuple[np.ndarray, affine.Affine, str]:
        if self._use_padded_bbox:
            cache_suffix = "-padded"
            polygon_mask = data.raster_padded
            transform = data.transform_padded
            if polygon_mask is None:
                raise ValueError("raster_padded is None")
        else:
            cache_suffix = ""
            polygon_mask = data.raster
            transform = data.transform

        return polygon_mask, transform, cache_suffix


class GaussianDensityFrameCreator(DensityFrameCreator):
    def __init__(
        self,
        data: CoreFrameDataSource,
        output_base_path: PathSource,
        *,
        sigma: float = 1.0,
        filter_size: Optional[int] = None,
        filter_target: str = "centroid",
        centroid_type: str = "standard",
        key: Optional[str] = None,
        dtype: str = "float32",
        use_padded_bbox: bool = True,
        only_filename: bool = False,
    ):
        self._data = data
        self._filter_size = filter_size
        if self._filter_size is not None and self._filter_size <= 0.0:
            raise ValueError("Invalid filter size.")
        self._sigma = sigma
        if self._sigma <= 0.0:
            raise ValueError("Invalid sigma.")
        self._dtype = dtype

        if filter_target not in ("centroid", "polygon"):
            raise ValueError("Invalid filter_target.")
        self._filter_target = filter_target

        if centroid_type not in ("standard", "energy"):
            raise ValueError("Invalid centroid type.")
        self._centroid_type = centroid_type

        if key is not None:
            self._key = key
        else:
            parts = ["gaussian-density"]
            parts.append(f"sgm_{self._sigma:.2f}")
            if self._filter_size is None:
                parts.append("fs_adaptive")
            else:
                parts.append(f"fs_{self._filter_size:.2f}")
            if self._filter_target == "centroid":
                parts.append("ftc")
            else:
                parts.append("ftp")
            if self._filter_target == "centroid" or self._filter_size is None:
                if self._centroid_type == "standard":
                    parts.append("sc")
                else:
                    parts.append("ec")
            self._key = self._compose_key(parts)

        self._output_base_path = pathlib.Path(output_base_path)
        self._only_filename = only_filename
        self._use_padded_bbox = use_padded_bbox

    def _run_polygon(
        self,
        polygon_id: int,
        data: PolygonRasterData,
        cache: BaseArrayCache,
    ) -> Tuple[np.ndarray, affine.Affine]:
        polygon_mask, transform, cache_suffix = self._get_raster(data)

        if self._centroid_type == "standard":
            centroid = cache.get_or_create(
                self._cache_key(
                    f"standard-centroid{cache_suffix}",
                    polygon_id,
                ),
                lambda: standard_centroid(polygon_mask),
            )
        else:
            polygon_energy_map = cache.get_or_create(
                self._cache_key(
                    f"energy-map{cache_suffix}",
                    polygon_id,
                ),
                lambda: edt_transform(polygon_mask),
            )
            centroid = cache.get_or_create(
                self._cache_key(
                    f"energy-centroid{cache_suffix}",
                    polygon_id,
                ),
                lambda: energy_centroid(polygon_energy_map),
            )

        if self._filter_size is None:
            filter_size = min(*polygon_mask.shape)
        else:
            filter_size = self._filter_size

        if self._filter_target == "centroid":
            target = centroid_mask(polygon_mask, centroid)
        else:
            target = polygon_mask

        target_filtered = apply_filter(
            target,
            get_gaussian_filter(filter_size, self._sigma),
        )
        return target_filtered, transform


class EDTDensityFrameCreator(DensityFrameCreator):
    def __init__(
        self,
        data: CoreFrameDataSource,
        output_base_path: PathSource,
        *,
        key: Optional[str] = None,
        thresh_z_score: Optional[float] = None,
        dtype: str = "float32",
        only_filename: bool = False,
        use_padded_bbox: bool = True,
    ):
        self._data = data
        self._thresh_z_score = thresh_z_score
        if self._thresh_z_score is not None and self._thresh_z_score < 0.0:
            raise ValueError("Invalid thresh_z_score.")
        self._dtype = dtype

        if key is not None:
            self._key = key
        else:
            parts = ["edt-density"]
            if self._thresh_z_score is not None:
                parts.append(f"th_{self._thresh_z_score:.2f}")
            self._key = self._compose_key(parts)

        self._output_base_path = pathlib.Path(output_base_path)
        self._only_filename = only_filename
        self._use_padded_bbox = use_padded_bbox

    def _run_polygon(
        self,
        polygon_id: int,
        data: PolygonRasterData,
        cache: BaseArrayCache,
    ) -> Tuple[np.ndarray, affine.Affine]:
        polygon_mask, transform, cache_suffix = self._get_raster(data)
        thresh_key = self._thresh_z_score or 0.0
        target_transformed = cache.get_or_create(
            self._cache_key(
                f"energy-map-th_{thresh_key:.2f}{cache_suffix}",
                polygon_id,
            ),
            lambda: edt_transform(
                polygon_mask,
                thresh_z_score=self._thresh_z_score,
            ),
        )
        return target_transformed, transform


class UniformDensityFrameCreator(DensityFrameCreator):
    def __init__(
        self,
        data: CoreFrameDataSource,
        output_base_path: PathSource,
        *,
        key: Optional[str] = None,
        dtype: str = "float32",
        only_filename: bool = False,
    ):
        self._data = data
        self._dtype = dtype

        if key is not None:
            self._key = key
        else:
            parts = ["uniform-density"]
            self._key = self._compose_key(parts)

        self._output_base_path = pathlib.Path(output_base_path)
        self._only_filename = only_filename
        self._use_padded_bbox = False

    def _run_polygon(
        self,
        _polygon_id: int,
        data: PolygonRasterData,
        _cache: BaseArrayCache,
    ) -> Tuple[np.ndarray, affine.Affine]:
        return uniform_transform(data.raster), data.transform


class THGaussianDensityFrameCreator(DensityFrameCreator):
    def __init__(
        self,
        data: CoreFrameDataSource,
        output_base_path: PathSource,
        *,
        sigma: float = 1.0,
        thresh_z_score: Optional[float] = None,
        filter_target: str = "centroid",
        centroid_type: str = "standard",
        key: Optional[str] = None,
        dtype: str = "float32",
        only_filename: bool = False,
        use_padded_bbox: bool = True,
    ):
        self._data = data
        self._thresh_z_score = thresh_z_score
        if self._thresh_z_score is not None and self._thresh_z_score <= 0.0:
            raise ValueError("Invalid thresh_z_score.")
        self._sigma = sigma
        if self._sigma <= 0.0:
            raise ValueError("Invalid sigma.")
        self._dtype = dtype

        if filter_target not in ("centroid", "polygon"):
            raise ValueError("Invalid filter_target.")
        self._filter_target = filter_target

        if centroid_type not in ("standard", "energy"):
            raise ValueError("Invalid centroid type.")
        self._centroid_type = centroid_type

        if key is not None:
            self._key = key
        else:
            parts = ["th-gaussian-density"]
            if self._sigma is None:
                parts.append("sgm_adaptive")
            else:
                parts.append(f"sgm_{self._sigma:.2f}")
            if self._thresh_z_score is not None:
                parts.append(f"tz_{self._thresh_z_score:.2f}")
            if self._filter_target == "centroid":
                parts.append("ftc")
            else:
                parts.append("ftp")
            if self._filter_target == "centroid" or self._sigma is None:
                if self._centroid_type == "standard":
                    parts.append("sc")
                else:
                    parts.append("ec")
            self._key = self._compose_key(parts)

        self._output_base_path = pathlib.Path(output_base_path)
        self._only_filename = only_filename
        self._use_padded_bbox = use_padded_bbox

    def _run_polygon(
        self,
        polygon_id: int,
        data: PolygonRasterData,
        cache: BaseArrayCache,
    ) -> Tuple[np.ndarray, affine.Affine]:
        polygon_mask, transform, cache_suffix = self._get_raster(data)

        if self._centroid_type == "standard":
            centroid = cache.get_or_create(
                self._cache_key(
                    f"standard-centroid{cache_suffix}",
                    polygon_id,
                ),
                lambda: standard_centroid(polygon_mask),
            )
        else:
            polygon_energy_map = cache.get_or_create(
                self._cache_key(
                    f"energy-map{cache_suffix}",
                    polygon_id,
                ),
                lambda: edt_transform(polygon_mask),
            )
            centroid = cache.get_or_create(
                self._cache_key(
                    f"energy-centroid{cache_suffix}",
                    polygon_id,
                ),
                lambda: energy_centroid(polygon_energy_map),
            )

        sigma = self._sigma
        if sigma is None:
            sigma = get_min_segment_distance(centroid, polygon_mask) * 0.5
        if self._filter_target == "centroid":
            target = centroid_mask(polygon_mask, centroid)
        else:
            target = polygon_mask

        target_filtered = apply_alt_gaussian_filter(
            target,
            sigma,
            thresh_z_score=self._thresh_z_score,
        )

        return target_filtered, transform


class DMGaussianDensityFrameCreator(RasterFrameDataCreator):
    def __init__(
        self,
        data: CoreFrameDataSource,
        output_base_path: PathSource,
        *,
        sigma: float = 1.0,
        filter_size: int = None,
        key: Optional[str] = None,
        dtype: str = "float32",
        only_filename: bool = False,
    ):
        self._data = data
        self._filter_size = filter_size
        if self._filter_size is not None and self._filter_size <= 0.0:
            raise ValueError("Invalid filter size.")
        self._sigma = sigma
        if self._sigma <= 0.0:
            raise ValueError("Invalid sigma.")
        if key is not None:
            self._key = key
        else:
            parts = ["dm-gaussian-density"]
            parts.append(f"sgm_{self._sigma:.2f}")
            parts.append(f"fs_{self._filter_size:.2f}")
            self._key = self._compose_key(parts)
        self._dtype = dtype
        self._output_base_path = pathlib.Path(output_base_path)
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
        min_pixel_value = None
        max_pixel_value = None
        if overwrite or not output_path.exists():
            area = self._data.get_area(area_id)
            area_data = self._data.get_area_raster_data(area_id, tile_id)
            polygons = self._data.get_polygons(tile_id, area_id)
            density_map = np.zeros(area_data.shape, dtype=self._dtype)
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True):
                input_path = self._data.get_tile_path(tile_id)
                with rasterio.open(input_path, "r") as src:
                    window = rasterio.features.geometry_window(
                        src,
                        [area.geometry],
                    )
                    transform = src.window_transform(window)
                    if len(polygons) > 0:
                        centroids = polygons.to_crs("EPSG:6933").centroid.to_crs(
                            polygons.crs
                        )
                        pad_width = self._filter_size * 2
                        for centroid in centroids:
                            dot_map = rasterio.features.geometry_mask(
                                [centroid],
                                out_shape=density_map.shape,
                                transform=transform,
                                invert=True,
                                all_touched=True,
                            ).astype(self._dtype)
                            # Apply the Gaussian filter on a window of the
                            # frame raster that contains the dot (centroid).
                            nonzero = np.argwhere(dot_map)
                            if len(nonzero) > 0:
                                centroid = nonzero[0]
                                bottom = max(0, centroid[0] - pad_width)
                                top = min(dot_map.shape[0], centroid[0] + pad_width)
                                left = max(0, centroid[1] - pad_width)
                                right = min(dot_map.shape[1], centroid[1] + pad_width)
                                hw = slice(bottom, top)
                                ww = slice(left, right)
                                dot_map[hw, ww] = apply_filter(
                                    dot_map[hw, ww],
                                    get_gaussian_filter(
                                        self._filter_size,
                                        self._sigma,
                                    ),
                                )
                                density_map += dot_map
                    raster_write(
                        output_path,
                        density_map,
                        transform,
                        self._data.crs,
                    )
            max_pixel_value = np.max(density_map)
            min_pixel_value = np.min(density_map)
            if verbose:
                print(
                    "Polygon count: {}, Density map sum: {:.2f}".format(
                        len(polygons),
                        np.sum(density_map),
                    )
                )
                print("Max pixel value: {}".format(max_pixel_value))

        return FrameDataCreatorResult(
            tile_id,
            area_id,
            self._key,
            payload=self.output_filename(output_path),
            meta=dict(
                min_pixel_value=min_pixel_value,
                max_pixel_value=max_pixel_value,
            ),
        )
