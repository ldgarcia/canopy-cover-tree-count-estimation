import dataclasses
import multiprocessing
import pathlib
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

import affine
import numpy as np
import rasterio.crs
import rasterio.features
import shapely.affinity
import shapely.geometry

from dlc.tools.common import get_n_processes
from dlc.tools.common import load_geodataframe
from dlc.tools.db import get_areas_with_tiles
from dlc.tools.db import get_polygons_with_tile_and_area
from dlc.tools.db import get_tiles_with_areas
from dlc.tools.types import GeoDataFrameOrigin
from dlc.tools.types import PathSource


@dataclasses.dataclass
class AreaRasterData:
    transform: affine.Affine
    shape: Tuple[int, int]
    crs: rasterio.crs.CRS


@dataclasses.dataclass
class PolygonRasterData:
    transform: affine.Affine
    raster: affine.Affine
    transform_padded: Optional[affine.Affine] = None
    raster_padded: Optional[np.ndarray] = None


@dataclasses.dataclass
class AreaRasterDataJobResult:
    area_id: int
    tile_id: int
    transform: affine.Affine
    shape: Tuple[int, int]
    crs: rasterio.crs.CRS


@dataclasses.dataclass
class PolygonRasterDataJobResult:
    polygon_id: int
    transform: affine.Affine
    raster: np.ndarray
    transform_padded: Optional[affine.Affine] = None
    raster_padded: Optional[np.ndarray] = None


@dataclasses.dataclass
class RasterDataJobResult:
    area: AreaRasterDataJobResult
    polygons: List[PolygonRasterDataJobResult]


AreaKey = Tuple[int, int]


class CoreFrameDataSource:
    def __init__(
        self,
        name: str,
        tiles_base_path: PathSource,
        tiles: GeoDataFrameOrigin,
        areas: GeoDataFrameOrigin,
        polygons: GeoDataFrameOrigin,
        pad_scale: Optional[float] = 2.0,
        *,
        dtype: str = "float32",
    ):
        self._name = name
        self._tiles_base_path = pathlib.Path(tiles_base_path)
        self._tiles = load_geodataframe(tiles, copy=True)
        self._areas = load_geodataframe(areas, copy=True)
        self._areas["loaded"] = False
        self._polygons = load_geodataframe(polygons, copy=True)
        self._polygons["loaded"] = False
        self._polygon_raster_data: Dict[int, PolygonRasterData] = dict()
        self._pad_scale = pad_scale
        self._area_raster_data: Dict[AreaKey, AreaRasterData] = dict()
        if "area_ids" not in self._tiles:
            self._tiles = get_tiles_with_areas(self._tiles, self._areas)
        if "tile_ids" not in self._areas:
            self._areas = get_areas_with_tiles(self._areas, self._tiles)
        if "area_id" not in self._polygons:
            self._polygons = get_polygons_with_tile_and_area(
                self._polygons,
                self._areas,
                self._tiles,
            )
        self._dtype = dtype
        self._rng = np.random.default_rng()

    @property
    def name(self):
        return self._name

    @property
    def crs(self) -> rasterio.crs.CRS:
        return self._areas.crs

    def get_tile_path(
        self,
        tile_id: int,
        check: bool = True,
    ) -> pathlib.Path:
        tile_filepath = self._tiles_base_path.joinpath(
            self._tiles.loc[tile_id, "file_path"],
        )
        if check and not tile_filepath.exists():
            msg = f"File {tile_filepath} does not exist for tile {tile_id}."
            raise Exception(msg)
        return tile_filepath

    def get_tile(self, tile_id: int):
        return self._tiles.iloc[tile_id]

    def get_area(self, area_id: int):
        return self._areas.query(f"id == {area_id}").iloc[0]

    def get_polygons(self, tile_id: int, area_id: int):
        return self._polygons.query(
            "tile_id == {} and area_id == {}".format(tile_id, area_id)
        )

    def get_random_polygon_raster_data(self) -> PolygonRasterData:
        # Primarily used to debug other functions
        # that work on polygon rasters (e.g. an EDT transform).
        keys = list(self._polygon_raster_data.keys())
        if len(keys) == 0:
            raise ValueError("No data")
        key_idx = self._rng.integers(0, len(keys), 1)[0]
        return self._polygon_raster_data[keys[key_idx]]

    def get_polygon_raster_data(
        self,
        area_id: int,
        tile_id: int,
        polygon_id: int,
    ) -> PolygonRasterData:
        area_key = self._get_area_key(area_id, tile_id)
        if area_key not in self._area_raster_data:
            result = self._load_raster_data(area_key)
            self._process_result(result)

        return self._polygon_raster_data[polygon_id]

    def get_area_raster_data(self, area_id: int, tile_id: int) -> AreaRasterData:
        area_key = self._get_area_key(area_id, tile_id)
        if area_key not in self._area_raster_data:
            result = self._load_raster_data(area_key)
            self._process_result(result)

        return self._area_raster_data[area_key]

    def _get_area_key(self, area_id: int, tile_id: int) -> AreaKey:
        return (area_id, tile_id)

    def _process_result(self, result: RasterDataJobResult) -> None:
        area_key = self._get_area_key(result.area.area_id, result.area.tile_id)
        self._area_raster_data[area_key] = AreaRasterData(
            result.area.transform,
            result.area.shape,
            result.area.crs,
        )
        for polygon_result in result.polygons:
            poly_key = polygon_result.polygon_id
            self._polygon_raster_data[poly_key] = PolygonRasterData(
                transform=polygon_result.transform,
                raster=polygon_result.raster,
                transform_padded=polygon_result.transform_padded,
                raster_padded=polygon_result.raster_padded,
            )
            self._polygons.loc[polygon_result.polygon_id, "loaded"] = True
        self._areas.loc[result.area.area_id, "loaded"] = True

    def _get_geometry_raster(
        self,
        tile: rasterio.DatasetReader,
        window_geometry: shapely.geometry.Polygon,
        raster_geometries: Iterable[shapely.geometry.Polygon],
    ) -> Tuple[np.ndarray, affine.Affine]:
        window = rasterio.features.geometry_window(tile, [window_geometry])
        shape = (window.height, window.width)
        transform = tile.window_transform(window)
        raster = rasterio.features.geometry_mask(
            raster_geometries,
            transform=transform,
            out_shape=shape,
            invert=True,
            all_touched=True,
        )
        raster = raster.astype(self._dtype)
        return raster, transform

    def _load_raster_data(
        self,
        area_key: AreaKey,
    ) -> RasterDataJobResult:
        area_id, tile_id = area_key
        # We may pass a query result
        # Then, area = self._areas.iloc[area_id] wouldn't work
        area = self._areas.query(f"id == {area_id}").iloc[0]
        polygon_results: List[PolygonRasterDataJobResult] = list()
        # See: https://git.io/JcD6q
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True):
            with rasterio.open(self.get_tile_path(tile_id), "r") as src:
                area_window = None
                try:
                    area_window = rasterio.features.geometry_window(
                        src,
                        [area.geometry],
                    )
                except Exception as e:
                    print(
                        "Error processing: tile_id={}, area_id={}".format(
                            tile_id,
                            area_id,
                        )
                    )
                    raise e
                area_transform = src.window_transform(area_window)
                area_shape = (area_window.height, area_window.width)
                area_crs = src.meta["crs"]
                for polygon_id, polygon in self.get_polygons(
                    tile_id, area_id
                ).iterrows():
                    bbox = shapely.geometry.box(*polygon.geometry.bounds)
                    try:
                        bbox_raster, bbox_transform = self._get_geometry_raster(
                            src,
                            bbox,
                            [polygon.geometry],
                        )
                    except Exception as e:
                        print(
                            "Error processing: tile_id={}, area_id={}, polygon_id={}".format(
                                tile_id,
                                area_id,
                                polygon_id,
                            )
                        )
                        raise e
                    raster_padded = None
                    transform_padded = None
                    if self._pad_scale is not None:
                        # We also want a scaled bounding box
                        # to create a padded raster.
                        bbox_padded = shapely.affinity.scale(
                            bbox,
                            xfact=self._pad_scale,
                            yfact=self._pad_scale,
                            origin="center",
                        )
                        raster_padded, transform_padded = self._get_geometry_raster(
                            src,
                            bbox_padded,
                            [polygon.geometry],
                        )

                    polygon_result = PolygonRasterDataJobResult(
                        polygon_id,
                        transform=bbox_transform,
                        raster=bbox_raster,
                        transform_padded=transform_padded,
                        raster_padded=raster_padded,
                    )
                    polygon_results.append(polygon_result)

        area_result = AreaRasterDataJobResult(
            area_id,
            tile_id,
            area_transform,
            area_shape,
            area_crs,
        )
        return RasterDataJobResult(
            area_result,
            polygon_results,
        )

    def load_raster_data(
        self,
        n_processes: Optional[int] = None,
        verbose: bool = True,
        job_slice: Optional[slice] = None,
    ) -> None:
        jobs = list()
        for area_id, area in self._areas.iterrows():
            for tile_id in area["tile_ids"]:
                jobs.append((area_id, tile_id))

        if job_slice is not None:
            jobs = jobs[job_slice]

        n_jobs = len(jobs)
        n_processes = get_n_processes(n_jobs, n_processes)

        if verbose:
            print(f"Will run {n_jobs} jobs with {n_processes} processes.")
        if n_processes > 1:
            with multiprocessing.Pool(n_processes) as pool:
                results = pool.map(self._load_raster_data, jobs)
        else:
            results = list(map(self._load_raster_data, jobs))

        self._area_raster_data.clear()
        self._polygon_raster_data.clear()
        for result in results:
            self._process_result(result)
        if verbose:
            print(f"Finished loading.")

    def suggested_filter_size(self, verbose: bool = True) -> int:
        """Suggest a size for the Gaussian filter used to blur point annotations."""
        xs = []
        ys = []
        for k, v in self._polygon_raster_data.items():
            ys.append(v.raster.shape[0])
            xs.append(v.raster.shape[1])
        xs = np.asarray(xs, dtype="float32")
        ys = np.asarray(ys, dtype="float32")

        xs_mean = np.mean(xs)
        ys_mean = np.mean(ys)
        filter_size = np.round(np.max([xs_mean, ys_mean]))
        if filter_size % 2 == 0:
            filter_size += 1

        if verbose:
            print("Dim\tMean\tQ1\tQ2\tQ3\tMin\tMax")
            print(
                "x",
                "\t",
                f"{xs_mean:.2f}",
                "\t",
                np.quantile(xs, 0.25),
                "\t",
                np.quantile(xs, 0.50),
                "\t",
                np.quantile(xs, 0.75),
                np.min(xs),
                np.max(xs),
            )
            print(
                "y",
                "\t",
                f"{ys_mean:.2f}",
                "\t",
                np.quantile(ys, 0.25),
                "\t",
                np.quantile(ys, 0.50),
                "\t",
                np.quantile(ys, 0.75),
                np.min(ys),
                np.max(ys),
            )
            print("Fiter size: ", filter_size)
        return filter_size
