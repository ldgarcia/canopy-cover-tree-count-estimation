# References:
# [1] Zunic, J., & Rosin, P. L. (2004). A new convexity measure for polygons.
#     IEEE Transactions on Pattern Analysis and Machine Intelligence, 26(7), 923-934.
"""This module contains functions related to manipulate tiles in a dataset."""
import dataclasses
import pathlib
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.coords
import rasterio.warp
import shapely.affinity
from tqdm import tqdm

from .common import load_geodataframe
from .types import GeoDataFrameOrigin, PathSource
from .splits import LatitudeObjectSplitter

__all__ = [
    "get_objects_from_images",
    "get_tiles_with_areas",
    "get_polygons_with_area",
    "get_areas_with_tiles",
]


def get_objects_from_images(
    input_base_path: pathlib.Path,
    search_pattern: str,
    *,
    output_path: Optional[PathSource] = None,
    crs: Optional[str] = None,
    different_crs: str = "skip",
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """Create a database of objects from a folder of images.

    Examples
    --------
    >>> tiles = get_objects_from_images("./tile_images", "*.tif",
    ...                                 output_path="./tiles.gpkg")
    >>> tiles.head()
    """

    # References:
    # https://shapely.readthedocs.io/en/stable/manual.html#shapely.geometry.box
    # https://geopandas.org/docs/reference/api/geopandas.GeoDataFrame.html#geopandas-geodataframe
    # https://rasterio.readthedocs.io/en/latest/quickstart.html#dataset-georeferencing
    # https://rasterio.readthedocs.io/en/latest/topics/reproject.html#reprojecting-a-geotiff-dataset
    # https://rasterio.readthedocs.io/en/latest/api/rasterio.warp.html#rasterio.warp.transform_bounds

    dataframe: dict = {
        "file_path": [],
        "file_size": [],
        "geometry": [],
        "reproject": [],
    }
    input_base_path = pathlib.Path(input_base_path)
    for file_path in tqdm(input_base_path.glob(search_pattern)):
        relative_file_path = file_path.relative_to(input_base_path)
        reproject = False
        try:
            dataset = rasterio.open(file_path)
        except Exception as e:
            print(f"Skipping: Could not open: {relative_file_path}")
            continue
        bounds = dataset.bounds
        if crs is None:
            # The first file defines the CRS to use if none is passed.
            crs = dataset.crs
            if verbose:
                print(f"Setting CRS = {crs}")
        elif crs != dataset.crs:
            # All entries must have the same CRS in a geodataframe.
            if different_crs == "error":
                msg = f"{relative_file_path} with CRS {dataset.crs} should've been CRS {crs}."
                raise ValueError(msg)
            elif different_crs == "reproject":
                if verbose:
                    print(f"Reprojecting {relative_file_path} into CRS {crs}")
                reproject = True
                bounds = rasterio.coords.BoundingBox(
                    *rasterio.warp.transform_bounds(
                        dataset.crs,
                        crs,
                        bounds.left,
                        bounds.bottom,
                        bounds.right,
                        bounds.top,
                    )
                )
            else:
                if verbose:
                    print(
                        "Skipping {} with different CRS: {}".format(
                            relative_file_path,
                            dataset.crs,
                        )
                    )
                continue

        dataframe["file_path"].append(str(relative_file_path))
        # Store file size in GBs
        file_size = file_path.stat().st_size * 1e-9
        dataframe["file_size"].append(file_size)
        dataframe["geometry"].append(
            shapely.geometry.box(
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
            )
        )
        # A flag to indicate if we should reproject the file.
        dataframe["reproject"].append(reproject)

    geodataframe = gpd.GeoDataFrame(dataframe, crs=crs)

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        geodataframe.to_file(output_path)

    if verbose:
        print(f"Finished! Kept {len(geodataframe)} objects.")

    return geodataframe


def get_tiles_with_areas(
    tiles: GeoDataFrameOrigin,
    areas: GeoDataFrameOrigin,
    *,
    verbose: bool = True,
):
    """Add area infomation to tiles."""
    # Adapted from:
    # https://gitlab.com/rscph/planetunet/-/blob/master/preprocessing.py
    tiles = load_geodataframe(tiles, copy=True)
    areas = load_geodataframe(areas, copy=False)

    if tiles.crs != areas.crs:
        raise ValueError("Tiles and areas have different CRS.")

    tiles["id"] = np.arange(len(tiles), dtype=int)
    join = gpd.sjoin(
        tiles,
        gpd.GeoDataFrame(geometry=areas.geometry, crs=areas.crs),
        op="intersects",
        how="inner",
    )
    area_ids = [[] for _ in range(len(tiles))]
    for _, tile in join.iterrows():
        area_ids[tile["id"]].append(int(tile["index_right"]))
    tiles["area_ids"] = area_ids
    tiles["n_areas"] = list(map(lambda x: len(x), area_ids))

    if verbose:
        print(
            "Found {}/{} tiles with at least one area.".format(
                len(tiles[tiles["n_areas"] > 0]),
                len(tiles),
            )
        )
    return tiles


def add_equiarea_props(data: gpd.GeoDataFrame) -> None:
    data_equiarea = data.to_crs("EPSG:6933")
    # We use the key parea so as not to conflict with the area property.
    data["size"] = data_equiarea.area
    # A simple area-based convexity measure mentioned in [1],
    # defined as the ratio between the shape area and the convex hull area.
    convex_hull_area = data_equiarea.convex_hull.area
    data["convexity_measure"] = data["size"] / convex_hull_area


def get_polygons_with_tile_and_area(
    polygons: GeoDataFrameOrigin,
    areas: GeoDataFrameOrigin,
    tiles: GeoDataFrameOrigin,
):
    """Get polygons with the corresponding area index and size in \
        square meters."""
    # Adapted from:
    # https://gitlab.com/rscph/planetunet/-/blob/master/preprocessing.py
    areas = load_geodataframe(areas, copy=False)
    tiles = load_geodataframe(tiles, copy=False)
    polygons = load_geodataframe(polygons, copy=True)
    drop: List[str] = []
    if "area_id" in polygons:
        drop.append("area_id")
    if "tile_id" in polygons:
        drop.append("tile_id")
    if "is_orphan" in polygons:
        drop.append("is_orphan")

    polygons.drop(columns=drop, inplace=True)

    if areas.crs != polygons.crs:
        raise ValueError("Areas and polygons have different CRS.")

    add_equiarea_props(polygons)

    polygons["id"] = np.arange(len(polygons), dtype=int)
    area_id = [None for _ in polygons["id"]]
    tile_id = [None for _ in polygons["id"]]
    is_orphan = [True for _ in polygons["id"]]

    join = gpd.sjoin(
        polygons,
        gpd.GeoDataFrame(geometry=areas.geometry, crs=areas.crs),
        op="intersects",
        how="inner",
    )
    # Assumes areas have an id column
    join = join.rename(
        columns={"index_right": "area_id"},
    )
    join = gpd.sjoin(
        join,
        gpd.GeoDataFrame(geometry=tiles.geometry, crs=tiles.crs),
        op="intersects",
        how="inner",
    )
    # Assumes tiles have an id column
    join = join.rename(
        columns={"index_right": "tile_id"},
    )
    for _, row in join.iterrows():
        area_id[row["id"]] = row["area_id"]
        tile_id[row["id"]] = row["tile_id"]
        is_orphan[row["id"]] = False

    polygons["area_id"] = area_id
    polygons["tile_id"] = tile_id
    polygons["is_orphan"] = is_orphan
    polygons["id"] = polygons.index

    return polygons


def add_n_polygons(frames: GeoDataFrameOrigin, polygons: GeoDataFrameOrigin):
    frames = load_geodataframe(frames, copy=True)
    polygons = load_geodataframe(polygons, copy=False)
    n_polygons: List[int] = []
    for _, row in frames.iterrows():
        n_polygons.append(
            len(
                polygons.query(
                    f"area_id == {row['area_id']} and tile_id == {row['tile_id']}"
                )
            )
        )
    frames["n_polygons"] = n_polygons
    return frames


def fix_overlapped_polygons(
    polygons: GeoDataFrameOrigin,
    *,
    area_id: Optional[int] = None,
    initial_scale: Optional[float] = 0.90,
    verbose: bool = True,
    multipolygon_strategy: str = "merge",
    verify: bool = True,
    max_retries: int = 3,
):
    """Fix overlaps between polygons.

    Parameters
    ----------
    polygons
        The polygons database.
    area_id: optional
        An area id to filter rows.
    max_passes: optional
        Maximum number of tries until overlapped pairs do not change.
    scale: optional
        A factor to scale the fixed polygons at the end of each pass.
    verbose
        Print diagnostic information.
    multipolygon_strategy
        How to handle heavily overlapped polygons.
    verify
        Verify no polygons overlap at the end.

    Returns
    -------
    A new database of non-overlapped polygons.

    Examples
    --------
    >>> x = fix_overlapped_polygons(polygons, area_id=350, scale=0.95)
    >>> if len(x) > 0:
    >>>   ax = x.boundary.plot(edgecolor="red")
    >>>   x_fixed = x.query("fixed == True")
    >>>   if len(x_fixed) > 0:
    >>>     x_fixed.plot(color="yellow", ax=ax, alpha=0.3)
    """

    polygons = load_geodataframe(polygons, copy=True)
    polygons["geometry"] = polygons.geometry.scale(
        xfact=initial_scale,
        yfact=initial_scale,
        origin="center",
    )
    if area_id is not None:
        if verbose:
            print(f"Filtering by area_id={area_id}.")
        polygons.query(f"area_id == {area_id}", inplace=True)
    n_start = len(polygons)
    cp = 0
    polygons["fixed"] = 0
    n_retries = 0
    last_n_overlapped = None
    while True:
        if max_retries is not None and n_retries == max_retries:
            break
        cp += 1
        overlapped_pairs = gpd.sjoin(
            polygons,
            polygons,
            how="inner",
            op="overlaps",
        )
        overlapped_pairs.query("id_left != id_right", inplace=True)
        n_overlapped = len(overlapped_pairs)
        if n_overlapped == 0:
            break
        if last_n_overlapped == n_overlapped:
            n_retries += 1
            print(f"Retrying fix {n_retries}/{max_retries} attempts.")
        else:
            n_retries = 0
            last_n_overlapped = n_overlapped
        if verbose:
            print(f"Executing pass {cp} (found {n_overlapped // 2} pairs).")
        polygons["delete"] = False
        for _, pair in overlapped_pairs.iterrows():
            try:
                idx_left = pair["id_left"]
                idx_right = pair["id_right"]
                left_polygon = polygons.loc[idx_left, "geometry"]
                right_polygon = polygons.loc[idx_right, "geometry"]
                if isinstance(left_polygon, pd.Series):
                    left_polygon = left_polygon.iloc[0]
                    if verbose:
                        print(f"Warning: Got series at idx_left = {idx_left}")
                if isinstance(right_polygon, pd.Series):
                    right_polygon = right_polygon.iloc[0]
                    print(f"Warning: Got series at idx_right = {idx_right}")
                new_polygon = left_polygon.difference(right_polygon)
                if new_polygon.geom_type == "MultiPolygon":
                    # The polygons in the pair are heavily overlapped.
                    # For an example, use the Sahara-Sahel dataset
                    # >>> ax = polygons.query("area_id == 82").boundary.plot(edgecolor="red")
                    # >>> ax.set_ylim(18 * 1e-5 + 1.467e1, 21 * 1e-5 + 1.467e1)
                    # >>> ax.set_xlim(-2.6 * 1e-5 - 1.14042e1, -0.4 * 1e-5 - 1.14042e1)
                    if verbose:
                        print(
                            """Warning: Found heavily overlapped polygons """
                            """in area_id={}, id_left={}, id_right={}""".format(
                                pair["area_id_left"],
                                idx_left,
                                idx_right,
                            )
                        )
                    if multipolygon_strategy == "merge":
                        new_polygon = left_polygon.union(right_polygon)
                        polygons.loc[idx_left, "geometry"] = new_polygon
                        polygons.loc[idx_right, "geometry"] = new_polygon
                        polygons.loc[idx_right, "delete"] = True
                    elif multipolygon_strategy == "delete_right":
                        new_polygon = polygons.loc[idx_left, "geometry"]
                        polygons.loc[idx_right, "geometry"] = new_polygon
                        polygons.loc[idx_right, "delete"] = True
                    else:
                        raise ValueError("Invalid value for multipolygon_strategy")
                else:
                    polygons.loc[idx_left, "geometry"] = new_polygon
                polygons.loc[idx_left, "fixed"] += 1
            except Exception as e:
                print(
                    "Error processing area_id={}, id_left={}, id_right={}".format(
                        pair["area_id_left"], pair["id_left"], pair["id_right"]
                    )
                )
                raise e
        polygons.query("delete == False", inplace=True)
    n_end = len(polygons)
    if verbose:
        print("Fixed {} polygons.".format(len(polygons.query("fixed == True"))))
        print(
            """Polygons at start where {} and at finish {}. """
            """Deleted {} after using '{}' strategy.""".format(
                n_start,
                n_end,
                n_start - n_end,
                multipolygon_strategy,
            )
        )
    polygons["overlapped"] = False
    if verify:
        print("Verifying results...")
        overlapped_pairs = gpd.sjoin(polygons, polygons, how="inner", op="overlaps")
        overlapped_pairs.query("id_left != id_right", inplace=True)
        for _, pair in overlapped_pairs.iterrows():
            polygons.loc[pair["id_left"], "overlapped"] = True
            polygons.loc[pair["id_left"], "fixed"] = 0
        n_overlapped = len(overlapped_pairs)
        if n_overlapped > 0:
            area_ids = set(overlapped_pairs["area_id_left"])
            print(
                f"""Warning: Found {n_overlapped // 2} overlapped pairs at finish.
                Areas: {area_ids}"""
            )
        if not polygons.is_valid.all():
            print("Warning: found non valid geometries.")
        if polygons.is_empty.any():
            print("Warning: found empty geometries.")
    add_equiarea_props(polygons)
    return polygons


def get_areas_with_tiles(
    areas: GeoDataFrameOrigin,
    tiles: GeoDataFrameOrigin,
    *,
    filter_no_tiles: bool = False,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    areas = load_geodataframe(areas, copy=True)
    tiles = load_geodataframe(tiles, copy=False)

    if tiles.crs != areas.crs:
        raise ValueError("Areas and tiles have different CRS.")

    areas["id"] = np.arange(len(areas), dtype=int)
    join = gpd.sjoin(
        areas,
        gpd.GeoDataFrame(geometry=tiles.geometry, crs=tiles.crs),
        op="intersects",
        how="inner",
    )
    tile_ids: List[List[int]] = [[] for _ in range(len(areas))]
    for _, area in join.iterrows():
        tile_ids[area["id"]].append(int(area["index_right"]))
    areas["tile_ids"] = tile_ids
    areas["n_tiles"] = list(map(lambda x: len(x), tile_ids))

    if verbose:
        print(
            "Found {}/{} areas with at least one tile.".format(
                len(areas.query("n_tiles > 0", inplace=False)),
                len(areas),
            )
        )
    if filter_no_tiles:
        if verbose:
            print("Filtered out areas with no tiles.")
            areas.query("n_tiles > 0", inplace=False)
    areas["id"] = areas.index
    return areas


def get_nonoutlier_range(
    objects: GeoDataFrameOrigin,
    key: str,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
) -> Tuple[float, float]:
    # See: https://en.wikipedia.org/wiki/Interquartile_range
    objects = load_geodataframe(objects, copy=False)
    q3, q1 = objects[key].quantile([0.75, 0.25])
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    if range_min is not None:
        lower_limit = max(range_min, lower_limit)
    if range_max is not None:
        upper_limit = min(range_max, upper_limit)
    return lower_limit, upper_limit


class OutliersKeySpec(TypedDict):
    name: str
    range_min: Optional[float]
    range_max: Optional[float]


def mark_query(
    objects: GeoDataFrameOrigin,
    key: str,
    query: str,
) -> gpd.GeoDataFrame:
    objects = load_geodataframe(objects, copy=True)
    objects[key] = False
    marked_objects = objects.query(query)
    objects.loc[marked_objects.index, key] = True
    return objects


def mark_outliers(
    objects: GeoDataFrameOrigin,
    keys: Tuple[OutliersKeySpec, ...],
    *,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    objects = load_geodataframe(objects, copy=True)
    # Mark outliers by each criteria
    nonoutliers_ranges = dict()
    nonoutliers_marks = dict(is_outlier=list())
    for key in keys:
        key_name = key["name"]
        key_col = f"is_{key_name}_outlier"
        nonoutliers_ranges[key_col] = get_nonoutlier_range(
            objects,
            key=key_name,
            range_min=key.get("range_min", None),
            range_max=key.get("range_max", None),
        )
        nonoutliers_marks[key_col] = list()
        if verbose:
            print(
                "Non-outlier range for {} is [{:.6f},{:.6f}]".format(
                    key_name,
                    nonoutliers_ranges[key_col][0],
                    nonoutliers_ranges[key_col][1],
                )
            )
    # Determine which rows are outliers
    for _, row in objects.iterrows():
        is_outlier = False
        for key in keys:
            key_name = key["name"]
            key_col = f"is_{key_name}_outlier"
            range_min, range_max = nonoutliers_ranges[key_col]
            is_outlier_key = row[key_name] < range_min or row[key_name] > range_max
            nonoutliers_marks[key_col].append(is_outlier_key)
            is_outlier = is_outlier_key or is_outlier
        nonoutliers_marks["is_outlier"].append(is_outlier)
    # Add new columns to existing dataframe
    objects["is_outlier"] = nonoutliers_marks["is_outlier"]
    for key in keys:
        key_name = key["name"]
        key_col = f"is_{key_name}_outlier"
        objects[key_col] = nonoutliers_marks[key_col]
    return objects


def add_n_polygons(
    areas: GeoDataFrameOrigin,
    polygons: GeoDataFrameOrigin,
) -> gpd.GeoDataFrame:
    areas = load_geodataframe(areas, copy=True)
    polygons = load_geodataframe(polygons, copy=False)
    n_polygons: List[int] = []
    for _, area in areas.iterrows():
        n_polygons.append(len(polygons.query(f"area_id == {area['id']}")))
    areas["n_polygons"] = n_polygons
    return areas


def mark_region(
    objects: GeoDataFrameOrigin,
    *,
    copy: bool = False,
) -> gpd.GeoDataFrame:
    # Definition of the Sahara model region from the [2]:
    # 24N to about 17N latitude
    objects = load_geodataframe(objects, copy=copy)
    sahara = gpd.GeoSeries(
        [shapely.geometry.box(-20.0, 17.0, 40.0, 24.0)], crs=objects.crs
    )
    region = []
    for _, row in objects.iterrows():
        if sahara.intersects(row.geometry)[0]:
            region.append("sahara")
        else:
            region.append("sahel")
    objects["region"] = region
    return objects


def mark_split(
    objects: GeoDataFrameOrigin, name: str, *, copy: bool = False
) -> gpd.GeoDataFrame:
    objects = load_geodataframe(objects, copy=copy)
    objects["split"] = name
    return objects


def consolidate_sahara_sahel_areas(
    train: GeoDataFrameOrigin,
    test: GeoDataFrameOrigin,
    *,
    tiles: GeoDataFrameOrigin,
) -> gpd.GeoDataFrame:
    """Consolidate the independent train and test files into a single file."""
    tiles = load_geodataframe(tiles, copy=False)

    train = load_geodataframe(train)
    train = mark_split(train, "train")
    if "old_id" not in train:
        train["old_id"] = train["id"]

    test = load_geodataframe(test)
    test = mark_split(test, "test")
    if "old_id" not in test:
        test["old_id"] = test["id"]

    areas = train.append(
        test,
        ignore_index=True,
        sort=False,
    )
    areas.drop(
        columns=[
            "sensor",
            "month",
            "off_nadir",
            "sun_elev",
            "sun_az",
            "sat_elev",
            "sat_az",
            "date_diff",
            "score",
        ],
        inplace=True,
    )
    areas = mark_region(areas)
    areas = get_areas_with_tiles(
        areas,
        tiles=tiles,
    )
    areas["id"] = np.arange(len(areas), dtype=int)

    return areas


def consolidate_sahara_sahel_polygons(
    train: GeoDataFrameOrigin,
    test: GeoDataFrameOrigin,
    *,
    tiles: GeoDataFrameOrigin,
    areas: GeoDataFrameOrigin,
) -> gpd.GeoDataFrame:
    """Consolidate the independent train and test files into a single file."""
    tiles = load_geodataframe(tiles, copy=False)
    # Assumes areas are already consolidated
    areas = load_geodataframe(areas, copy=False)

    train = load_geodataframe(train)
    train = mark_split(train, "train")
    if "old_id" not in train:
        train["old_id"] = train["id"]

    test = load_geodataframe(test)
    test = mark_split(test, "test")
    if "old_id" not in test:
        test["old_id"] = test["id"]

    polygons = train.append(
        test,
        ignore_index=True,
        sort=False,
    )
    # polygons["id"] = np.arange(len(polygons), dtype=int)
    polygons = get_polygons_with_tile_and_area(
        polygons,
        areas=areas,
        tiles=tiles,
    )
    polygons = mark_region(polygons)

    return polygons


@dataclasses.dataclass
class Database:
    tiles: gpd.GeoDataFrame
    areas: gpd.GeoDataFrame
    polygons: gpd.GeoDataFrame
    predicted_polygons: Optional[gpd.GeoDataFrame] = None

    def __post_init__(self):
        self._rng = np.random.default_rng()

    def to_files(self, output_path: PathSource) -> None:
        output_path = pathlib.Path(output_path)
        print(f"Saving database to {output_path}")
        self.tiles.drop(columns=["area_ids"]).to_file(
            output_path.joinpath("tiles.geojson"),
            driver="GeoJSON",
        )
        self.areas.drop(columns=["tile_ids"]).to_file(
            output_path.joinpath("areas.geojson"),
            driver="GeoJSON",
        )
        self.polygons.to_file(
            output_path.joinpath("polygons.geojson"),
            driver="GeoJSON",
        )
        if self.predicted_polygons is not None:
            self.predicted_polygons.to_file(
                output_path.joinpath("predicted-polygons.geojson"),
                driver="GeoJSON",
            )

    def get_area_and_tile(self, area_id: Optional[int] = None):
        if area_id is None:
            while True:
                area_id = self._rng.integers(0, len(self.areas))
                if self.areas.loc[area_id, "n_tiles"] >= 1:
                    break
        tile_id = self.areas.loc[area_id, "tile_ids"][0]

        return area_id, tile_id


OUTLIER_SPEC: Tuple[OutliersKeySpec, ...] = (
    dict(name="size", range_min=None, range_max=None),
    dict(name="convexity_measure", range_min=None, range_max=None),
)

IS_DESIRED_QUERY = "convexity_measure >= 0.80 and size <= 200"


def load_database(
    *,
    tiles: GeoDataFrameOrigin,
    areas: GeoDataFrameOrigin,
    polygons: GeoDataFrameOrigin,
    predicted_polygons: Optional[GeoDataFrameOrigin] = None,
    outliers: bool = False,
) -> Database:
    areas = get_areas_with_tiles(areas, tiles=tiles)
    tiles = get_tiles_with_areas(tiles, areas=areas)
    polygons = get_polygons_with_tile_and_area(
        polygons,
        areas=areas,
        tiles=tiles,
    )
    polygons = mark_query(
        polygons,
        "is_desired",
        IS_DESIRED_QUERY,
    )
    if outliers:
        polygons = mark_outliers(polygons, OUTLIER_SPEC)
    areas = add_n_polygons(areas, polygons)
    database = Database(tiles=tiles, areas=areas, polygons=polygons)
    if predicted_polygons is not None:
        predicted_polygons = load_geodataframe(predicted_polygons)
        database.predicted_polygons = predicted_polygons
    return database


def load_database_sahara_sahel(
    *,
    tiles: GeoDataFrameOrigin,
    train_areas: GeoDataFrameOrigin,
    test_areas: GeoDataFrameOrigin,
    train_polygons: GeoDataFrameOrigin,
    test_polygons: GeoDataFrameOrigin,
    test_predicted_polygons: Optional[GeoDataFrameOrigin] = None,
    outliers: bool = False,
):
    # test_predicted_polygons contains the predicted polygons from [2].
    areas = consolidate_sahara_sahel_areas(
        train_areas,
        test_areas,
        tiles=tiles,
    )
    tiles = get_tiles_with_areas(tiles, areas=areas)
    tiles = mark_region(tiles)
    polygons = consolidate_sahara_sahel_polygons(
        train_polygons,
        test_polygons,
        tiles=tiles,
        areas=areas,
    )

    polygons = mark_query(
        polygons,
        "is_desired",
        IS_DESIRED_QUERY,
    )

    if outliers:
        sahara_polygons = polygons.query("region == 'sahara'")
        sahara_polygons = mark_outliers(sahara_polygons, OUTLIER_SPEC)
        sahel_polygons = polygons.query("region == 'sahel'")
        sahel_polygons = mark_outliers(sahel_polygons, OUTLIER_SPEC)

        for key in (
            "is_outlier",
            "is_convexity_measure_outlier",
            "is_size_outlier",
        ):
            polygons[key] = False
            polygons.loc[sahara_polygons.index, key] = sahara_polygons[key]
            polygons.loc[sahel_polygons.index, key] = sahel_polygons[key]

    areas = add_n_polygons(areas, polygons)

    database = Database(tiles=tiles, areas=areas, polygons=polygons)
    if test_predicted_polygons is not None:
        test_predicted_polygons = get_polygons_with_tile_and_area(
            test_predicted_polygons,
            areas=areas,
            tiles=tiles,
        )
        database.predicted_polygons = test_predicted_polygons

    return database


def load_database_rwanda(
    *,
    tiles: GeoDataFrameOrigin,
    areas: GeoDataFrameOrigin,
    polygons: GeoDataFrameOrigin,
    seed: Optional[int] = None,
    outliers: bool = False,
    test_size: float = 0.15,
    bins: Union[str, int] = 5,
    poly_count_bins: Union[str, int] = 4,
) -> Database:
    database = load_database(
        tiles=tiles,
        areas=areas,
        polygons=polygons,
        outliers=outliers,
    )
    splitter = LatitudeObjectSplitter(
        bins=bins,
        poly_count_bins=poly_count_bins,
    )
    splits, _ = splitter.train_val_split(
        database.areas,
        val_size=test_size,
        seed=seed,
    )
    splits_map = {v: k for k, v in splitter.get_splits_map().items()}
    database.areas["split"] = [splits_map[x] for x in splits]
    splits = []
    for _, polygon in database.polygons.iterrows():
        if not np.isnan(polygon["area_id"]):
            splits.append(database.areas.loc[polygon["area_id"], "split"])
        else:
            splits.append(None)
    database.polygons["split"] = splits
    return database
