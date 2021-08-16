import dataclasses
import multiprocessing
import pathlib
from functools import partial
from itertools import product
from pprint import pprint
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from geopandas.geodataframe import GeoDataFrame

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


import geopandas as gpd
import pandas as pd

from dlc.tools.common import load_geodataframe
from dlc.tools.common import get_n_processes
from dlc.tools.types import PathSource, GeoDataFrameOrigin
from dlc.tools.cache import CacheDictType
from dlc.tools.cache import ArrayCache
from dlc.tools.cache import SharedArrayCache
from dlc.frames.creators.data import CoreFrameDataSource
from dlc.frames.creators.base import FrameDataCreator
from dlc.frames.creators.base import RasterFrameDataCreator
from dlc.frames.creators.base import FrameDataCreatorResult
from dlc.frames.creators.image import ImageFrameCreator
from dlc.frames.creators.segmentation import SegmentationMaskFrameCreator
from dlc.frames.creators.segmentation import SegmentationBoundaryWeightsFrameCreator
from dlc.frames.creators.segmentation import OutlierWeightsMaskFrameCreator
from dlc.frames.creators.density import GaussianDensityFrameCreator
from dlc.frames.creators.density import THGaussianDensityFrameCreator
from dlc.frames.creators.density import DMGaussianDensityFrameCreator
from dlc.frames.creators.density import EDTDensityFrameCreator
from dlc.frames.creators.density import UniformDensityFrameCreator
from dlc.frames.creators.scalar import FramePropertiesDataCreator
from dlc.frames.creators.scalar import SaharaSahelFramePropertiesDataCreator


@dataclasses.dataclass
class FrameFactoryJob:
    area_id: int
    tile_id: int
    creator: FrameDataCreator


@dataclasses.dataclass
class FrameDataFactoryResult:
    """Define the results of a frame data factory run.

    Parameters
    ----------
    frames
        The newly created or updated frames database.
    results
        The raw results collection from the frame data creators.
    """

    frames: gpd.GeoDataFrame
    results: List[FrameDataCreatorResult]


class FrameDataStats:
    def __init__(self):
        self._stats = {}

    def check_key(self, ns: str, key: str, init: Any) -> None:
        if ns not in self._stats:
            self._stats[ns] = dict()
        if key not in self._stats[ns]:
            self._stats[ns][key] = init

    def update_min(self, ns: str, key: str, value: Optional[float]) -> None:
        if value is not None:
            self.check_key(ns, key, value)
            self._stats[ns][key] = min(self._stats[ns][key], value)

    def update_max(self, ns: str, key: str, value: Optional[float]) -> None:
        if value is not None:
            self.check_key(ns, key, value)
            self._stats[ns][key] = max(self._stats[ns][key], value)

    def print(self):
        pprint(self._stats)


class FrameDataFactory:
    """Create frame data using multiple frame data creators.

    Parameters
    ----------
    creators: optional
        A list of frame creators.

    Examples
    --------
    >>> factory = FrameDataFactory()
    >>> factory.add_creator(image_creator)
    >>> factory.add_creator(density_creator)
    >>> print(factory.keys)
    >>> results = factory.run_jobs(tiles, areas,
    ...                            output_path="./frames.geojson")
    Notes
    -----
    The output database format should be a GeoJSON file, since other formats
    limit the column name length to 10 characters.
    """

    def __init__(
        self,
        creators: Optional[List[FrameDataCreator]] = None,
    ):
        if creators is None:
            self._creators = []
        else:
            self._creators = creators

    @property
    def keys(self):
        return [creator.key for creator in self._creators]

    def add_creator(self, creator: FrameDataCreator) -> None:
        """Add a creator to the factory.

        Parameters
        ----------
        creator
            The frame data creator instance.
        """
        self._creators.append(creator)

    def _run_job(
        self,
        job: FrameFactoryJob,
        cache_dict: Optional[CacheDictType] = None,
        overwrite: bool = True,
    ) -> FrameDataCreatorResult:

        cache = None
        if cache_dict is not None:
            cache = ArrayCache(cache_dict)

        return job.creator.run(
            job.area_id,
            job.tile_id,
            verbose=False,
            overwrite=overwrite,
            cache=cache,
        )

    def _run_job_mp(
        self,
        job: FrameFactoryJob,
        cache_dict: Optional[CacheDictType] = None,
        overwrite: bool = True,
        shared: bool = False,
    ) -> FrameDataCreatorResult:
        cache = None
        if cache_dict is not None:
            if shared:
                cache = SharedArrayCache(cache_dict)
            else:
                pid = multiprocessing.current_process().pid
                if not pid in cache_dict:
                    cache_dict[pid] = dict()
                cache = ArrayCache(cache_dict[pid])

        return job.creator.run(
            job.area_id,
            job.tile_id,
            verbose=False,
            overwrite=overwrite,
            cache=cache,
        )

    def _run_jobs(
        self,
        jobs,
        *,
        verbose: bool,
        overwrite: bool,
        use_cache: bool,
    ) -> List[FrameDataCreatorResult]:
        cache_dict = None
        if use_cache:
            cache_dict = dict()
        results = list(
            map(
                partial(
                    self._run_job,
                    overwrite=overwrite,
                    cache_dict=cache_dict,
                ),
                jobs,
            )
        )
        if use_cache:
            cache = ArrayCache(cache_dict)
            print(
                "Cache statistics: hits = {}, misses = {}, size = {:.2f} Mb".format(
                    cache.hits,
                    cache.misses,
                    cache.size * 1e-6,
                )
            )
            cache.clear()
        return results

    def _run_jobs_mp(
        self,
        jobs,
        *,
        n_processes: int,
        verbose: bool,
        overwrite: bool,
        use_cache: bool,
        shared: bool = False,
    ) -> List[FrameDataCreatorResult]:
        with multiprocessing.Manager() as manager:
            cache_dict = None
            if use_cache:
                if shared:
                    print("Using shared cache.")
                    cache_dict = manager.dict(lock=True)
                else:
                    print("Using independent cache for each process.")
                    cache_dict = dict()
            with multiprocessing.Pool(processes=n_processes) as pool:
                results = pool.map(
                    partial(
                        self._run_job_mp,
                        overwrite=overwrite,
                        cache_dict=cache_dict,
                        shared=shared,
                    ),
                    jobs,
                )
            if use_cache:
                if shared:
                    caches = [SharedArrayCache(cache_dict)]
                else:
                    caches = [cache_dict[k] for k in cache_dict.keys()]
                for cache in caches:
                    if verbose:
                        print(
                            "Cache statistics: hits = {}, misses = {}, size = {:.2f} Mb".format(
                                cache.hits,
                                cache.misses,
                                cache.size * 1e-6,
                            )
                        )
                    cache.clear()
        return results

    def run_jobs(
        self,
        db: Union[
            Tuple[GeoDataFrameOrigin, GeoDataFrameOrigin],
            GeoDataFrameOrigin,
        ],
        *,
        output_path: Optional[PathSource] = None,
        n_processes: Optional[int] = None,
        job_slice: Optional[slice] = None,
        dry_run: bool = False,
        overwrite: bool = True,
        verbose: bool = True,
        skip_existing_keys: bool = False,
        save_keys: Optional[List[str]] = None,
        driver: str = "GeoJSON",
        cache_enabled: bool = False,
        cache_shared: bool = False,
        verify: bool = True,
    ) -> FrameDataFactoryResult:
        """Run the frame creation jobs.

        Parameters
        ----------
        db
            Either a pair of tiles and areas dataframes or a frames dataframe.
        output_path: optional
            A path to save the frames database.
        n_processes: optional
            The target number of CPUs to use. By default uses all available.
        job_slice: optional
            The slice of jobs to work on.
        dry_run
            Only display the work that would be done.
        overwrite
            Overwrite existing files.
        tile_filter_fn: optional
            A function that receives a tile and outputs True if it should \
                be processed.
        save_keys: optional
            A list of keys for which we wish to save results in the frames \
                database. By default all are saved.
        verify: optional
            Verify the created files.

        Returns
        -------
        FrameDataFactoryResult
            The results obtained after running all the jobs.
        """
        creators = self._creators

        if isinstance(db, tuple):
            tiles, areas = db
            tiles = load_geodataframe(tiles, copy=False)
            areas = load_geodataframe(areas, copy=False)
            frames = {
                "area_id": [],
                "tile_id": [],
                "geometry": [],
            }
            for tile_id, tile in tiles.iterrows():
                for area_id in tile["area_ids"]:
                    geometry = tile["geometry"].intersection(
                        areas.loc[area_id, "geometry"],
                    )
                    frames["area_id"].append(area_id)
                    frames["tile_id"].append(tile_id)
                    frames["geometry"].append(geometry)
            if verbose:
                print("Will create new frames database.")
            frames = gpd.GeoDataFrame(frames, crs=tiles.crs)
        else:
            if verbose:
                print("Will update existing frames database.")
            frames = load_geodataframe(db, copy=True)
            if skip_existing_keys:
                new_creators = []
                for creator in creators:
                    if creator.key not in frames:
                        new_creators.append(creator)
                    else:
                        print(f"Skipping {creator.key}.")
                n_diff = len(creators) - len(new_creators)
                print(f"Skipped {n_diff} creators with keys in frames db.")
                if n_diff > 0:
                    creators = new_creators

        jobs = []
        n_files = 0
        for (area_id, tile_id), creator in product(
            zip(frames["area_id"], frames["tile_id"]),
            creators,
        ):
            jobs.append(FrameFactoryJob(area_id, tile_id, creator))
            if isinstance(creator, RasterFrameDataCreator):
                n_files += 1

        n_total_jobs = len(jobs)

        if job_slice is None:
            job_slice = slice(0, len(jobs))
        jobs = jobs[job_slice]
        n_jobs = len(jobs)
        n_processes = get_n_processes(n_jobs, n_processes)

        if verbose:
            print(
                """Will run {}/{} jobs ({} frames x {} data creators) """
                """using {} processes and {}.""".format(
                    n_jobs,
                    n_total_jobs,
                    len(frames),
                    len(self._creators),
                    n_processes,
                    job_slice,
                )
            )
            print(f"Should produce {n_files} files.")

        results: List[FrameDataCreatorResult] = []

        if not dry_run:
            if n_processes > 1:
                results = self._run_jobs_mp(
                    jobs,
                    n_processes=n_processes,
                    verbose=verbose,
                    overwrite=overwrite,
                    use_cache=cache_enabled,
                    shared=cache_shared,
                )
            else:
                results = self._run_jobs(
                    jobs,
                    verbose=verbose,
                    overwrite=overwrite,
                    use_cache=cache_enabled,
                )
            # Temporarily set (tile_id, area_id) as the database index.
            frames.set_index(
                ["tile_id", "area_id"],
                drop=True,
                inplace=True,
            )

            # We keep track of which columns are already created.
            seen_keys = set()

            def check_key(result: FrameDataCreatorResult) -> None:
                """Check if a column exists in the dataframe for the \
                    key of the given result, and create it if needed."""
                # References:
                # https://stackoverflow.com/a/57293727
                if result.key not in seen_keys and result.key not in frames:
                    if isinstance(result.payload, dict):
                        # We flatten the dictionary into individual columns
                        # Does not support nested dictionaries with depth > 1.
                        for key, value in result.payload.items():
                            if (
                                result.payload_main_key is not None
                                and result.payload_main_key == key
                            ):
                                dtype = type(value)
                                frames[result.key] = pd.Series(dtype=dtype)
                            elif key in result.payload_no_prefix_keys:
                                dtype = type(value)
                                frames[key] = pd.Series(dtype=dtype)
                                seen_keys.add(key)
                            else:
                                compound_key = f"{result.key}_{key}"
                                if compound_key not in seen_keys:
                                    dtype = type(value)
                                    frames[compound_key] = pd.Series(dtype=dtype)
                                    seen_keys.add(compound_key)
                    else:
                        dtype = type(result.payload)
                        frames[result.key] = pd.Series(dtype=dtype)
                    seen_keys.add(result.key)

            # Now, we iterate over our results and save them
            # to the database (dataframe).

            for result in results:
                result_key = result.key
                if save_keys is None or result_key in save_keys:
                    check_key(result)
                    index = (result.tile_id, result.area_id)
                    if isinstance(result.payload, dict):
                        for key, value in result.payload.items():
                            if (
                                result.payload_main_key is not None
                                and result.payload_main_key == key
                            ):
                                frames.loc[index, result_key] = value
                            elif key in result.payload_no_prefix_keys:
                                frames.loc[index, key] = value
                            else:
                                compound_key = f"{result_key}_{key}"
                                frames.loc[index, compound_key] = value
                    else:
                        frames.loc[index, result_key] = result.payload

            # Reset the index (otherwise to_file below complains)
            frames.reset_index(inplace=True)
            if output_path is not None:
                output_path = pathlib.Path(output_path)
                frames.to_file(output_path, driver=driver)

            if verbose:
                print(f"Finished {len(results)} jobs.")

            if verbose:
                stats = self._compute_stats(results)
                print("Result statistics: ")
                stats.print()

            if verify:
                self.verify(frames)

        return FrameDataFactoryResult(frames, results)

    def verify(
        self,
        frames: GeoDataFrame,
    ) -> None:
        target = 0
        found = 0
        print("Verifying files...")
        for creator in self._creators:
            if isinstance(creator, RasterFrameDataCreator):
                if creator.key in frames:
                    values = frames[creator.key].values
                    target += len(values)
                    print(f"Checking {creator.output_base_path} for {creator.key}")
                    for value in values:
                        if isinstance(value, str):
                            path = value
                            if creator.only_filename:
                                path = creator.output_base_path.joinpath(value)
                            if not path.exists():
                                print(f"File does not exist: {path}")
                            else:
                                found += 1
                else:
                    print(f"Warning: {creator.key} not found within frames.")

        print(f"Found {found}/{target} files.")

    def _compute_stats(
        self,
        results: List[FrameDataCreatorResult],
    ):
        stats = FrameDataStats()
        for result in results:
            if result.meta is not None:
                if "min_pixel_value" in result.meta:
                    stats.update_min(
                        result.key,
                        "min_pixel_value",
                        result.meta["min_pixel_value"],
                    )
                if "max_pixel_value" in result.meta:
                    stats.update_max(
                        result.key,
                        "max_pixel_value",
                        result.meta["max_pixel_value"],
                    )
        return stats


class GaussianOptions(TypedDict):
    sigma: float
    filter_size: Optional[int]
    centroid_type: str
    filter_target: str


class THGaussianOptions(TypedDict):
    sigma: float
    thresh_z_score: Optional[float]
    centroid_type: str
    filter_target: str


class DMGaussianOptions(TypedDict):
    sigma: float
    filter_size: int


def create_and_configure_factory(
    data: CoreFrameDataSource,
    output_base_path: PathSource,
    *,
    creator_names: Optional[Tuple[str, ...]] = None,
    gaussian_options: List[GaussianOptions] = list(),
    th_gaussian_options: List[THGaussianOptions] = list(),
    dm_gaussian_options: List[DMGaussianOptions] = list(),
    predicted_data: Optional[CoreFrameDataSource] = None,
) -> FrameDataFactory:

    factory = FrameDataFactory()

    if creator_names is not None:
        creator_names = set(creator_names)

    if creator_names is None or "image" in creator_names:
        factory.add_creator(
            ImageFrameCreator(
                data,
                output_base_path,
                only_filename=True,
            )
        )
    if creator_names is None or "edt-density" in creator_names:
        factory.add_creator(
            EDTDensityFrameCreator(
                data,
                output_base_path,
                only_filename=True,
            )
        )
    if creator_names is None or "uniform-density" in creator_names:
        factory.add_creator(
            UniformDensityFrameCreator(
                data,
                output_base_path,
                only_filename=True,
            )
        )
    if creator_names is None or "gaussian-density" in creator_names:
        for options in gaussian_options:
            factory.add_creator(
                GaussianDensityFrameCreator(
                    data,
                    output_base_path,
                    sigma=options["sigma"],
                    filter_size=options["filter_size"],
                    centroid_type=options["centroid_type"],
                    filter_target=options["filter_target"],
                    only_filename=True,
                )
            )

    if creator_names is None or "th-gaussian-density" in creator_names:
        for options in th_gaussian_options:
            factory.add_creator(
                THGaussianDensityFrameCreator(
                    data,
                    output_base_path,
                    sigma=options["sigma"],
                    thresh_z_score=options["thresh_z_score"],
                    centroid_type=options["centroid_type"],
                    filter_target=options["filter_target"],
                    only_filename=True,
                )
            )

    if creator_names is None or "dm-gaussian-density" in creator_names:
        for options in dm_gaussian_options:
            factory.add_creator(
                DMGaussianDensityFrameCreator(
                    data,
                    output_base_path,
                    sigma=options["sigma"],
                    filter_size=options["filter_size"],
                    only_filename=True,
                )
            )

    if creator_names is None or "segmentation-mask" in creator_names:
        factory.add_creator(
            SegmentationMaskFrameCreator(
                data,
                output_base_path,
                only_filename=True,
            )
        )

    if creator_names is None or "segmentation-boundary-weights" in creator_names:
        factory.add_creator(
            SegmentationBoundaryWeightsFrameCreator(
                data,
                output_base_path,
                only_filename=True,
            )
        )

    if creator_names is None or "desired-weights" in creator_names:
        factory.add_creator(
            OutlierWeightsMaskFrameCreator(
                data,
                output_base_path,
                only_filename=True,
                query="is_desired == False",
                key="desired-weights",
            )
        )

    if creator_names is None or "outlier-weights" in creator_names:
        factory.add_creator(
            OutlierWeightsMaskFrameCreator(
                data,
                output_base_path,
                only_filename=True,
            )
        )

    if creator_names is None or "props" in creator_names:
        factory.add_creator(FramePropertiesDataCreator(data))

    if creator_names is None or "model" in creator_names:
        if data.name == "sahara-sahel":
            factory.add_creator(SaharaSahelFramePropertiesDataCreator(data))

    if predicted_data is not None:
        factory.add_creator(
            SegmentationMaskFrameCreator(
                predicted_data,
                output_base_path,
                only_filename=True,
                key="paper-predictions",
            )
        )

    # Verify that there are no duplicate keys.
    seen = set()
    for key in factory.keys:
        if key in seen:
            msg = f"Duplicate key: {key}"
            raise Exception(msg)
        else:
            seen.add(key)

    return factory
