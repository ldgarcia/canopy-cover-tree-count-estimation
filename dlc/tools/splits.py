"""This module contains code to create splits."""
import abc
import math
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import geopandas as gpd
import numpy as np
import shapely
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

from .common import load_geodataframe
from .types import GeoDataFrameOrigin


SplitResult1 = Tuple[int, ...]
SplitResult2 = Tuple[Tuple[int, ...], ...]


class ObjectSplitter(object, metaclass=abc.ABCMeta):
    """Define an object splitter interface."""

    def get_splits_map(self) -> Dict[str, int]:
        return {"training": 0, "test": 1, "reserved": 2}

    @abc.abstractmethod
    def train_val_split(
        self,
        objects: GeoDataFrameOrigin,
        *,
        val_size: float = 0.2,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[SplitResult1, Optional[Any]]:
        pass

    @abc.abstractmethod
    def stratified_k_fold(
        self,
        objects: GeoDataFrameOrigin,
        *,
        n_splits: int = 5,
        shuffle=True,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[SplitResult2, Optional[Any]]:
        """Create the object splits.

        Parameters
        ----------
        objects
            The objects database.
        n_splits
            The number of CV splits.
        shuffle
            Shuffle or not.
        seed: optional
            A seed for the random number generator.
        verbose
            Display diagnostic information.

        Returns
        -------
        ObjectSplitResult
            A list of split classes and optionally implementation-specific \
                extra information.
        """
        pass

    @abc.abstractmethod
    def train_val_split_k_fold(
        self,
        objects: GeoDataFrameOrigin,
        *,
        val_size: float = 0.2,
        n_splits: int = 3,
        shuffle=True,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        pass


class LatitudeObjectSplitter(ObjectSplitter):
    """Create latitude-balanced object splits, using SK-Learn's utilities.

    Examples
    --------
    >>> splitter = LatitudeObjectSplitter()
    >>> split, sampled_areas = splitter.train_val_split(frames)
    """

    def __init__(
        self,
        bins: Union[str, int] = "auto",
        poly_count_bins: Union[str, int] = "auto",
        centroid_projection: Optional[str] = "EPSG:6933",
        **kwargs,
    ) -> None:
        self.bins = bins  # controls the latitude histogram binning
        self.poly_count_bins = poly_count_bins
        self.centroid_projection = centroid_projection

    def _fix_histogram(self, counts, bins):
        """Fix a potentially conflicting histogram."""
        fixed_counts = [0]
        fixed_bins = [bins[0]]
        for i in range(len(counts)):
            count = counts[i]
            bin = bins[i]
            if count < 1:
                continue
            elif (count < 2) or (fixed_counts[-1] < 2):
                fixed_counts[-1] += count
            else:
                fixed_counts.append(count)
                fixed_bins.append(bin)
        fixed_bins.append(bins[-1])
        return fixed_counts, fixed_bins

    def _to_object_classes(
        self,
        objects: GeoDataFrameOrigin,
    ):
        # References:
        # https://gis.stackexchange.com/a/390563
        objects = load_geodataframe(objects, copy=False)

        # We compute a histogram of the centroid latitudes
        # to create "latitudinal bins" from which to sample tiles
        # uniformly at random. The target number of tiles to sample from
        # each area is proportional to the bin counts.
        if self.centroid_projection is not None:
            centroids = objects.to_crs(self.centroid_projection).centroid
            centroids = centroids.to_crs(objects.crs)
        else:
            centroids = objects.centroid
        # The latitude is the y-coordinate of the centroid
        counts, bins = np.histogram(centroids.geometry.y, bins=self.bins)
        counts, bins = self._fix_histogram(counts, bins)
        # Compute probability of area selection
        counts = np.array(counts, dtype=float)
        p = counts / np.sum(counts)
        # Compute area boxes
        minx = objects.bounds["minx"].min()
        maxx = objects.bounds["maxx"].max()
        areas = [
            shapely.geometry.box(minx, start, maxx, end)
            for start, end in zip(bins, bins[1:])
        ]
        sampled_areas = gpd.GeoDataFrame(
            {"geometry": areas, "count": p},
            crs=objects.crs,
        )

        # We use this array for computing the object class
        # the first component assigns the area id and the second
        # component tells if the object has polygons or not.
        object_class = np.zeros((len(objects), 2), dtype=int)

        # 'intersects' is initialized to be the neutral element
        intersects = np.zeros_like(centroids, dtype=bool)
        for area_id, area in enumerate(areas):
            intersects = np.logical_or(
                intersects,
                centroids.within(area).to_numpy(),
            )
            if np.sum(intersects.astype(int)) < 2:
                # the area has a single element, so we merge it with the next one
                continue
            intersects = centroids.within(area).to_numpy()
            poly_counts = objects["n_polygons"][intersects]
            counts, bins = np.histogram(poly_counts, bins=self.poly_count_bins)
            counts, bins = self._fix_histogram(counts, bins)
            object_class[intersects, 0] = area_id
            n = len(counts)
            for i in range(n):
                subclass = i + 1
                start = bins[i]
                end = bins[i + 1]
                pred_start = poly_counts >= start
                pred_end = poly_counts < end if i < n - 1 else poly_counts <= end
                idx_mask = np.logical_and(pred_start, pred_end)
                assert np.any(idx_mask)
                idx = np.flatnonzero(intersects)
                idx = idx[idx_mask]
                object_class[idx, 1] = subclass
            # Reset 'intersects' to be the neutral element.
            intersects = np.zeros_like(centroids, dtype=bool)
        # Now reduce the class tuples into a scalar
        object_class = object_class[:, 0] + 100 * object_class[:, 1]
        return object_class, sampled_areas

    def _to_split_list(self, train_idx, test_idx, n):
        splits = np.full(n, 2, dtype=int)
        splits[train_idx] = 0
        splits[test_idx] = 1
        return tuple(splits)

    def train_val_split(
        self,
        objects: GeoDataFrameOrigin,
        *,
        val_size: float = 0.2,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[SplitResult1, Optional[Any]]:
        object_class, sampled_areas = self._to_object_classes(objects)
        # Now we perform the train test split using SK-Learn's utility function.
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size, random_state=seed
        )
        # Create a dummy X vector
        dummy = np.ones_like(object_class, dtype=bool)
        train_idx, test_idx = next(splitter.split(dummy, object_class))
        n = len(train_idx) + len(test_idx)
        assert n == len(dummy)
        splits_list = self._to_split_list(train_idx, test_idx, len(dummy))
        if verbose:
            print("Created train-validation split:")
            print(f"Training frames: {len(train_idx)}")
            print(f"Validation frames: {len(test_idx)}")
        return splits_list, sampled_areas

    def stratified_k_fold(
        self,
        objects: GeoDataFrameOrigin,
        *,
        n_splits: int = 3,
        shuffle=True,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[SplitResult2, Optional[Any]]:

        object_class, sampled_areas = self._to_object_classes(objects)
        # Now we perform the stratified split using SK-Learn's utility function.
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=seed
        )
        # Create a dummy X vector
        dummy = np.ones_like(object_class, dtype=bool)
        n = len(dummy)
        splits_list = tuple(
            self._to_split_list(t, v, n)
            for t, v in splitter.split(
                dummy,
                object_class,
            )
        )
        assert len(splits_list) == n_splits
        return splits_list, sampled_areas

    def train_val_split_k_fold(
        self,
        objects: GeoDataFrameOrigin,
        *,
        val_size: float = 0.2,
        n_splits: int = 3,
        shuffle=True,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        object_class, sampled_areas = self._to_object_classes(objects)
        # Now we perform the train test split using SK-Learn's utility function.
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size, random_state=seed
        )
        # Create a dummy X vector
        dummy = np.ones_like(object_class, dtype=bool)
        train_idx, test_idx = next(splitter.split(dummy, object_class))
        if verbose:
            print(f"Initial split: {len(train_idx)} train and {len(test_idx)} test")
        # Second splitter
        splitter2 = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=seed
        )
        object_class2 = object_class[train_idx]
        dummy2 = np.ones_like(object_class2, dtype=bool)
        splits2 = splitter2.split(
            dummy2,
            object_class2,
        )
        splits2 = [(train_idx[t], train_idx[v]) for t, v in splits2]
        if verbose:
            for i, split in enumerate(splits2):
                t, v = split
                nt = len(t)
                nv = len(v)
                print(f"Split {i}: {nt} in train and {nv} in validation")
        n = len(dummy)
        splits_list = tuple(self._to_split_list(t, v, n) for t, v in splits2)
        assert len(splits_list) == n_splits
        return splits_list, sampled_areas
