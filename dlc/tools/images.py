# References
# [1]
#   Kariryaa, A., et al. (2021). PlanetUNet codebase.
from collections.abc import Iterable
from functools import partial
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import rasterio.plot
import tensorflow as tf

from .cache import BaseArrayCache
from .scalers import standardize_image_np
from .vendor import new_py_function

LocalStandardizationType = Union[Optional[float], List[Optional[float]]]


def decode_path(path: Union[str, bytes]) -> str:
    if isinstance(path, bytes):
        return path.decode()
    return path


def decode_paths(paths) -> List[List[str]]:
    decoded_paths = []
    if isinstance(paths, tf.RaggedTensor):
        row_splits = paths.row_splits.numpy()
        values = paths.values.numpy()
        for start, end in zip(row_splits, row_splits[1:]):
            decoded_paths.append(
                list(
                    map(
                        decode_path,
                        values[start:end].tolist(),
                    )
                )
            )
    elif isinstance(paths, tf.Tensor):
        decoded_paths.append(list(map(decode_path, paths.numpy())))
    else:
        decoded_paths.append(list(map(decode_path, paths)))
    return decoded_paths


def load_image(
    path: str,
    *,
    masked: bool = False,
    cache: Optional[BaseArrayCache] = None,
    dtype: Optional[str] = None,
) -> np.ndarray:
    if cache is not None:
        image = cache.read(path)
        if image is not None:
            return image
    # See: https://github.com/mapbox/rasterio/issues/2053#issuecomment-744579992
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True):
        with rasterio.open(path, "r") as src:
            image = src.read(masked=masked)
            if dtype is not None:
                image = image.astype(dtype)
            image = np.transpose(image, axes=[1, 2, 0])
            if cache is not None:
                cache.write(path, image)
            return image


PatchDefaultValSpec = List[Union[Any, Tuple[Any, ...]]]


def initialize_patch(
    shape: Tuple[int, int, int],
    dtype: str,
    defaults: PatchDefaultValSpec,
) -> np.ndarray:
    patch = np.zeros(shape, dtype=dtype)
    if defaults is not None:
        if isinstance(defaults, Iterable):
            if len(defaults) != shape[-1]:
                msg = "Shape mismatch: defaults ({}) and image ({})".format(
                    len(defaults),
                    shape[-1],
                )
                raise Exception(msg)
            for i, default in enumerate(defaults):
                patch[:, :, i] = default
        else:
            patch[:, :, 0] = defaults
    return patch


def create_patch(
    image: np.ndarray,
    *,
    y_range: Tuple[int, int],
    x_range: Tuple[int, int],
    size: Tuple[int, int],
    dtype: str = "float32",
    defaults: PatchDefaultValSpec = None,
) -> np.ndarray:
    y_slice = slice(*y_range)
    x_slice = slice(*x_range)
    height, width = size
    shape = (height, width, image.shape[-1])
    patch = initialize_patch(shape, dtype, defaults)
    slice_height = y_range[1] - y_range[0]
    slice_width = x_range[1] - x_range[0]
    patch[0:slice_height, 0:slice_width, :] = image[y_slice, x_slice, :]
    return patch


DefaultsSpec = Optional[Tuple[PatchDefaultValSpec, PatchDefaultValSpec]]


class ImageLoader:
    def __init__(
        self,
        *,
        local_standardization_p: LocalStandardizationType = None,
        seed: Optional[int] = None,
        masked: bool = False,
        cache: Optional[BaseArrayCache] = None,
        dtype: Optional[str] = "float32",
        defaults: DefaultsSpec = None,
    ):
        self._rng = np.random.default_rng(seed=seed)
        self._masked = masked
        self._cache = cache
        self._dtype = dtype
        self._defaults = defaults

        if not isinstance(local_standardization_p, Iterable):
            self._local_standardization_p = [local_standardization_p]
        else:
            self._local_standardization_p = local_standardization_p

    def load_np(self, spec):
        images_paths = decode_paths(spec["paths"])

        if len(images_paths) != len(self._local_standardization_p):
            msg = "Shape mismatch: images ({}) and local_standarization_p ({})"
            msg = msg.format(
                len(images_paths),
                len(self._local_standardization_p),
            )
            raise ValueError(msg)

        if self._defaults is not None and len(images_paths) != len(self._defaults):
            msg = "Shape mismatch: images ({}) and defaults ({})"
            msg = msg.format(
                len(images_paths),
                len(self._defaults),
            )
            raise ValueError(msg)

        create_patch_fn = partial(
            create_patch,
            y_range=spec["y_slice"],
            x_range=spec["x_slice"],
            size=spec["size"],
            dtype=self._dtype,
        )
        load_image_fn = partial(
            load_image,
            masked=self._masked,
            cache=self._cache,
            dtype=self._dtype,
        )

        patches: List[np.ndarray] = []
        for i, image_paths in enumerate(images_paths):
            stack: List[np.ndarray] = []
            standardize_probability = self._local_standardization_p[i]
            if self._defaults is not None:
                defaults = self._defaults[i]
            else:
                defaults = None
            standardize_type = None
            for j, path in enumerate(image_paths):
                default = None
                if defaults is not None:
                    default = defaults[j]
                image = load_image_fn(path)
                # Define if/how to standardize the patch
                if standardize_probability is not None:
                    standardize_type = "image"
                    if (
                        standardize_probability == 1.0
                        or self._rng.uniform(size=1) < standardize_probability
                    ):
                        standardize_type = "patch"
                # Create patch
                patch = None
                if standardize_type == "image":
                    image = standardize_image_np(image, axis=(0, 1))
                    patch = create_patch_fn(image, defaults=default)
                elif standardize_type == "patch":
                    patch = create_patch_fn(image, defaults=default)
                    patch = standardize_image_np(patch, axis=(0, 1))
                else:
                    patch = create_patch_fn(image, defaults=default)
                stack.append(patch)
            patches.append(np.dstack(stack))

        if len(patches) == 1:
            return patches[0]
        return patches

    def load(self, spec):
        # Wrap function to use with Tensorflow
        if isinstance(spec["paths"], tf.RaggedTensor):
            return new_py_function(
                self.load_np,
                [spec],
                [self._dtype, self._dtype],
            )
        return new_py_function(
            self.load_np,
            [spec],
            self._dtype,
        )

    def preload_cache(self, generator) -> None:
        if self._cache is None:
            raise ValueError("No cache")
        for path in generator():
            value = load_image(path, dtype=self._dtype)
            self._cache.write(str(path), value)
