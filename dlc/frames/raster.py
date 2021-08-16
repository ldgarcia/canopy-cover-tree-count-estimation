import dataclasses
import pathlib
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import affine
import numpy as np
import rasterio.crs

from dlc.tools.types import PathSource


def raster_op(
    src: np.ndarray,
    src_t: affine.Affine,
    dst: np.ndarray,
    dst_t: affine.Affine,
    op: Callable[[Any, Any], Any],
) -> None:
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            # Transform: src indices -> geospatial coordinates -> dst indices
            # Note: ~t denotes the inverse of t
            (b, a) = ((j, i) * src_t) * ~dst_t
            if 0 <= a < dst.shape[0] and 0 <= b < dst.shape[1]:
                a = min(round(a), dst.shape[0] - 1)
                b = min(round(b), dst.shape[1] - 1)
                dst[a, b] = op(dst[a, b], src[i, j])
    return dst


def raster_add(
    src: np.ndarray,
    src_t: affine.Affine,
    dst: np.ndarray,
    dst_t: affine.Affine,
) -> None:
    raster_op(src, src_t, dst, dst_t, lambda dst, src: dst + src)


def raster_max(
    src: np.ndarray,
    src_t: affine.Affine,
    dst: np.ndarray,
    dst_t: affine.Affine,
) -> None:
    raster_op(src, src_t, dst, dst_t, lambda dst, src: max(dst, src))


def raster_min(
    src: np.ndarray,
    src_t: affine.Affine,
    dst: np.ndarray,
    dst_t: affine.Affine,
) -> None:
    raster_op(src, src_t, dst, dst_t, lambda dst, src: min(dst, src))


def raster_write(
    output_path: PathSource,
    raster: np.ndarray,
    transform: affine.Affine,
    crs: rasterio.crs.CRS,
    *,
    nodata: Optional[Union[float, int]] = None,
) -> None:
    output_path = pathlib.Path(output_path)
    if raster.ndim != 3:
        raster = np.expand_dims(raster, axis=0)
    meta = dict(
        driver="GTiff",
        count=raster.shape[0],
        height=raster.shape[1],
        width=raster.shape[2],
        transform=transform,
        crs=crs,
        dtype=raster.dtype,
        nodata=nodata,
    )
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(raster)


# Experimental stuff


@dataclasses.dataclass
class Window:
    top: int = np.iinfo(np.int64).max
    bottom: int = np.iinfo(np.int64).min
    left: int = np.iinfo(np.int64).max
    right: int = np.iinfo(np.int64).min

    def update_height(self, value: int) -> None:
        if value < self.top:
            self.top = value
        if value > self.bottom:
            self.bottom = value

    def update_width(self, value: int) -> None:
        if value < self.left:
            self.left = value
        if value > self.right:
            self.right = value

    def view(self, raster: np.ndarray):
        return raster[self.top : self.bottom + 1, self.left : self.right + 1]


def raster_window(
    src: np.ndarray,
    src_t: affine.Affine,
    dst: np.ndarray,
    dst_t: affine.Affine,
) -> Tuple[Window, Window]:
    src_window = Window()
    dst_window = Window()
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            # Transform: src indices -> geospatial coordinates -> dst indices
            # Note: ~t denotes the inverse of t
            (b, a) = ((j, i) * src_t) * ~dst_t
            if 0 <= a < dst.shape[0] and 0 <= b < dst.shape[1]:
                a = min(round(a), dst.shape[0] - 1)
                b = min(round(b), dst.shape[1] - 1)
                src_window.update_height(i)
                src_window.update_width(j)
                dst_window.update_height(a)
                dst_window.update_width(b)

    return src_window, dst_window


def safe_view_add(src: np.ndarray, dst: np.ndarray) -> None:
    y_end = min(src.shape[0], dst.shape[0])
    x_end = min(src.shape[1], dst.shape[1])
    dst[0:y_end, 0:x_end] += src[0:y_end, 0:x_end]


def safe_view_copy(src: np.ndarray, dst: np.ndarray) -> None:
    y_end = min(src.shape[0], dst.shape[0])
    x_end = min(src.shape[1], dst.shape[1])
    dst[0:y_end, 0:x_end] = src[0:y_end, 0:x_end]
