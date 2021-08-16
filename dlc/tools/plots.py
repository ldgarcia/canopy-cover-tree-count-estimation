"""Contains plotting functions."""
import copy
import pathlib
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pydantic
import rasterio.plot
from descartes import PolygonPatch
from matplotlib.collections import PatchCollection

from .common import load_geodataframe
from .images import load_image
from .scalers import normalize_image_np
from .scalers import standardize_image_np
from .types import GeoDataFrameOrigin
from .types import PathSource


def plot_object_splits(
    objects: Union[pathlib.Path, str, gpd.GeoDataFrame],
    splits: List[float],
    *,
    areas: Optional[gpd.GeoDataFrame] = None,
    colors: List[str] = ["white", "red"],
    window: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
):
    """Plot objects splits and optionally the sampled areas.

    Examples
    --------
    >>> plot_object_splits(tile_splits, areas=sampled_areas)
    """
    objects = load_geodataframe(objects, copy=True)
    objects["split"] = splits
    # References:
    # https://stackoverflow.com/a/38885389
    split_classes = set(splits)
    if len(colors) != len(split_classes):
        raise ValueError("Incorrect number of colors")
    cmap = mpl.colors.ListedColormap(colors, N=len(colors))
    ax = objects.plot(
        column="split",
        cmap=cmap,
        vmin=min(split_classes),
        vmax=max(split_classes),
    )
    ax = objects.boundary.plot(ax=ax, color="black")
    if window is not None:
        ax.set_ylim(*window[0])
        ax.set_xlim(*window[1])
    if areas is not None:
        ax = areas.plot(
            column="count",
            ax=ax,
            alpha=0.4,
            legend=True,
            cmap="jet",
        )
    return ax


def plot_polygons_in_area(
    areas: Union[pathlib.Path, str, gpd.GeoDataFrame],
    polygons: Union[pathlib.Path, str, gpd.GeoDataFrame],
    area_id: Optional[int] = None,
    *,
    polygon_id: Optional[int] = None,
    seed: Optional[int] = None,
    plot_type: str = "area",
    figsize: Optional[Tuple[int]] = None,
    ax=None,
    verbose: bool = True,
    plot_centroids: bool = False,
    polygon_col: Optional[str] = None,
):
    """Plot an area to inspect it."""
    # References:
    # https://gitlab.com/rscph/planetunet/-/blob/master/preprocessing.py
    rng = np.random.default_rng(seed)
    areas = load_geodataframe(areas, copy=False)
    polygons = load_geodataframe(polygons, copy=False)

    if areas.crs != polygons.crs:
        raise ValueError("Areas and polygons have different CRS.")

    # If an area index is not provided, get one at random.
    if area_id is None:
        area_id = rng.integers(0, len(areas), size=1)[0]

    polygons_in_area = polygons[polygons["area_id"] == area_id]

    if plot_type == "area":
        ax = areas.iloc[area_id : area_id + 1].plot(
            color="brown",
            alpha=0.2,
            figsize=figsize,
            ax=ax,
        )
        if len(polygons_in_area) > 0:
            if polygon_col is None:
                ax = polygons_in_area.plot(
                    ax=ax,
                    color="green",
                    figsize=figsize,
                )
            else:
                ax = polygons_in_area.plot(
                    ax=ax,
                    column=polygon_col,
                    figsize=figsize,
                )
    else:
        ax = areas.boundary.iloc[area_id : area_id + 1].plot(
            color="brown",
            figsize=figsize,
            ax=ax,
        )
        if len(polygons_in_area) > 0:
            ax = polygons_in_area.boundary.plot(
                ax=ax,
                color="green",
                figsize=figsize,
            )
    # Optionally, highlight a polygon.
    if polygon_id is not None:
        ax = polygons_in_area[polygon_id : polygon_id + 1].plot(
            ax=ax,
            color="orange",
            figsize=figsize,
        )
    if plot_centroids:
        ax = polygons_in_area["centroid"].plot(
            ax=ax,
            color="yellow",
            figsize=figsize,
            marker="x",
            markersize=2.0,
        )
    if verbose:
        print(
            "Area {} has {} polygons.".format(
                area_id,
                len(polygons_in_area),
            )
        )
    return ax


def get_polygon_patches(polygons, *, edgecolor="red"):
    # See: https://gis.stackexchange.com/a/193695
    polygon_patches = []
    for _, polygon in polygons.iterrows():
        patch = PolygonPatch(
            polygon["geometry"],
            edgecolor=edgecolor,
            facecolor="none",
            linewidth=1,
        )
        polygon_patches.append(patch)
    return polygon_patches


def plot_frame(
    image: np.ndarray,
    *,
    title: Optional[str] = None,
    cmap: Optional[str] = None,
    norm=None,
    mask_color: Optional[str] = None,
    log: bool = False,
    bins: Union[int, str] = 20,
    show_hist: bool = True,
    output_path: Optional[PathSource] = None,
    polygons: Optional[Any] = None,
    polygon_color: str = "red",
    transform: Optional[Any] = None,
    nodata: Optional[Any] = None,
    figsize: Optional[Tuple[int, int]] = None,
    zoom_to: Optional[Tuple[float, float, float, float]] = None,
    show_title: bool = False,
    show_band_names: bool = True,
    show_colorbar: bool = False,
    show_axis: bool = True,
):
    """Plot bands and a histogram of a given raster array."""
    n_bands = image.shape[2]
    if cmap is None:
        cmap = mpl.cm.get_cmap("gray")
    else:
        cmap = mpl.cm.get_cmap(cmap)

    if mask_color is not None:
        cmap.set_bad(color=mask_color, alpha=1.0)

    n_cols = n_bands
    if show_hist:
        n_cols += 1

    if figsize is None:
        fig = plt.figure(figsize=(5.3 * n_cols, 5.0))
    else:
        fig = plt.figure(figsize=figsize)
    if show_colorbar:
        gs = fig.add_gridspec(2, n_cols, height_ratios=[1, 0.05])
        if not show_hist:
            cax = fig.add_subplot(gs[1, :])
        else:
            cax = fig.add_subplot(gs[1, : n_cols - 1])
    else:
        gs = fig.add_gridspec(1, n_cols)
        cax = None

    if show_title and title is not None:
        fig.suptitle(title)

    if polygons is not None:
        polygon_patches = get_polygon_patches(
            polygons,
            edgecolor=polygon_color,
        )
    else:
        polygon_patches = None

    extent = None
    if transform is not None:
        extent = rasterio.plot.plotting_extent(image, transform=transform)

    image_to_plot = image
    if nodata is not None:
        image_to_plot = np.ma.masked_array(image, mask=(image == nodata))

    im = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    for i, band in enumerate(range(1, n_bands + 1)):
        ax = fig.add_subplot(gs[0, i])
        if show_band_names:
            ax.set_title(f"Band {band}")
        ax.imshow(
            image_to_plot[:, :, i],
            cmap=cmap,
            norm=norm,
            origin="upper",
            extent=extent,
        )
        if not show_axis:
            ax.axis("off")
        if polygon_patches is not None:
            polygon_collection = PatchCollection(
                polygon_patches,
                match_original=True,
            )
            ax.add_collection(polygon_collection)
        if zoom_to is not None:
            ax.set_ylim(zoom_to[2:])
            ax.set_xlim(zoom_to[:2])
    if show_colorbar:
        fig.colorbar(im, cax=cax, orientation="horizontal")

    if show_hist:
        ax = fig.add_subplot(gs[0, n_bands])
        raster_shape = (image.shape[2], image.shape[0], image.shape[1])
        rasterio.plot.show_hist(
            image_to_plot.reshape(raster_shape),
            bins=bins,
            stacked=False,
            alpha=0.3,
            log=log,
            histtype="stepfilled",
            title="Histogram",
            ax=ax,
        )

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        plt.savefig(output_path)

    return fig


def plot_frame_nodata(
    path: PathSource,
    *,
    title: Optional[str] = None,
    output_path: Optional[PathSource] = None,
):
    path = pathlib.Path(path)
    with rasterio.open(path, "r") as src:
        n_bands = src.count
        fig = plt.figure(figsize=(5.3 * n_bands, 4.5))
        gs = fig.add_gridspec(1, n_bands)
        if title is not None:
            fig.suptitle(title)
        cmap = mpl.colors.ListedColormap(["white", "red"], N=2)
        for band, nodataval in zip(range(1, n_bands + 1), src.nodatavals):
            img = src.read(band, masked=True)
            ax = fig.add_subplot(gs[0, band - 1])
            ax.set_title(f"Band {band} (nodata = {nodataval})")
            if nodataval is not None:
                nodata_img = np.where(img.mask, 1.0, 0.0)
                ax.imshow(nodata_img, cmap=cmap)
            else:
                ax.imshow(np.zeros_like(img), cmap=cmap)
    if output_path is not None:
        output_path = pathlib.Path(output_path)
        plt.savefig(output_path)
    return fig


def plot_frame_by_key(
    frames: GeoDataFrameOrigin,
    key: str,
    base_path: PathSource,
    *,
    index: Optional[int] = None,
    area_id: Optional[int] = None,
    tile_id: Optional[int] = None,
    **kwargs,
):
    rng = np.random.default_rng()
    base_path = pathlib.Path(base_path)
    frames = load_geodataframe(frames)

    if index is not None:
        filename = frames.loc[index, key]
        area_id = frames.loc[index, "area_id"]
        tile_id = frames.loc[index, "tile_id"]
    elif area_id is not None and tile_id is not None:
        filename = frames.query(
            f"area_id == {area_id} and tile_id == {tile_id}".format(
                area_id,
                tile_id,
            ),
            inplace=False,
        )[key].item()
    else:
        index = rng.integers(0, len(frames), size=1)[0]
        filename = frames.loc[index, key]
        area_id = frames.loc[index, "area_id"]
        tile_id = frames.loc[index, "tile_id"]

    image_path = base_path.joinpath(filename)
    with rasterio.open(image_path, "r") as src:
        transform = src.transform
        kwargs["transform"] = transform

    image = load_image(str(image_path), masked=kwargs.get("masked", False))

    if kwargs.get("standardize", False):
        image = standardize_image_np(image, axis=(0, 1))

    if kwargs.get("normalize", False):
        image = normalize_image_np(image, axis=(0, 1))

    polygons = kwargs.get("polygons", None)
    if polygons is not None:
        polygons_in_area = polygons.query(
            f"area_id == {area_id}",
            inplace=False,
        )
        title = "Area {}, Tile {}, {} polygons".format(
            area_id,
            tile_id,
            len(polygons_in_area),
        )
    else:
        polygons_in_area = None
        title = f"Area {area_id}, Tile {tile_id}"

    return plot_frame(image, title=title, **kwargs)


ArrayLike = Union[np.ndarray, List]


BinSpec = Union[ArrayLike, int]


BinsSpec = Union[int, Tuple[BinSpec, BinSpec]]


def plot_1d_histogram(
    objects: GeoDataFrameOrigin,
    *,
    key: str,
    label: str,
    figsize: Tuple[float, float] = (4, 3.5),
    color: str = "purple",
    title: Optional[str] = None,
    bins: BinsSpec = 20,
    log: bool = False,
    fontsize: float = 12.0,
):
    objects = load_geodataframe(objects, copy=False)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.hist(objects[key], bins=bins, log=log, color=color)
    ax.set_ylabel("Frequency", fontsize=fontsize * 0.8)
    ax.set_xlabel(label, fontsize=fontsize)
    ax.grid()
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    return fig


def plot_2d_histogram(
    objects: GeoDataFrameOrigin,
    *,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    figsize: Tuple[float, float] = (8, 3.5),
    cmap: str = "jet_r",
    norm: Optional[Any] = mpl.colors.LogNorm(),
    title: Optional[str] = None,
    bins: BinsSpec = 20,
    fontsize: float = 12.0,
    vlines: Tuple[float, ...] = (),
    hlines: Tuple[float, ...] = (),
):
    # Reference: https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
    objects = load_geodataframe(objects, copy=False)
    H, xedges, yedges = np.histogram2d(
        objects[x_key],
        objects[y_key],
        bins=bins,
    )
    H = H.T
    X, Y = np.meshgrid(xedges, yedges)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    cmap = mpl.cm.get_cmap(cmap)
    im = ax.pcolormesh(X, Y, H, cmap=cmap, norm=norm)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    if len(vlines) > 0 and len(hlines) > 0:
        ax.vlines(
            vlines,
            yedges[0],
            max(hlines),
            color="white",
            linestyle="dashed",
        )
        ax.hlines(
            hlines,
            max(vlines),
            xedges[-1],
            color="white",
            linestyle="dashed",
        )

    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    cbar = fig.colorbar(im)
    # cbar.set_label("Frequency", rotation=270, fontsize=fontsize)
    return fig, dict(xedges=xedges, yedges=yedges)


def plot_box(
    objects: GeoDataFrameOrigin,
    *,
    key: str,
    label: str,
    figsize=(4, 2.5),
    fontsize: float = 12.0,
):
    # See: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html
    objects = load_geodataframe(objects, copy=False)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.boxplot(objects[key], vert=False, sym="")
    ax.set_xlabel(label, fontsize=fontsize)
    ax.get_yaxis().set_visible(False)
    return fig


class ColorMapSpec(pydantic.BaseModel):
    keys: Tuple[Union[str, Tuple[str, int], Tuple[str, str, int]], ...]
    cmap: str = "gray"
    mask_value: Optional[float] = None
    mask_color: str = "white"
    mask_rel: str = "less_equal"


class ColorMapFactory:
    def __init__(self):
        self._cmaps = dict()
        self._groups = dict()

    def add_group(self, group: ColorMapSpec):
        for key in group.keys:
            if group.cmap in self._cmaps:
                cmap = copy.copy(self._cmaps[group.cmap])
            else:
                cmap = copy.copy(mpl.cm.get_cmap(group.cmap))
                if group.mask_value is not None:
                    cmap.set_bad(
                        color=group.mask_color,
                        alpha=1.0,
                    ),
            self._groups[key] = (cmap, group.mask_value, group.mask_rel)

    def add_groups(self, groups: Iterable[ColorMapSpec]):
        for group in groups:
            self.add_group(group)

    def add_listed_cmap(self, key: str, colors: List[str]):
        self._cmaps[key] = mpl.colors.ListedColormap(colors, N=len(colors))

    def get_cmap_data(self, data, *keys):
        cmap, mask_value, mask_rel = (mpl.cm.get_cmap("gray"), None, None)
        for key in keys:
            if key in self._groups:
                cmap, mask_value, mask_rel = self._groups[key]
                break
        if mask_value is not None:
            if mask_rel == "less_equal":
                data = np.ma.masked_less_equal(data, mask_value)
            elif mask_rel == "equal":
                data = np.ma.masked_equal(data, mask_value)
            elif mask_rel == "greater_equal":
                data = np.ma.masked_greater_equal(data, mask_value)
            else:
                raise ValueError("Invalid value for mask_rel")
        return cmap, data


def plot_batch(
    xs,
    ys,
    yhs=None,
    title: Optional[str] = None,
    cmf: Optional[ColorMapFactory] = None,
    show_hist: bool = True,
):
    if cmf is None:
        cmf = ColorMapFactory()

    n_cols = xs.shape[0]
    n_rows = xs.shape[-1]
    if not show_hist:
        n_rows += ys.shape[-1]
    else:
        n_rows += 2 * ys.shape[-1]
    if yhs is not None:
        if not show_hist:
            n_rows += yhs.shape[-1]
        else:
            n_rows += 2 * yhs.shape[-1]
    fig, axs = plt.subplots(
        figsize=(n_cols * 2.5, n_rows * 2.75),
        nrows=n_rows,
        ncols=n_cols,
        sharex=not show_hist,
        sharey=not show_hist,
    )
    for i in range(xs.shape[0]):
        row = 0
        # Features
        for k in range(xs.shape[-1]):
            cmap, data = cmf.get_cmap_data(
                xs[i, :, :, k],
                ("features", k),
                "features",
            )
            axs[row, i].imshow(data, cmap)
            if i == 0:
                axs[row, i].set_title(f"Features (band {k + 1})")
            row += 1
        # Annotations
        for k in range(ys.shape[-1]):
            cmap, data = cmf.get_cmap_data(
                ys[i, :, :, k],
                ("annotations", k),
                "annotations",
            )
            axs[row, i].imshow(data, cmap=cmap)
            if i == 0:
                axs[row, i].set_title(f"Annotations (band {k + 1})")
            row += 1
            if show_hist:
                axs[row, i].hist(
                    ys[i, :, :, k].numpy().flatten(),
                    bins=10,
                    stacked=False,
                    log=True,
                    color="purple",
                )
                row += 1
        # Predictions
        if yhs is not None:
            for k in range(yhs.shape[-1]):
                cmap, data = cmf.get_cmap_data(
                    yhs[i, :, :, k],
                    ("predictions", k),
                    "predictions",
                )
                axs[row, i].imshow(data, cmap=cmap)
            if i == 0:
                axs[row, i].set_title(f"Predictions (band {k + 1})")
            row += 1
            if show_hist:
                axs[row, i].hist(
                    yhs[i, :, :, k].numpy().flatten(),
                    bins=10,
                    stacked=False,
                    log=True,
                    color="purple",
                )
                row += 1
        if title is not None:
            fig.suptitle(title)
    return fig


def plot_multioutput_batch(
    xs,
    ys,
    yhs=None,
    yhs_names: Optional[Iterable[str]] = None,
    keys: Tuple[str, ...] = (),
    title: Optional[str] = None,
    cmf: Optional[ColorMapFactory] = None,
    show_hist: bool = True,
):
    if cmf is None:
        cmf = ColorMapFactory()

    n_cols = xs.shape[0]
    n_rows = xs.shape[-1]
    for key in keys:
        if not show_hist:
            n_rows += ys[key].shape[-1]
        else:
            n_rows += 2 * ys[key].shape[-1]
    if yhs is not None and yhs_names is not None:
        for name, value in zip(yhs_names, yhs):
            if name in keys:
                if not show_hist:
                    n_rows += value.shape[-1]
                else:
                    n_rows += 2 * value.shape[-1]

    fig, axs = plt.subplots(
        figsize=(n_cols * 2.5, n_rows * 2.75),
        nrows=n_rows,
        ncols=n_cols,
        sharex=not show_hist,
        sharey=not show_hist,
    )
    for i in range(xs.shape[0]):
        row = 0
        # Features
        for k in range(xs.shape[-1]):
            cmap, data = cmf.get_cmap_data(
                xs[i, :, :, k],
                ("features", k),
                "features",
            )
            axs[row, i].imshow(data, cmap)
            if i == 0:
                axs[row, i].set_title(f"Features (band {k + 1})")
            row += 1
        # Annotations
        for key, value in ys.items():
            if key in keys:
                for k in range(value.shape[-1]):
                    cmap, data = cmf.get_cmap_data(
                        value[i, :, :, k],
                        (key, k),
                        key,
                        ("annotations", k),
                        "annotations",
                    )
                    axs[row, i].imshow(data, cmap=cmap)
                    if i == 0:
                        axs[row, i].set_title(f"Annotations (band {k + 1})")
                    row += 1
                    if show_hist:
                        axs[row, i].hist(
                            value[i, :, :, k].numpy().flatten(),
                            bins=10,
                            stacked=False,
                            log=True,
                            color="purple",
                        )
                        row += 1
        # Predictions
        if yhs is not None and yhs_names is not None:
            for name, value in zip(yhs_names, yhs):
                if name in keys:
                    for k in range(value.shape[-1]):
                        cmap, data = cmf.get_cmap_data(
                            value[i, :, :, k],
                            (name, k),
                            name,
                            ("predictions", k),
                            "predictions",
                        )
                        axs[row, i].imshow(data, cmap=cmap)
                    if i == 0:
                        axs[row, i].set_title(f"Predictions (band {k + 1})")
                    row += 1
                    if show_hist:
                        axs[row, i].hist(
                            value[i, :, :, k].flatten(),
                            bins=10,
                            stacked=False,
                            log=True,
                            color="purple",
                        )
                        row += 1
        if title is not None:
            fig.suptitle(title)
    return fig
