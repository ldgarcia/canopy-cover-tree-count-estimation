import importlib
import pathlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tabulate
import yaml

from dlc.models.base.settings import Settings
from dlc.models.base.settings import SplitterSpec
from dlc.tools.datasets import ImageDatasetGenerator
from dlc.tools.images import ImageLoader

PathLike = Union[pathlib.Path, str]


def make_table(
    models_path: PathLike,
    settings_path: PathLike,
    *,
    is_test: bool = False,
    monitor: str = "val_loss",
    monitor_best: str = "min",
    monitor_label: str = "Loss",
    floatfmt: str = ".4f",
    tablefmt: str = "simple",
    showindex: str = "always",
    hidecolumns: List[str] = ["Label", "Loss", "Folder"],
    group_by: str = "Dataset",
    verbose: bool = True,
    format_rows: str = "best",
    none_replacement: str = "Pending",
    best_marker_fmt: str = "* {:.4f}",
):
    models_path = pathlib.Path(models_path)
    settings_path = pathlib.Path(settings_path)
    with open(settings_path, "r") as settings_fp:
        settings = yaml.load(settings_fp, Loader=yaml.Loader)
    table: Dict[str, Any] = dict()
    table["Label"] = []
    table["Dataset"] = []
    for key, label in settings.get("static_columns", dict()).items():
        if settings["static_columns_pos"].get(key, 0) == 0:
            table[label] = []
    table["Model"] = []
    for key, label in settings.get("static_columns", dict()).items():
        if settings["static_columns_pos"].get(key, 0) == 1:
            table[label] = []
    for column in settings["columns"]:
        table[column["label"]] = []
    table["Best Epoch"] = []
    table[monitor_label] = []
    for key, label in settings.get("static_columns", dict()).items():
        if settings["static_columns_pos"].get(key, 0) == 2:
            table[label] = []
    table["Folder"] = []

    # Find the value for each column according to the best monitor
    monitor_op = np.nanargmin if monitor_best == "min" else np.nanargmax
    available = 0
    for i, model in enumerate(settings["models"]):
        if not is_test:
            data_path = models_path.joinpath(model["folder"]).joinpath(
                "training.csv",
            )
            try:
                data = pd.read_csv(data_path.as_posix())
                available += 1
            except FileNotFoundError:
                data = None
        else:
            data_path = models_path.joinpath(model["folder"]).joinpath(
                "test_eval.yml",
            )
            try:
                with open(data_path.as_posix(), "r") as fp:
                    data = yaml.load(fp, Loader=yaml.Loader)
            except FileNotFoundError:
                data = None
        table["Label"].append(i)
        table["Dataset"].append(settings["datasets"][model["dataset"]])
        table["Model"].append(model["label"])
        for key, label in settings.get("static_columns", dict()).items():
            if key in model["static_columns"]:
                table[label].append(model["static_columns"][key])
            else:
                table[label].append(None)
        if data is not None:
            if not is_test:
                best_epoch = monitor_op(data[monitor])
                table[monitor_label].append(data[monitor][best_epoch])
                table["Best Epoch"].append(data["epoch"][best_epoch])
                for column in settings["columns"]:
                    if column["key"] in data:
                        table[column["label"]].append(data[column["key"]][best_epoch])
                    else:
                        table[column["label"]].append(None)
            else:
                for column in settings["columns"]:
                    key: str = column["key"]
                    if key.startswith("val_"):
                        key = key[4:]
                    if key in data:
                        table[column["label"]].append(data[key])
                    else:
                        table[column["label"]].append(None)
        else:
            if not is_test:
                table[monitor_label].append("Pending")
                table["Best Epoch"].append("Pending")
            for column in settings["columns"]:
                table[column["label"]].append(None)
        table["Folder"].append(model["folder"])
    if is_test:
        print(table.keys())
        table.pop("Best Epoch", None)
        table.pop("Loss", None)
    table_df = pd.DataFrame(table)
    if is_test:
        table_df["Best Epoch"] = None
        table_df["Loss"] = None

    if verbose:
        completeness = available / len(settings["models"])
        print(f"Data completeness: {completeness:.2%}")

    if format_rows == "best":
        # Find the best row per dataset-column pair
        best = dict()
        group_by_labels = table_df[group_by].unique()
        for label in group_by_labels:
            rows = table_df.query(f"{group_by} == '{label}'")
            for column in settings["columns"]:
                if column["best"] == "max":
                    best_index = rows[column["label"]].argmax()
                else:
                    best_index = rows[column["label"]].argmin()
                best[(label, column["label"])] = rows.iloc[best_index]["Label"]

        # Transform each column to show the best with an asterisk
        def formatrow(row):
            for column in settings["columns"]:
                best_index = best[(row[group_by], column["label"])]
                if not np.isnan(row[column["label"]]):
                    if best_index == row["Label"]:
                        row[column["label"]] = best_marker_fmt.format(
                            row[column["label"]]
                        )
                    else:
                        row[column["label"]] = f"{row[column['label']]:.4f}"
                else:
                    row[column["label"]] = none_replacement
            return row

    elif format_rows == "replace_none":

        def formatrow(row):
            for column in settings["columns"]:
                if row[column["label"]] is None or np.isnan(row[column["label"]]):
                    row[column["label"]] = none_replacement
            return row

    else:
        raise ValueError("Invalid value for format_rows")

    table_df = table_df.apply(formatrow, axis=1)

    return (
        tabulate.tabulate(
            table_df.drop(columns=hidecolumns),
            headers="keys",
            floatfmt=floatfmt,
            tablefmt=tablefmt,
            showindex=showindex,
        ),
        table_df,
    )


def make_plot(
    models_path: PathLike,
    settings_path: PathLike,
    *,
    key: str = "loss",
    key_best: str = "min",
    key_label: str = "Loss",
    fontsize: int = 12,
    figsize: Tuple[float, float] = (12.0, 6.5),
    label_type: str = "number",
    label_cols: Optional[Tuple[str, ...]] = None,
    legend_pos: int = 0,
    show_legend: bool = True,
    legend_cols: Optional[int] = None,
    yscale: str = "linear",
    dataset: Optional[str] = None,
    start_at: int = 0,
    ignore: Tuple[int, ...] = (),
    include: Tuple[int, ...] = (),
    monitor_key: str = "val_loss",
    monitor_best: str = "min",
    sharex: bool = True,
    sharey: bool = True,
    hlines: Tuple[float, ...] = (),
    combined: bool = False,
    legend_loc: Optional[str] = None,
    bbox_to_anchor: Optional[Any] = None,
):
    models_path = pathlib.Path(models_path)
    settings_path = pathlib.Path(settings_path)
    with open(settings_path, "r") as settings_fp:
        settings = yaml.load(settings_fp, Loader=yaml.Loader)
    # Load data for each model
    val_key = f"val_{key}"
    plot_data: Dict[Any, Any] = dict()
    stop_at = np.inf
    monitor_list: List[Tuple[Union[str, int], float]] = list()
    for i, model in enumerate(settings["models"]):
        if len(ignore) > 0 and i in ignore:
            continue
        elif len(include) > 0 and i not in include:
            continue
        elif dataset is not None and settings["datasets"][model["dataset"]] != dataset:
            continue

        data_path = models_path.joinpath(model["folder"]).joinpath("training.csv")
        try:
            model_data = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"Warning: '{data_path}' was not found.")
            continue
        if label_type == "number":
            label = i
        else:
            if label_cols is not None:
                label_suffix = ", ".join(
                    [model["static_columns"][x] for x in label_cols]
                )
                label = f"{i} [{label_suffix}]"
            else:
                label = f"{i} [{model['label']}]"
        plot_data[label] = dict()
        plot_data[label]["train"] = model_data[key]
        plot_data[label]["val"] = model_data[val_key]
        plot_data[label]["skip"] = np.all(np.isnan(model_data[key]))
        stop_at = min(stop_at, len(model_data["epoch"]))
        if monitor_best == "min":
            best_index = model_data[monitor_key].argmin()
        else:
            best_index = model_data[monitor_key].argmax()
        monitor_list.append((label, model_data[val_key][best_index]))

    stop_at = int(stop_at)
    # Make plot
    if not combined:
        fig, axs = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
        )
        axs[0].set_ylabel(f"Validation\n{key_label}", fontsize=fontsize)
        axs[1].set_ylabel(f"Training\n{key_label}", fontsize=fontsize)
        axs[0].set_yscale(yscale)
        axs[1].set_yscale(yscale)
    else:
        fig, axs = plt.subplots(
            1,
            1,
            figsize=figsize,
        )
        axs = [axs]
        axs[0].set_ylabel(key_label, fontsize=fontsize)
        axs[0].set_yscale(yscale)
    axs[0].set_xlabel("Epoch")
    axs[0].set_title(
        f"History (epochs {start_at + 1} to {stop_at})",
        fontsize=fontsize,
    )
    # See: https://stackoverflow.com/a/49091378
    norm = mpl.colors.Normalize(vmin=0, vmax=len(monitor_list))

    if key_best == "min":

        def key_sort(x):
            return x[1]

    else:

        def key_sort(x):
            return -x[1]

    epochs = np.arange(start_at + 1, stop_at + 1, dtype="int")
    for j, (label, _) in enumerate(sorted(monitor_list, key=key_sort)):
        data = plot_data[label]
        if not data["skip"]:
            line_color = mpl.cm.jet(norm(j))
            if not combined:
                axs[0].plot(
                    epochs,
                    data["val"][start_at:stop_at],
                    label=label,
                    color=line_color,
                    zorder=len(monitor_list) - j,
                )
                axs[1].plot(
                    epochs,
                    data["train"][start_at:stop_at],
                    label=label,
                    color=line_color,
                    zorder=len(monitor_list) - j,
                )
            else:
                axs[0].plot(
                    epochs,
                    data["val"][start_at:stop_at],
                    label=f"Val {label}",
                    color=line_color,
                    zorder=len(monitor_list) - j,
                )
                axs[0].plot(
                    epochs,
                    data["train"][start_at:stop_at],
                    label=f"Train {label}",
                    color=line_color,
                    zorder=len(monitor_list) - j,
                    linestyle="dashed",
                )
    if len(hlines) > 0:
        for ax in axs:
            ax.hlines(
                hlines,
                start_at + 1,
                stop_at + 1,
                color="magenta",
                linestyle="dashed",
            )
    if show_legend:
        loc = legend_loc
        handles, labels = axs[0].get_legend_handles_labels()
        if legend_pos == 0:
            box = axs[0].get_position()
            axs[0].set_position(
                [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.80]
            )
            if not combined:
                box = axs[1].get_position()
                axs[1].set_position(
                    [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.80]
                )
            ncol = 10 if label_type == "number" else 2
            if bbox_to_anchor is None:
                bbox_to_anchor = (0.5, -0.10)
            if loc is None:
                loc = "upper center"
        else:
            box = axs[0].get_position()
            axs[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            if not combined:
                box = axs[1].get_position()
                axs[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ncol = 3 if label_type == "number" else 1
            if bbox_to_anchor is None:
                bbox_to_anchor = (1, -0.1)
            if loc is None:
                loc = "center left"
        axs[0].legend(
            handles,
            labels,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            fancybox=True,
            shadow=True,
            ncol=ncol if legend_cols is None else legend_cols,
            fontsize=fontsize,
        )
    return fig


def load_model(path):
    with open(path.joinpath("settings.yml"), "r") as fp:
        settings_str = fp.read()
        settings_str = settings_str.replace(
            "scale_fn: !!python/name:tensorflow_addons.optimizers.cyclical_learning_rate.%3Clambda%3E ''",
            "scale_fn: null",
        )
        settings_dict = yaml.load(settings_str)
    module = importlib.import_module(f"dlc.models.{settings_dict['model']}")
    settings_t = getattr(module, "Settings", Settings)
    load_model = getattr(module, "load_model")
    settings = settings_t.construct(**settings_dict)
    model = load_model(
        input_shape=settings.input_shape,
        batch_size=settings.training_batch_size,
        weights_file=path.joinpath("weights.h5"),
        **dict(settings),
    )
    return model, settings


def load_models(base_path, folder_names):
    models = []
    settings = []
    for folder_name in folder_names:
        model, settings_ = load_model(base_path.joinpath(folder_name))
        models.append(model)
        settings.append(settings_)
    return models, settings


def get_images_ds(frames_path, settings, cache=None):
    frames = gpd.read_file(frames_path.joinpath("frames.geojson"))
    frames.query(settings.evaluation_frames_query, inplace=True)

    images_ds_gen = ImageDatasetGenerator(
        frames,
        splits=None,
        splits_map=None,
        image_keys=(settings.feature_keys, settings.annotation_keys),
        input_base_path=frames_path,
    )
    p = settings.testing_local_standardization_p
    image_loader = ImageLoader(
        local_standardization_p=p,
        cache=cache,
        seed=settings.seed,
        defaults=settings.loader_patch_defaults,
    )
    _, images_ds = images_ds_gen.get_sequential_patches(
        settings.patch_size,
        split="test",
        shuffle=False,
        seed=settings.seed,
        verbose=settings.verbose,
        return_cardinality=True,
    )
    images_ds = images_ds.map(image_loader.load)
    return images_ds


def load_models_from_settings(
    settings_path,
    models_path,
    frames_path,
    include=None,
    label_cols=None,
    number_labels=False,
    cache=None,
):
    with open(settings_path, "r") as settings_fp:
        settings = yaml.load(settings_fp, Loader=yaml.Loader)
    folder_names = list()
    model_labels = list()
    for i, model in enumerate(settings["models"]):
        if include is None or i in include:
            if number_labels:
                label = f"Model {i}"
            else:
                if label_cols is not None:
                    label_suffix = ", ".join(
                        [model["static_columns"][x] for x in label_cols]
                    )
                    label = f"{i} [{label_suffix}]"
                else:
                    label = f"{i} [{model['label']}]"
            model_labels.append(label)
            folder_names.append(model["folder"])
    models, settings = load_models(models_path, folder_names)
    ds = [get_images_ds(frames_path, s, cache=cache) for s in settings]
    return dict(
        models=models,
        settings=settings,
        labels=model_labels,
        ds=ds,
    )


class PredictionPlotter:
    _default_p_band_idx = 0
    _default_y_band_idx = 0
    _p_img_key = None
    _p_scalar_key = None

    def __init__(
        self,
        images_ds,
        models,
        labels,
        *,
        y_band_idx: Optional[Tuple[int, ...]] = None,
        p_band_idx: Optional[Tuple[int, ...]] = None,
        scale: float = 1.0,
        fontsize: int = 12,
    ):
        self._ds = images_ds
        self._models = models
        self._model_output_names = []
        for model in self._models:
            ns = model.output_names
            self._model_output_names.append({n: i for (i, n) in enumerate(ns)})
        self._labels = labels
        self._y_band_idx = y_band_idx
        self._p_band_idx = p_band_idx
        self._scale = scale
        self._fontsize = fontsize
        self._cmap = mpl.cm.get_cmap("gray")
        for x, _ in images_ds[0].take(1):
            self._img_bands = x.shape[-1]

    def get_data(self, n_skip: int):
        data = list()
        for model, ds, label in zip(self._models, self._ds, self._labels):
            for x, y in ds.skip(n_skip).take(1):
                p = model.predict(np.expand_dims(x, 0))
                data.append(dict(label=label, x=x, y=y, p=p))
        return data

    def plot_common_gt(
        self,
        n_skip: int,
        *,
        show_img: bool = True,
        show_hist: bool = False,
        bins: int = 10,
        figsize=(12, 6),
        include=None,
        scalar_format: str = "{:.2f}",
        log: bool = False,
    ):
        if include is None:
            n_cols = len(self._models) + 1
        else:
            n_cols = len(include) + 1
        if show_img:
            n_cols += 1
        n_rows = 1
        if show_hist:
            n_rows += 1
        fig = plt.figure(figsize=figsize)
        gs = mpl.gridspec.GridSpec(n_rows, n_cols)
        data = self.get_data(n_skip)

        col = 0
        if show_img:
            for row in range(min(self._img_bands, n_rows)):
                ax = fig.add_subplot(gs[row, col])
                if row == 0:
                    ax.set_title("Image", fontsize=self._fontsize)
                ax.imshow(data[0]["x"][:, :, row], cmap="gray")
                ax.axis("off")
            col += 1

        # Assumes a shared GT
        y = self.get_y_img(0, data[0])
        ys = self.get_y_scalar(0, data[0])
        ax = fig.add_subplot(gs[0, col])
        ys_f = scalar_format.format(ys)
        ax.set_title(f"True\n{ys_f}", fontsize=self._fontsize)
        ax.imshow(y, cmap=self._cmap)
        ax.axis("off")
        hax = None
        if show_hist:
            ax = fig.add_subplot(gs[1, col])
            hax = ax
            ax.hist(y.flatten() / self._scale, color="purple", bins=bins, log=log)
            ax.set_ylabel("Frequency")
            ax.tick_params(labelleft=False)
            if not log:
                ax.ticklabel_format(useOffset=False, axis="y")
            ax.set_xlabel("Pixel value")
        col += 1

        for i, d in enumerate(data):
            if include is not None and i not in include:
                continue
            p = self.get_p_img(i, d)
            ps = self.get_p_scalar(i, d)
            ps_f = scalar_format.format(ps)
            ax = fig.add_subplot(gs[0, col])
            ax.set_title(f"{d['label']} \n {ps_f}")
            ax.imshow(p, cmap=self._cmap)
            ax.axis("off")
            if show_hist:
                ax = fig.add_subplot(gs[1, col], sharey=hax, sharex=hax)
                ax.hist(
                    p.flatten() / self._scale,
                    color="purple",
                    bins=bins,
                    log=log,
                )
                ax.tick_params(labelleft=False)
                if not log:
                    ax.ticklabel_format(useOffset=False, axis="y")
            col += 1

        return fig

    def plot_unique_gt(
        self,
        n_skip: int,
        *,
        show_img: bool = True,
        show_hist: bool = False,
        bins: int = 10,
        figsize=(12, 6),
        include=None,
        scalar_format: str = "{:.2f}",
        log: bool = False,
    ):
        if include is None:
            n_cols = len(self._models)
        else:
            n_cols = len(include)
        if show_img:
            n_cols += 1
        n_rows = 2
        if show_hist:
            n_rows += 2
        fig = plt.figure(figsize=figsize)
        gs = mpl.gridspec.GridSpec(n_rows, n_cols)
        data = self.get_data(n_skip)

        col = 0
        if show_img:
            for row in range(min(self._img_bands, n_rows)):
                ax = fig.add_subplot(gs[row, col])
                if row == 0:
                    ax.set_title("Image", fontsize=self._fontsize)
                ax.imshow(data[0]["x"][:, :, row], cmap="gray")
                ax.axis("off")
            col += 1

        hax = None
        for i, d in enumerate(data):
            if include is not None and i not in include:
                continue
            p = self.get_p_img(i, d)
            y = self.get_y_img(i, d)
            ps = self.get_p_scalar(i, d)
            ps_f = scalar_format.format(ps)
            ys = self.get_y_scalar(i, d)
            ys_f = scalar_format.format(ys)

            ax = fig.add_subplot(gs[0, col])
            ax.set_title(f"{d['label']}")
            ax.imshow(y, cmap=self._cmap)
            ax.yaxis.set_ticklabels([])
            ax.yaxis.set_ticks([])
            ax.set_ylabel(f"True: {ys_f}")
            ax.xaxis.set_visible(False)

            ax = fig.add_subplot(gs[1, col])
            ax.imshow(p, cmap=self._cmap)
            ax.yaxis.set_ticklabels([])
            ax.yaxis.set_ticks([])
            ax.set_ylabel(f"Pred.: {ps_f}")
            ax.xaxis.set_visible(False)

            if show_hist:
                if hax is None:
                    ax = fig.add_subplot(gs[2, col])
                else:
                    ax = fig.add_subplot(gs[2, col], sharey=hax)
                hax = ax
                ax.hist(y.flatten() / self._scale, color="purple", bins=bins, log=log)
                ax.tick_params(labelleft=False)
                if not log:
                    ax.ticklabel_format(useOffset=False, axis="y")
                if i == 0:
                    ax.set_ylabel("True")
                ax = fig.add_subplot(gs[3, col], sharey=hax, sharex=hax)
                hax = ax
                ax.hist(p.flatten() / self._scale, color="purple", bins=bins, log=log)
                ax.tick_params(labelleft=False)
                if not log:
                    ax.ticklabel_format(useOffset=False, axis="y")
                if i == 0:
                    ax.set_xlabel("Pixel value")
                    ax.set_ylabel("Pred.")
            col += 1

        return fig

    def get_p_img(self, i, d):
        band_idx = self._default_p_band_idx
        if self._p_band_idx is not None:
            band_idx = self._p_band_idx[i]
        idx = self._model_output_names[i][self._p_img_key]
        return d["p"][idx][0, :, :, band_idx]

    def get_y_img(self, i, d):
        band_idx = self._default_y_band_idx
        if self._y_band_idx is not None:
            band_idx = self._y_band_idx[i]
        return d["y"][:, :, band_idx].numpy()

    def get_p_scalar(self, i, d):
        idx = self._model_output_names[i][self._p_scalar_key]
        return d["p"][idx][0, 0] / self._scale

    def get_y_scalar(self, i, d):
        raise NotImplementedError()


class CountPredictionPlotter(PredictionPlotter):
    _p_img_key = "density_map"
    _p_scalar_key = "count"

    def __init__(
        self,
        images_ds,
        models,
        labels,
        *,
        scale: float = 1e2,
        mask: bool = False,
        **kwargs,
    ):
        super().__init__(images_ds, models, labels, scale=scale, **kwargs)
        self._mask = mask
        self._cmap = mpl.cm.get_cmap("gray")
        if self._mask:
            self._cmap.set_bad(color="purple", alpha=1.0)

    def get_y_img(self, i, d):
        img = super().get_y_img(i, d) * self._scale
        if self._mask:
            return np.ma.masked_less_equal(img, 1e-7)
        return img

    def get_y_scalar(self, i, d):
        img = super().get_y_img(i, d)
        return np.sum(img)


class DMCoverPredictionPlotter(PredictionPlotter):
    _default_y_band_idx = 1
    _p_img_key = "density_map"
    _p_scalar_key = "cover"

    def __init__(self, images_ds, models, labels, threshold: float = 0.1, **kwargs):
        super().__init__(images_ds, models, labels, **kwargs)
        self._cmap = mpl.cm.get_cmap("gray")
        self._threshold = threshold

    def get_y_scalar(self, i, d):
        img = super().get_y_img(i, d)
        return np.mean(img)

    def get_p_img(self, i, d):
        img = super().get_p_img(i, d)
        return np.where(img > self._threshold, 1.0, 0.0)


class CoverPredictionPlotter(PredictionPlotter):
    _y_img_key = "segmentation_map"
    _y_scalar_key = "cover"
    _p_img_key = "segmentation_map"
    _p_scalar_key = "cover"

    def __init__(self, images_ds, models, labels, **kwargs):
        super().__init__(images_ds, models, labels, **kwargs)
        self._cmap = mpl.cm.get_cmap("gray")

    def get_y_scalar(self, i, d):
        img = super().get_y_img(i, d)
        return np.mean(img)
