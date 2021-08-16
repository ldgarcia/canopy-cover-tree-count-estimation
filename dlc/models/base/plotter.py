import abc
import importlib
import pathlib
from typing import List
from typing import Optional
from typing import Union

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

import dlc.tools.cache
import dlc.tools.config
import dlc.tools.datasets
import dlc.tools.images
import dlc.tools.plots
import dlc.tools.splits
from dlc.models.base.settings import Settings

__all__ = ["Plotter"]


def _load_model(settings: Settings, filename: str = "weights.h5") -> tf.keras.Model:
    module = importlib.import_module(f"dlc.models.{settings.model}")
    load_model = getattr(module, "load_model")
    model = load_model(
        input_shape=settings.input_shape,
        batch_size=settings.training_batch_size,
        weights_file=settings.get_weights_file(filename),
        **dict(settings),
    )
    model.compile(
        optimizer=settings.get_optimizer(),
        loss=settings.get_loss(),
        loss_weights=settings.loss_weights,
        metrics=settings.get_metrics(),
        run_eagerly=settings.run_eagerly,
    )
    return model


def _plot_validation_batch(settings: Settings):
    # Load trained model
    best_model = _load_model(settings)
    latest_model = _load_model(settings, filename="model.h5")
    # Load validation data
    frames = gpd.read_file(settings.frames_path)
    frames_query = settings.frames_query
    if frames_query is not None:
        print(f"Selecting from {len(frames)} frame rows...")
        print(f"Executing query: {frames_query}")
        frames.query(frames_query, inplace=True)
    print(f"Selected {len(frames)} frame rows.")

    splitter = settings.get_splitter()
    if not hasattr(splitter, settings.splitter.method):
        raise ValueError("Invalid splitter method")
    split_method = getattr(splitter, settings.splitter.method)
    splits, _ = split_method(
        frames,
        seed=settings.splitter_seed,
        **settings.splitter.method_parameters,
    )
    if settings.cross_validation_split_index is not None:
        splits = splits[settings.cross_validation_split_index]
    image_ds_gen = dlc.tools.datasets.ImageDatasetGenerator(
        frames,
        splits=splits,
        splits_map=splitter.get_splits_map(),
        image_keys=(
            settings.feature_keys,
            settings.annotation_keys,
        ),
        input_base_path=settings.frames_directory,
        seed=settings.seed,
    )
    p = settings.validation_local_standardization_p
    image_loader = dlc.tools.images.ImageLoader(
        local_standardization_p=p,
        cache=None,
        seed=settings.seed,
        defaults=settings.loader_patch_defaults,
    )
    ds = image_ds_gen.get_sequential_patches(
        settings.patch_size,
        split="test",
        shuffle=False,
        seed=settings.seed,
        verbose=settings.verbose,
        return_cardinality=False,
    )
    if settings.validation_n_skip is not None:
        ds = ds.skip(settings.validation_n_skip)
    ds = ds.map(image_loader.load)
    for transformer in settings.get_transformers():
        ds = ds.map(transformer)
    ds = ds.batch(settings.validation_batch_size)

    n_skip = settings.best_plot_n_skip
    n_take = settings.best_plot_n_take
    cmf = dlc.tools.plots.ColorMapFactory()
    cmf.add_groups(settings.validation_plot_cmap)
    for model, prefix in zip([best_model, latest_model], ["best", "latest"]):
        output_path = settings.out_dir.joinpath(f"plots/{prefix}")
        if not output_path.exists():
            output_path.mkdir(parents=True)
        if settings.validation_plot_keys is not None:
            for batch_id, (xs, ys) in enumerate(ds.skip(n_skip).take(n_take)):
                yhs = model.predict(xs)
                yhs_names = model.output_names
                fig = dlc.tools.plots.plot_multioutput_batch(
                    xs,
                    ys,
                    yhs,
                    yhs_names=yhs_names,
                    keys=settings.validation_plot_keys,
                    cmf=cmf,
                )

                fig.savefig(output_path.joinpath(f"{prefix}-{batch_id}.png"))
                plt.close(fig)
        else:
            for batch_id, (xs, ys) in enumerate(ds.skip(n_skip).take(n_take)):
                yhs = model.predict(xs)
                fig = dlc.tools.plots.plot_batch(
                    xs,
                    ys,
                    yhs,
                    keys=settings.validation_plot_keys,
                    cmf=cmf,
                )
                fig.savefig(output_path.joinpath(f"{prefix}-{batch_id}.png"))
                plt.close(fig)


def _plot_history(
    keys: List[str],
    history: Union[dict, tf.keras.callbacks.History],
    *,
    title: Optional[str] = None,
    output_path: Union[None, str, pathlib.Path] = None,
    yscale: str = "linear",
) -> None:
    if isinstance(history, tf.keras.callbacks.History):
        data = history.history
    else:
        data = history
    val_keys = [f"val_{key}" for key in keys]
    fig = plt.figure(figsize=(4.5 * len(keys), 3.5))
    gs = fig.add_gridspec(1, len(keys))
    if title:
        fig.suptitle(title)
    x_ticks = None
    for i, (key, val_key) in enumerate(zip(keys, val_keys)):
        x_ticks = list(range(len(data[key])))
        ax = fig.add_subplot(gs[0, i])
        ax.plot(x_ticks, data[key], label=key)
        ax.plot(x_ticks, data[val_key], label=val_key)
        ax.set_ylabel(key)
        ax.set_xlabel("Epoch")
        # ax.set_xticks(x_ticks)
        ax.set_yscale(yscale)
        ax.legend()
    if output_path is not None:
        output_path = pathlib.Path(output_path)
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        fig.savefig(output_path)
        plt.close(fig)


class Plotter(metaclass=abc.ABCMeta):
    def __init__(self, settings: dlc.models.base.settings.Settings):
        self.settings = settings

    def run(self):
        history_path = self.settings.out_dir.joinpath("training.csv")
        if history_path.exists():
            history = pd.read_csv(history_path)
            if len(history["epoch"]) > 1:
                for plot in self.settings.plots:
                    output_path = self.settings.out_dir.joinpath("plots")
                    output_path = output_path.joinpath(plot["filename"])
                    _plot_history(
                        plot["keys"],
                        history,
                        output_path=str(output_path),
                        yscale=plot.get("yscale", "linear"),
                    )
                # Print validation batch samples
                _plot_validation_batch(self.settings)
            else:
                print("Skipping plots (epochs < 2).")
        else:
            print("No history to plot.")


if __name__ == "__main__":
    config = dlc.tools.config.read_config()
    setting_dict = config["settings"]
    module = importlib.import_module(f"dlc.models.{setting_dict['model']}")
    settings_t = getattr(module, "Settings", Settings)
    settings = settings_t(**setting_dict)
    plotter = Plotter(settings)
    plotter.run()
