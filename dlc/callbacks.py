import pathlib
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output
from tensorflow.python.keras.utils import tf_utils

from dlc.tools.plots import ColorMapFactory
from dlc.tools.plots import ColorMapSpec
from dlc.tools.plots import plot_batch
from dlc.tools.plots import plot_multioutput_batch
from dlc.tools.stopwatch import StopWatch


class BaseBatchPredictionPlotCallback(tf.keras.callbacks.Callback):
    """Plots predictions for a fixed batch."""

    # References: https://git.io/JZ9El
    #             https://www.tensorflow.org/tutorials/images/segmentation#train_the_model
    def __init__(
        self,
        validation_data,
        *,
        cmap: Tuple[ColorMapSpec, ...] = (),
        n_take: int = 1,
        n_skip: int = 0,
        n_max_samples: Optional[int] = None,
        output_path: Optional[Union[str, pathlib.Path]] = None,
        filename: Optional[str] = None,
        save_best_only: bool = False,
        interactive: bool = False,
        mode: str = "auto",
        monitor: str = "val_loss",
        verbose: bool = True,
        show_hist: bool = True,
    ) -> None:
        self._stopwatch = StopWatch()
        self._validation_data = validation_data
        self._output_path = output_path
        if self._output_path is not None:
            self._output_path = pathlib.Path(self._output_path)
            if not self._output_path.exists():
                self._output_path.mkdir(parents=True, exist_ok=False)
        self._filename = filename
        self._interactive = interactive
        self._n_max_samples = n_max_samples
        self._n_take = n_take
        self._n_skip = n_skip
        self._verbose = verbose
        self._show_hist = show_hist
        self._cmf = ColorMapFactory()
        self._cmf.add_groups(cmap)
        # Adapted from: https://git.io/JZ9ab
        self._save_best_only = save_best_only
        self._monitor = monitor
        self._last_fig = None
        if mode == "min":
            self._monitor_op = np.less
            self.best = np.Inf
        elif mode == "max":
            self._monitor_op = np.greater
            self.best = -np.Inf
        else:
            if not hasattr(self, "monitor"):
                self.monitor = self._monitor

            if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self._monitor_op = np.greater
                self.best = -np.Inf
            else:
                self._monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        self._stopwatch.start()
        if self._interactive:
            clear_output(wait=True)
        # Adapted from: https://git.io/JZ9wn
        logs = logs or {}
        logs = tf_utils.sync_to_numpy_or_python_type(logs)
        current = logs.get(self._monitor)
        if not self._save_best_only or self._monitor_op(current, self.best):
            if self._verbose:
                print("Plotting predictions on validation batch samples...")
            self.best = current
            self.plot(epoch)
        self._stopwatch.stop()

    def compute_time_statistics(self) -> Tuple[float, float]:
        mean = self._stopwatch.mean()
        std = self._stopwatch.std()
        return mean, std


class BatchPredictionPlotCallback(BaseBatchPredictionPlotCallback):
    def __init__(
        self,
        validation_data,
        **kwargs,
    ):
        super().__init__(validation_data, **kwargs)

    def plot(self, epoch=None) -> None:
        for batch_id, (xs, ys) in enumerate(
            self._validation_data.skip(self._n_skip).take(self._n_take),
        ):
            title = "Epoch {} ({}: {:.4f})".format(
                epoch + 1,
                self._monitor,
                self.best,
            )
            yhs = self.model.predict(xs)
            fig = plot_batch(
                xs,
                ys,
                yhs,
                title=title,
                cmf=self._cmf,
                show_hist=self._show_hist,
            )
            if self._output_path is not None:
                fig.savefig(
                    self._output_path.joinpath(
                        "{:04d}-{}.png".format(
                            epoch + 1,
                            batch_id + 1,
                        )
                    )
                )
            plt.close(fig)


class MultiOutputBatchPredictionPlotCallback(BaseBatchPredictionPlotCallback):
    def __init__(
        self,
        validation_data,
        *,
        keys: Tuple[str, ...] = (),
        **kwargs,
    ):
        super().__init__(validation_data, **kwargs)
        self._keys = keys

    def plot(self, epoch=None) -> None:
        for batch_id, (xs, ys) in enumerate(
            self._validation_data.skip(self._n_skip).take(self._n_take),
        ):
            title = "Epoch {} ({}: {:.4f})".format(
                epoch + 1,
                self._monitor,
                self.best,
            )
            yhs = self.model.predict(xs)
            yhs_names = self.model.output_names
            fig = plot_multioutput_batch(
                xs,
                ys,
                yhs,
                yhs_names=yhs_names,
                title=title,
                cmf=self._cmf,
                show_hist=self._show_hist,
                keys=self._keys,
            )
            if self._output_path is not None:
                fig.savefig(
                    self._output_path.joinpath(
                        "{:04d}-{}.png".format(
                            epoch + 1,
                            batch_id + 1,
                        )
                    )
                )
            plt.close(fig)
