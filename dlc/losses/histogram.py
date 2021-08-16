# References:
# [1] Avi-Aharon, M., Arbelle, A., & Raviv, T. R. (2020).
#     DeepHist: Differentiable Joint and Color Histogram Layers for
#     Image-to-Image Translation.
#     arXiv preprint arXiv:2005.03995. URL: https://arxiv.org/abs/2005.03995
from typing import Optional
from typing import Tuple

import tensorflow as tf

from dlc.layers.histogram import histogram
from dlc.losses.emd import _squared_earth_movers_distance
from dlc.losses.mse import _mean_squared_error
from dlc.losses.tanimoto import tanimoto
from dlc.losses.wrapper import LossFunctionWrapper


__all__ = ["MSEPlusHistogram", "TanimotoPlusHistogram"]


@tf.function
def _histogram_mse(
    y_true,
    y_pred,
    *,
    weights,
    # histogram
    k,
    start,
    stop,
    # weighting
    alpha1=0.5,
    alpha2=0.5,
):
    primary_loss = _mean_squared_error(y_true, y_pred, weights=weights)
    hist_true = histogram(
        y_true,
        k=k,
        start=start,
        stop=stop,
    )
    hist_pred = histogram(
        y_pred,
        k=k,
        start=start,
        stop=stop,
    )
    hist_loss = _squared_earth_movers_distance(hist_true, hist_pred)
    loss = alpha1 * primary_loss + alpha2 * hist_loss
    return loss


class MSEPlusHistogram(LossFunctionWrapper):
    def __init__(
        self,
        name="mse_hist",
        reduction=tf.keras.losses.Reduction.AUTO,
        ground_truth_index=0,
        weights_index=None,
        # kwargs
        alpha1=0.5,
        alpha2=0.5,
        k=64,
        start=0.0,
        stop=1.0,
    ):
        super(MSEPlusHistogram, self).__init__(
            _histogram_mse,
            name=name,
            reduction=reduction,
            ground_truth_index=ground_truth_index,
            weights_index=weights_index,
            # kwargs
            alpha1=alpha1,
            alpha2=alpha2,
            k=k,
            start=start,
            stop=stop,
        )


@tf.function
def _histogram_tanimoto(
    y_true,
    y_pred,
    *,
    weights,
    # histogram
    k,
    start,
    stop,
    # weighting
    alpha1=0.5,
    alpha2=0.5,
):
    primary_loss = tanimoto(y_true, y_pred, weights=weights)
    hist_true = histogram(
        y_true,
        k=k,
        start=start,
        stop=stop,
    )
    hist_pred = histogram(
        y_pred,
        k=k,
        start=start,
        stop=stop,
    )
    hist_loss = _squared_earth_movers_distance(hist_true, hist_pred)
    loss = alpha1 * primary_loss + alpha2 * hist_loss
    return loss


class TanimotoPlusHistogram(LossFunctionWrapper):
    def __init__(
        self,
        name="tanimoto_hist",
        reduction=tf.keras.losses.Reduction.AUTO,
        ground_truth_index=0,
        weights_index=None,
        # kwargs
        alpha1=0.5,
        alpha2=0.5,
        k=64,
        start=0.0,
        stop=1.0,
    ):
        super(TanimotoPlusHistogram, self).__init__(
            _histogram_tanimoto,
            name=name,
            reduction=reduction,
            ground_truth_index=ground_truth_index,
            weights_index=weights_index,
            # kwargs
            alpha1=alpha1,
            alpha2=alpha2,
            k=k,
            start=start,
            stop=stop,
        )


class MutualInformation(tf.keras.losses.Loss):
    # TODO: Status WIP
    # See [1, Eqs. 15 through 17]
    def __init__(
        self,
        k: int,
        start: float,
        stop: float,
        name="mutual_info",
        **kwargs,
    ):
        super(MutualInformation, self).__init__(name=name, **kwargs)
        self.histogram = HistogramLayer(k, start, stop)
        self.joint_histogram = JointHistogramLayer(k, start, stop)

    def call(self, y_true, y_pred):
        # Assumes y_true and y_pred are 2D image batches
        hist_true = self.histogram(y_true)
        hist_pred = self.histogram(y_pred)
        hist_joint = self.joint_histogram([y_true, y_pred])
        # See [1, Eq. 15]
        mutual_info = tf.multiply(
            hist_joint,
            tf.math.log1p(
                tf.math.divide_no_nan(
                    hist_joint,
                    # Compute outer product
                    tf.tensordot(
                        hist_true,
                        hist_pred,
                        axes=(0, 0),
                    ),
                )
            ),
        )
        mutual_info = tf.reduce_sum(mutual_info, axis=(1, 2))
        # See [1, Eq. 17]
        joint_entropy = tf.multiply(hist_joint, tf.math.log1p(hist_joint))
        joint_entropy = -tf.reduce_sum(joint_entropy, axis=(1, 2))
        # See [1, Eq. 16]
        loss = 1.0 - tf.math.divide_no_nan(mutual_info, joint_entropy)

        reduction = self._get_reduction()
        if reduction == tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE:
            return tf.reduce_mean(loss)
        return loss
