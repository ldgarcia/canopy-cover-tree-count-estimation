# [1] Avi-Aharon, M., Arbelle, A., & Raviv, T. R. (2020).
#     DeepHist: Differentiable Joint and Color Histogram Layers for
#     Image-to-Image Translation.
#     arXiv preprint arXiv:2005.03995. URL: https://arxiv.org/abs/2005.03995
#
# [2] Rubner, Y., Tomasi, C., & Guibas, L. J. (2000).
#     The earth mover's distance as a metric for image retrieval.
#     International journal of computer vision, 40(2), 99-121.
#
# [3] Hou, L., Yu, C. P., & Samaras, D. (2016).
#     Squared earth mover's distance-based loss for training deep neural networks.
#     arXiv preprint arXiv:1611.05916. URL: https://arxiv.org/abs/1611.05916
import tensorflow as tf

__all__ = ["EarthMoversDistance"]


@tf.function
def _squared_earth_movers_distance(
    y_true,
    y_pred,
):
    # See [2, Eq. 14] and [3, Eq. 10]
    # Assumptions from [3] interpreted for histograms:
    # Compared histograms:
    # 1) Must have equal mass.
    # 2) Use the same bins (umk-wk, umk+wk) for any k.
    y_true_cdf = tf.math.cumsum(y_true, axis=-1)
    y_pred_cdf = tf.math.cumsum(y_pred, axis=-1)
    distance = tf.math.reduce_sum(
        tf.math.squared_difference(
            y_true_cdf,
            y_pred_cdf,
        ),
        axis=-1,
    )
    return distance


class SquaredEarthMoversDistance(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        distance = _squared_earth_movers_distance(y_true, y_pred)
        reduction = self._get_reduction()
        if reduction == tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE:
            return tf.reduce_mean(distance)
        return distance
