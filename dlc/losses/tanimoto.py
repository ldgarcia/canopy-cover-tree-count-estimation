# Note: references information is listed in the losses module file.
import tensorflow as tf

from dlc.losses.wrapper import LossFunctionWrapper


@tf.function
def _dice3_index(
    y_true,
    y_pred,
    *,
    weights,
    eps,
):
    """Compute a weighted Dice3 index."""
    # Assumes inputs with shape: (batch_size, height, width, 1)
    # See: [1, Equation 3]
    p0 = y_pred
    g0 = y_true

    num = tf.keras.backend.sum(weights * p0 * g0)
    den = tf.keras.backend.sum(weights * (p0 ** 2 + g0 ** 2))
    index = (num + eps) / (den - num + eps)
    return index


@tf.function
def tanimoto_index(
    y_true,
    y_pred,
    *,
    weights=None,
    eps=tf.keras.backend.epsilon(),
):
    """Compute a weighted Tanimoto index."""
    if weights is None:
        weights = tf.ones_like(y_true)
    # Assumes inputs with shape: (batch_size, height, width, 1)
    # See: [1, Equation 4]
    d3_t = _dice3_index(
        y_true,
        y_pred,
        weights=weights,
        eps=eps,
    )
    d3_nt = _dice3_index(
        1.0 - y_true,
        1.0 - y_pred,
        weights=weights,
        eps=eps,
    )
    index = 0.5 * (d3_t + d3_nt)
    return index


@tf.function
def tanimoto(
    y_true,
    y_pred,
    *,
    weights=None,
    eps=tf.keras.backend.epsilon(),
):
    """Compute a weighted Tanimoto loss."""
    # Assumes inputs with shape: (batch_size, height, width, 1)
    index = tanimoto_index(y_true, y_pred, weights=weights, eps=eps)
    return 1.0 - index


class Tanimoto(LossFunctionWrapper):
    """Compute a weighted Tanimoto (with complement) loss."""

    def __init__(
        self,
        name="tanimoto",
        reduction: str = tf.keras.losses.Reduction.NONE,
        ground_truth_index=0,
        weights_index=None,
        eps=tf.keras.backend.epsilon(),
    ):
        super().__init__(
            tanimoto,
            name=name,
            reduction=reduction,
            # Custom to our pipeline:
            ground_truth_index=ground_truth_index,
            weights_index=weights_index,
            # Custom to this loss function:
            eps=eps,
        )
