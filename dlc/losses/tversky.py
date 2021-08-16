# Note: references information is listed in the losses module file.
import tensorflow as tf

from dlc.losses.wrapper import LossFunctionWrapper


__all__ = ["Tversky", "AssymetricFocalTversky"]


@tf.function
def _tversky_index(
    # Adapted from: https://git.io/JWXSm
    # See also: https://git.io/JWXDg
    y_true,
    y_pred,
    *,
    weights,
    alpha,
    beta,
    eps,
):
    p_t = y_pred  # proba of pixel belonging to target class
    p_nt = 1.0 - p_t  # proba of pixel belonging to non-target class
    g_t = y_true
    g_nt = 1.0 - g_t

    # Assumes input with shape [batch, N1, N2, 1].
    tp = tf.keras.backend.sum(weights * p_t * g_t)
    fp = tf.keras.backend.sum(weights * p_t * g_nt)
    fn = tf.keras.backend.sum(weights * p_nt * g_t)
    index = (tp + eps) / (tp + (alpha * fp) + (beta * fn) + eps)
    return index


@tf.function
def _tversky_loss(
    y_true,
    y_pred,
    *,
    weights,
    alpha,
    beta,
    eps=tf.keras.backend.epsilon(),
):
    # We assume that the ground truth for binary segmentation consists
    # in a single-channel mask, with 1 represeting the target class and
    # 0 its complement.
    # This is inspired in how the Tanimoto loss with complement is
    # formulated by [4].

    index_t = _tversky_index(
        y_true,
        y_pred,
        alpha=alpha,
        beta=beta,
        weights=weights,
        eps=eps,
    )
    loss_t = 1.0 - index_t

    # Complement of target class.
    index_nt = _tversky_index(
        1.0 - y_true,
        1.0 - y_pred,
        alpha=alpha,
        beta=beta,
        weights=weights,
        eps=eps,
    )
    loss_nt = 1.0 - index_nt

    # Weighted loss
    loss = 0.5 * (loss_t + loss_nt)
    return loss


class Tversky(LossFunctionWrapper):
    """The loss function from [6]."""

    def __init__(
        self,
        name="tversky",
        reduction=tf.keras.losses.Reduction.NONE,
        ground_truth_index=0,
        # To support sample-weighting as in the U-Net paper and as in [1].
        weights_index=None,
        # Using hyperparameters from the reference paper.
        alpha=0.3,  # importance of False Positives
        beta=None,  # importance of False Negatives
        eps=tf.keras.backend.epsilon(),
    ):
        if beta is None:
            beta = 1.0 - alpha
        super().__init__(
            _tversky_loss,
            name=name,
            reduction=reduction,
            ground_truth_index=ground_truth_index,
            weights_index=weights_index,
            # kwargs
            alpha=alpha,
            beta=beta,
            eps=eps,
        )


@tf.function
def _assymetric_focal_tversky_loss(
    # Adapted from: https://git.io/J81Xf
    y_true,
    y_pred,
    *,
    weights,
    alpha,
    beta,
    gamma,
    eps=tf.keras.backend.epsilon(),
):
    # We assume that the ground truth for binary segmentation consists
    # in a single-channel mask, so, for computing the assymetric loss,
    # we compute its complement (as also done in the Tverky loss above).
    # This is inspired in how the Tanimoto loss with complement is
    # formulated by [4].

    # Target class (i.e., foreground in our case).
    index_t = _tversky_index(
        y_true,
        y_pred,
        alpha=alpha,
        beta=beta,
        weights=weights,
        eps=eps,
    )
    loss_t = 1.0 - index_t
    modulated_loss_t = tf.keras.backend.pow(loss_t, 1.0 - gamma)

    # Complement of target class.
    index_nt = _tversky_index(
        1.0 - y_true,
        1.0 - y_pred,
        alpha=alpha,
        beta=beta,
        weights=weights,
        eps=eps,
    )
    loss_nt = 1.0 - index_nt

    # Weighted loss
    loss = modulated_loss_t + loss_nt
    return loss


class AssymetricFocalTversky(LossFunctionWrapper):
    def __init__(
        self,
        name="assymetric_focal_tversky",
        reduction=tf.keras.losses.Reduction.NONE,
        ground_truth_index=0,
        weights_index=None,
        alpha=0.6,  # importance of False Positives
        beta=None,  # importance of False Negatives
        # Using default hyperparameter from [3] and [5].
        gamma=0.2,  # controls focal modulation
        eps=tf.keras.backend.epsilon(),
    ):
        if beta is None:
            beta = 1.0 - alpha
        super().__init__(
            _assymetric_focal_tversky_loss,
            name=name,
            reduction=reduction,
            ground_truth_index=ground_truth_index,
            weights_index=weights_index,
            # kwargs
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            eps=eps,
        )
