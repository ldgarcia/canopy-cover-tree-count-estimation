# Note: references information is listed in the losses module file.
import tensorflow as tf

from dlc.losses.wrapper import LossFunctionWrapper

__all__ = ["AssymetricFocal"]


@tf.function
def _assymetric_focal_loss(
    # Adapted from: https://git.io/J81dl
    # See also: https://git.io/JWMQy
    y_true,
    y_pred,
    *,
    weights,
    alpha,
    beta,
    gamma,
    eps=tf.keras.backend.epsilon(),
):
    y_pred = tf.keras.backend.clip(y_pred, eps, 1.0 - eps)
    # Target class
    bce_t = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    # Complement of target class
    modulating_factor = tf.keras.backend.pow(y_pred, gamma)
    bce_nt = tf.keras.backend.binary_crossentropy(1.0 - y_true, 1.0 - y_pred)
    # Weighted loss
    loss = alpha * bce_t + beta * modulating_factor * bce_nt
    loss = tf.keras.backend.mean(weights * loss)
    return loss


class AssymetricFocal(LossFunctionWrapper):
    def __init__(
        self,
        name="assymetric_focal",
        reduction=tf.keras.losses.Reduction.NONE,
        ground_truth_index=0,
        weights_index=None,
        alpha=0.6,  # importance of False Positives
        beta=None,  # importance of False Negatives
        gamma=0.2,  # controls focal modulation
        eps=tf.keras.backend.epsilon(),
    ):
        if beta is None:
            beta = 1.0 - alpha
        super().__init__(
            _assymetric_focal_loss,
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
