# Note: references information is listed in the losses module file.
import tensorflow as tf

from dlc.losses.focal import _assymetric_focal_loss
from dlc.losses.tversky import _assymetric_focal_tversky_loss
from dlc.losses.wrapper import LossFunctionWrapper

__all__ = ["UnifiedFocal"]


@tf.function
def _unified_focal_loss(
    # Adapted from: https://git.io/J81xb
    y_true,
    y_pred,
    *,
    weights,
    alpha,
    beta,
    lmbda,
    gamma,
    eps=tf.keras.backend.epsilon(),
):
    tversky_loss = _assymetric_focal_tversky_loss(
        y_true,
        y_pred,
        weights=weights,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        eps=eps,
    )
    focal_loss = _assymetric_focal_loss(
        y_true,
        y_pred,
        weights=weights,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        eps=eps,
    )

    term1 = lmbda * tversky_loss
    term2 = (1.0 - lmbda) * focal_loss
    loss = term1 + term2
    return loss


class UnifiedFocal(LossFunctionWrapper):
    def __init__(
        self,
        name="unified_focal",
        reduction=tf.keras.losses.Reduction.AUTO,
        ground_truth_index=0,
        weights_index=None,
        # Default hyperparamters from [5].
        alpha=0.6,  # importance of False Positives (called delta in [5])
        beta=None,  # importance of False Negatives
        # Using default hyperparameter from [5].
        gamma=0.2,  # controls focal modulation
        # Using default hyperparameter from [5].
        lmbda=0.5,
        eps=tf.keras.backend.epsilon(),
    ):
        if beta is None:
            beta = 1.0 - alpha
        super().__init__(
            _unified_focal_loss,
            name=name,
            reduction=reduction,
            ground_truth_index=ground_truth_index,
            weights_index=weights_index,
            # kwargs
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lmbda=lmbda,
            eps=eps,
        )
