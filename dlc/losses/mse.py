import tensorflow as tf

from dlc.losses.wrapper import LossFunctionWrapper


@tf.function
def _mean_squared_error(y_true, y_pred, *, weights):
    loss = tf.keras.backend.square(y_true - y_pred)
    loss = tf.keras.backend.mean(weights * loss, axis=(1, 2, 3))
    return loss


class MeanSquaredErrorV2(LossFunctionWrapper):
    def __init__(
        self,
        name="mse",
        weights_index=None,
        **kwargs,
    ):
        super(MeanSquaredErrorV2, self).__init__(
            _mean_squared_error,
            name=name,
            weights_index=weights_index,
            **kwargs,
        )
