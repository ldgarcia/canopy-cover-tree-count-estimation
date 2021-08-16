from typing import Optional

import tensorflow as tf
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import weights_broadcast_ops


class LossFunctionWrapper(tf.keras.losses.Loss):
    # Adapted from https://git.io/JWQrD and https://git.io/JWhew
    def __init__(
        self,
        fn: tf.function,
        name=None,
        reduction=tf.keras.losses.Reduction.AUTO,
        # Custom to our pipeline:
        ground_truth_index: int = 0,
        weights_index: Optional[int] = None,
        # Custom to the specific loss function:
        **kwargs,
    ):
        super().__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

        self._ground_truth_index = ground_truth_index
        self._weights_index = weights_index

    def call(self, y_true, y_pred):
        ground_truth = y_true[..., self._ground_truth_index]
        ground_truth = tf.expand_dims(ground_truth, -1)
        weights = tf.constant(1.0, dtype=y_pred.dtype)
        if self._weights_index is not None:
            weights = y_true[..., self._weights_index]
            weights = tf.expand_dims(weights, -1)
        # y_pred, ground_truth = losses_utils.squeeze_or_expand_dimensions(
        #     y_pred, y_true=ground_truth
        # )
        weights = weights_broadcast_ops.broadcast_weights(
            weights=weights, values=ground_truth
        )
        # tensor_kwargs = {
        #    k: tf.constant(v, dtype=y_true.dtype) for k, v in self._fn_kwargs.items()
        # }
        ag_fn = autograph.tf_convert(self.fn, ag_ctx.control_status_ctx())
        return ag_fn(
            ground_truth,
            y_pred,
            **{**self._fn_kwargs, "weights": weights},
        )

    def get_config(self):
        config = dict(
            ground_truth_index=self._ground_truth_index,
            weights_index=self._weights_index,
        )
        for k, v in self._fn_kwargs.items():
            config[k] = (
                tf.keras.backend.eval(v)
                if (tf.is_tensor(v) or isinstance(v, tf.Variable))
                else v
            )
        base_config = super().get_config()
        return {**base_config, **config}
