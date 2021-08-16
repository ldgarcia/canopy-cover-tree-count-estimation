import tensorflow as tf


__all__ = ["MeanAbsoluteErrorV2"]


class MeanAbsoluteErrorV2(tf.keras.metrics.MeanAbsoluteError):
    def __init__(self, y_true_index=None, **kwargs):
        super().__init__(**kwargs)
        self.y_true_index = y_true_index

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.y_true_index is not None:
            y_true = y_true[..., self.y_true_index]
            y_true = tf.expand_dims(y_true, -1)
        super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = {
            "y_true_index": self.y_true_index,
        }
        base_config = super().get_config()
        return {**base_config, **config}
