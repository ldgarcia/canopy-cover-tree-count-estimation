import tensorflow as tf

from dlc.metrics.r_square import RSquareV2


__all__ = ["BinaryRSquareV2"]


class BinaryRSquareV2(RSquareV2):
    def __init__(
        self,
        name="binary_r_square",
        threshold=0.5,
        **kwargs,
    ):
        self._threshold = threshold
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        return super().update_state(
            y_true,
            y_pred >= self._threshold,
            sample_weight=sample_weight,
        )

    def get_config(self):
        config = {
            "threshold": self._threshold,
        }
        base_config = super().get_config()
        return {**base_config, **config}
