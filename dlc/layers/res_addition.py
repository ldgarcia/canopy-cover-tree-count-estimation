from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.base_layer import Layer

__all__ = ["EncoderResAddition", "DecoderResAddition"]

# Note: the idea of these custom layers is to implement addition residuals
# in an encoder-decoder architecture without having to resort to 1x1xf
# convolutions to increase the number of filters of input features.
# As currently, they work until the evaluation stage, after which a
# batch shape error is raised. Need to debug this util this alternative
# can be explored further. The U-Net unit showcasing their use is
# the *res_sdunet* unit.


class EncoderResAddition(Layer):
    def __init__(self, **kwargs):
        super(EncoderResAddition, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, (tuple, list)):
            error_msg = f"An EncoderResAddition layer should be called on a tuple, got shape: {input_shape}"
            raise ValueError(error_msg)
        if len(input_shape) != 2:
            error_msg = f"An EncoderResAddition layer should be called on a 2-tuple, got shape: {input_shape}"
            raise ValueError(error_msg)
        x_shape = input_shape[0]
        r_shape = input_shape[1]
        if x_shape[1] != r_shape[1]:
            error_msg = "x and r have different height"
            raise ValueError(error_msg)
        if x_shape[2] != r_shape[2]:
            error_msg = "x and r have different width"
            raise ValueError(error_msg)
        self._height = x_shape[1]
        self._width = x_shape[2]
        self._x_depth = x_shape[3]
        self._r_depth = r_shape[3]
        if self._r_depth > x_shape[3]:
            error_msg = "r has more channels than x"
            raise ValueError(error_msg)

    def call(self, input):
        x, r = input
        batch_size = backend.shape(x)[0]
        x_a = backend.array_ops.slice(
            x,
            [0, 0, 0, 0],
            [batch_size, self._height, self._width, self._r_depth],
        )
        x_b = backend.array_ops.slice(
            x,
            [0, 0, 0, self._r_depth],
            [
                batch_size,
                self._height,
                self._width,
                self._x_depth - self._r_depth,
            ],
        )
        y = backend.array_ops.concat([x_a + r, x_b], axis=3)
        return y


class DecoderResAddition(Layer):
    def __init__(self, **kwargs):
        super(DecoderResAddition, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, (tuple, list)):
            error_msg = f"An DecoderResAddition layer should be called on a tuple, got shape: {input_shape}"
            raise ValueError(error_msg)
        if len(input_shape) != 2:
            error_msg = f"An DecoderResAddition layer should be called on a 2-tuple, got shape: {input_shape}"
            raise ValueError(error_msg)
        x_shape = input_shape[0]
        r_shape = input_shape[1]
        if x_shape[1] != r_shape[1]:
            error_msg = "x and r have different height"
            raise ValueError(error_msg)
        if x_shape[2] != r_shape[2]:
            error_msg = "x and r have different width"
            raise ValueError(error_msg)
        self._height = x_shape[1]
        self._width = x_shape[2]
        self._x_depth = x_shape[3]
        if self._x_depth > r_shape[3]:
            error_msg = "x has more channels than r"
            raise ValueError(error_msg)

    def call(self, input):
        x, r = input
        batch_size = backend.shape(x)[0]
        r_a = backend.array_ops.slice(
            r,
            [0, 0, 0, 0],
            [batch_size, self._height, self._width, self._x_depth],
        )
        y = x + r_a
        return y
