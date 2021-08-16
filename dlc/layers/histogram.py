# References:
# [1] Yusuf, I., Igwegbe, G., & Azeez, O. (2020).
#     Differentiable Histogram with Hard-Binning.
#     arXiv preprint arXiv:2012.06311. URL: https://arxiv.org/abs/2012.06311
#
# [2] Avi-Aharon, M., Arbelle, A., & Raviv, T. R. (2020).
#     DeepHist: Differentiable Joint and Color Histogram Layers for
#     Image-to-Image Translation.
#     arXiv preprint arXiv:2005.03995. URL: https://arxiv.org/abs/2005.03995
import tensorflow as tf


@tf.function
def histogram(inputs, *, k: int, start: float, stop: float):
    edges = tf.linspace(start, stop, num=k + 1)
    wk = tf.constant(0.5) * (edges[1:] - edges[:-1])
    muk = edges[:-1] + wk
    # Assumes 1 channel input
    kernel_ones = tf.ones((1, 1, 1, k))
    # See: [2, p.8, figure 4]
    x = inputs
    # Compute x - muk
    x = tf.keras.backend.conv2d(
        x,
        kernel_ones,
        padding="same",
        data_format="channels_last",
    )
    x = x - muk
    # Compute |x - muk|
    x = tf.math.abs(x)
    # Compute wk - |x - muk|
    x = tf.keras.backend.conv2d(
        x,
        tf.constant(-1.0) * kernel_ones,
        padding="same",
        data_format="channels_last",
    )
    x = x + wk
    # Compute 1.01 ^ (wk - |x - muk|)
    x = tf.math.pow(tf.constant(1.01), x)
    # Compute the "ReLU at one". See [1, Eq. 2].
    x = tf.keras.backend.relu(x, threshold=1.0)
    # Perform the global average pooling
    x = tf.keras.backend.mean(x, axis=[1, 2])
    return x


class HistogramLayer(tf.keras.layers.Layer):
    def __init__(self, k: int, start: float, stop: float, **kwargs):
        """Compute a differentiable histogram of equally-sized linearly-spaced bins.

        Parameters
        ----------
        k
          The number of bins
        start
          The start value of the first bin
        stop
          The stop value of the last bin (exclusive).
        name
          The name of the layer.
        """
        # Implements the layer described in [1].
        super(HistogramLayer, self).__init__(**kwargs)
        self.k = k
        self.start = start
        self.stop = stop
        # Compute the k + 1 edges that define the k bins.
        edges = tf.linspace(self.start, self.stop, num=self.k + 1)
        # The kth bin corresponds to (muk - wk, muk + wk)
        # See [1, Eq. 1]
        # The line below is equivalent to wk = np.diff(edges) / 2
        self.wk = tf.constant(0.5) * (edges[1:] - edges[:-1])
        self.muk = edges[:-1] + self.wk

    def build(self, input_shape):
        # Adapted from TF. See: https://git.io/J89L8 and https://git.io/J89q0
        input_channel = int(input_shape[-1])
        kernel_size = (1, 1)
        kernel_shape = kernel_size + (input_channel, self.k)
        self.kernel_ones = tf.ones(kernel_shape)
        self.kernel_minus_ones = tf.constant(-1.0) * self.kernel_ones

    def call(self, inputs):
        # Compute the activation maps
        h = self.activation_map(inputs)
        # Perform the global average pooling
        h = tf.keras.backend.mean(h, axis=[1, 2])

        return h

    def activation_map(self, inputs):
        # Returns the activation map
        # See: [2, p.8, figure 4]
        x = inputs
        # Compute x - muk
        x = tf.keras.backend.conv2d(
            x,
            self.kernel_ones,
            padding="same",
            data_format="channels_last",
        )
        x = x - self.muk
        # Compute |x - muk|
        x = tf.math.abs(x)
        # Compute wk - |x - muk|
        x = tf.keras.backend.conv2d(
            x,
            self.kernel_minus_ones,
            padding="same",
            data_format="channels_last",
        )
        x = x + self.wk
        # Compute 1.01 ^ (wk - |x - muk|)
        x = tf.math.pow(tf.constant(1.01), x)
        # Compute the "ReLU at one". See [1, Eq. 2].
        x = tf.keras.backend.relu(x, threshold=1.0)
        return x

    def get_config(self):
        return dict(k=self.k, start=self.start, stop=self.stop)


class JointHistogramLayer(HistogramLayer):
    """Compute a differentiable joint histogram."""

    # Implements the joint histogram from [2] but using the activation maps from [1].

    def build(self, input_shape):
        input_shape1 = tf.TensorShape(input_shape[0])
        input_shape2 = tf.TensorShape(input_shape[1])
        if not input_shape1.is_compatible_with(input_shape2):
            raise ValueError("Shape mismatch")
        super(JointHistogramLayer, self).build(input_shape1)
        # See: https://git.io/J4CJ2
        self.map_flattened_shape = tf.TensorShape(
            (
                input_shape1[0],
                (input_shape1[1] * input_shape1[2]),
                self.k,
            )
        )

    def call(self, inputs):
        # Compute the activation maps
        # Shape: (batch_size, height, width, k)
        histogram_map1 = self.activation_map(inputs[0])
        histogram_map2 = self.activation_map(inputs[1])
        # Flatten the histograms
        # Let n = height * width
        # Shape: (batch_size, n, k)
        histogram_map1 = tf.reshape(histogram_map1, self.map_flattened_shape)
        histogram_map2 = tf.reshape(histogram_map2, self.map_flattened_shape)
        # Compute the joint histogram
        # See [2, p.8, eq. 13]
        # Input shape: (batch_size, k, n) x (batch_size, n, k)
        # Output shape: (batch_size, k, k)
        joint_histogram = tf.linalg.matmul(
            histogram_map1, histogram_map2, transpose_a=True
        )
        # Normalize the joint histogram
        factor = tf.constant(1.0 / histogram_map1.shape[1])
        joint_histogram = factor * joint_histogram

        return joint_histogram
