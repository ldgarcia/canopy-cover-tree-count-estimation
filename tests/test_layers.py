import unittest

import numpy as np
import tensorflow as tf

from dlc.layers.histogram import histogram
from dlc.layers.histogram import HistogramLayer
from dlc.layers.histogram import JointHistogramLayer


def create_histogram_model(k, start, stop, input_shape, batch_size):
    input = tf.keras.Input(input_shape, batch_size, name="input")
    output = HistogramLayer(k, start, stop, name="histogram")(input)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model


def create_joint_histogram_model(k, start, stop, input_shape, batch_size):
    input1 = tf.keras.Input(input_shape, batch_size, name="input1")
    input2 = tf.keras.Input(input_shape, batch_size, name="input2")
    output = JointHistogramLayer(k, start, stop, name="joint_histogram")(
        [input1, input2]
    )
    model = tf.keras.Model(inputs=[input1, input2], outputs=output)
    return model


class TestHistogram(unittest.TestCase):
    def setUp(self):
        self.k = 3
        self.start = -1.0
        self.stop = 1.0 + 1e-3
        # We create a batch of three 2x2 images
        # All values should be in one bin.
        x1 = tf.expand_dims(tf.expand_dims([[0.2, 0.2], [0.2, 0.2]], -1), 0)
        # Values are distributed 0.25/0.75 in two bins.
        x2 = tf.expand_dims(tf.expand_dims([[0.2, 0.4], [0.2, 0.2]], -1), 0)
        # All values are out of range.
        x3 = tf.expand_dims(tf.expand_dims([[-4.0, -4.0], [4.0, 4.0]], -1), 0)
        self.xs = tf.concat([x1, x2, x3], axis=0)
        self.hs_true = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.75, 0.25],
                [0.0, 0.0, 0.0],
            ],
            dtype="float32",
        )

    def test_layer_stand_alone(self):
        l = HistogramLayer(self.k, self.start, self.stop)
        hs = l(self.xs).numpy()
        np.testing.assert_allclose(hs, self.hs_true, rtol=1e-2)

    def test_layer_in_model(self):
        input_shape = self.xs.shape[1:]
        batch_size = self.xs.shape[0]
        model = create_histogram_model(
            self.k,
            self.start,
            self.stop,
            input_shape,
            batch_size,
        )
        hs = model.predict(self.xs)
        np.testing.assert_allclose(hs, self.hs_true, rtol=1e-2)

    def test_function(self):
        hs = histogram(self.xs, k=self.k, start=self.start, stop=self.stop)
        np.testing.assert_allclose(hs, self.hs_true, rtol=1e-2)


class TestJointHistogram(unittest.TestCase):
    def setUp(self):
        self.k = 20
        self.start = -1.0
        self.stop = 1.0 + 1e-3
        self.edges = np.linspace(
            self.start,
            self.stop,
            endpoint=True,
            num=self.k + 1,
        )
        self.img1 = tf.random.normal((1, 20, 20, 1))
        self.img2 = tf.random.normal((1, 20, 20, 1))

    def reference(self, img1, img2):
        ref, _, _ = np.histogram2d(
            img1.numpy().flatten(),
            img2.numpy().flatten(),
            bins=(self.edges, self.edges),
        )
        ref /= img1.shape[1] * img1.shape[2]
        return ref

    def test_stand_alone(self):
        l = JointHistogramLayer(self.k, self.start, self.stop)
        output = l([self.img1, self.img2])
        expected = self.reference(self.img1, self.img2)
        self.assertEqual(output.shape[1:], expected.shape)
        np.testing.assert_allclose(expected, output[0, ...].numpy(), atol=1e-4)

    def test_in_model(self):
        model = create_joint_histogram_model(
            self.k,
            self.start,
            self.stop,
            self.img1.shape[1:],
            self.img1.shape[0],
        )
        output = model.predict([self.img1, self.img2])
        expected = self.reference(self.img1, self.img2)
        self.assertEqual(output.shape[1:], expected.shape)
        np.testing.assert_allclose(expected, output[0, ...], atol=1e-4)
