import unittest

import numpy as np
import tensorflow as tf

from dlc.losses.emd import SquaredEarthMoversDistance
from dlc.losses.histogram import MSEPlusHistogram
from dlc.losses.histogram import TanimotoPlusHistogram
from dlc.losses.mse import MeanSquaredErrorV2
from dlc.losses.tanimoto import Tanimoto
from dlc.losses.tanimoto import tanimoto
from dlc.losses.tanimoto import tanimoto_index
from dlc.losses.tversky import Tversky
from dlc.losses.wrapper import LossFunctionWrapper


class TestWrapper(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        @tf.function
        def bce(y_true, y_pred, *, weights):
            result = tf.keras.backend.binary_crossentropy(y_true, y_pred)
            return result * weights

        class WrappedBCE(LossFunctionWrapper):
            def __init__(
                self,
                weights_index=None,
                **kwargs,
            ):
                super(WrappedBCE, self).__init__(
                    bce,
                    weights_index=weights_index,
                    **kwargs,
                )

        self.wrapped_bce = WrappedBCE

    def test_wrapper_without_weights(self):
        true = tf.round(tf.random.uniform(shape=(8, 256, 256, 1)))
        pred = tf.random.uniform(shape=(8, 256, 256, 1))
        wbce = self.wrapped_bce()
        result = wbce(true, pred)
        tf_wbce = tf.keras.losses.BinaryCrossentropy()
        target_result = tf_wbce(true, pred)
        self.assertAlmostEqual(target_result.numpy(), result.numpy(), 6)

    def test_wrapper_with_weights(self):
        true = np.round(np.random.uniform(size=(8, 256, 256, 2)))
        # For testing purposes, we use a binary mask.
        true[..., 1] = np.random.uniform(size=(8, 256, 256))
        true = true.astype("float32")
        pred = np.random.uniform(size=(8, 256, 256, 1))
        pred = pred.astype("float32")
        true = tf.constant(true, dtype=tf.float32)
        pred = tf.constant(pred, dtype=tf.float32)
        # Our wrapper
        wbce1 = self.wrapped_bce(weights_index=1)
        result1 = wbce1(true, pred).numpy()
        # Manually compute the reduced weighted-result.
        result2 = tf.keras.backend.binary_crossentropy(
            tf.expand_dims(true[..., 0], -1), pred
        )
        result2 = result2 * tf.expand_dims(true[..., 1], -1)
        result2 = tf.reduce_mean(result2, axis=(1, 2))
        result2 = tf.reduce_mean(result2)

        result2 = result2.numpy()
        self.assertAlmostEqual(result2, result1, 6)


class TestTanimotoLoss(unittest.TestCase):
    def test_tanimoto_index_1x1(self):
        # We create a batch of 1x1 images with shape: (3, 1, 1, 1)
        ytrue = np.expand_dims(np.asarray([[[1.0]], [[1.0]], [[0.5]]]), axis=-1)
        ypred = np.expand_dims(np.asarray([[[1.0]], [[0.0]], [[0.5]]]), axis=-1)
        output = tanimoto_index(ytrue, ypred, weights=None)
        expected = np.expand_dims(np.asarray([1.0, 0.0, 1.0]), axis=-1)
        np.testing.assert_allclose(
            expected,
            output,
            atol=1e-7,
        )

    def test_tanimoto_loss_fn_1x1(self):
        ytrue = np.expand_dims(np.asarray([[[1.0]], [[1.0]], [[0.5]]]), axis=-1)
        ypred = np.expand_dims(np.asarray([[[1.0]], [[0.0]], [[0.5]]]), axis=-1)
        output = tanimoto(ytrue, ypred)
        expected = np.expand_dims(np.asarray([0.0, 1.0, 0.0]), axis=-1)
        np.testing.assert_allclose(
            expected,
            output,
            atol=1e-7,
        )

    def test_tanimoto_loss_fn_3x1(self):
        # We create a batch of 3x1 images with shape: (1, 3, 1, 1)
        ytrue = np.expand_dims(np.asarray([[[1.0, 0.0, 1.0]]]), axis=-1)
        ypred = np.expand_dims(np.asarray([[[1.0, 0.0, 1.0]]]), axis=-1)
        output = tanimoto(ytrue, ypred)
        expected = np.expand_dims(np.asarray([0.0]), axis=-1)
        np.testing.assert_allclose(
            expected,
            output,
            atol=1e-7,
        )

    def test_tanimoto_loss_class_1x1(self):
        ytrue = np.expand_dims(np.asarray([[[1.0]], [[1.0]], [[0.5]]]), axis=-1)
        ypred = np.expand_dims(np.asarray([[[1.0]], [[0.0]], [[0.5]]]), axis=-1)
        ytrue = tf.constant(ytrue, dtype=tf.float32)
        ypred = tf.constant(ypred, dtype=tf.float32)
        tl = Tanimoto()
        output = tl(ytrue, ypred).numpy()
        expected = tf.reduce_mean(tanimoto(ytrue, ypred))
        np.testing.assert_allclose(
            expected,
            output,
            atol=1e-7,
        )


class TestMeanSquaredErrorV2(unittest.TestCase):
    def test_mse_w(self):
        ytrue = tf.random.normal(shape=(3, 10, 10, 1))
        ypred = tf.random.normal(shape=(3, 10, 10, 1))
        weights = tf.ones_like(ypred)
        mse = tf.losses.MeanSquaredError()
        mse_v2 = MeanSquaredErrorV2(weights_index=1)

        loss = mse(ytrue, ypred)
        loss_v2 = mse_v2(tf.concat([ytrue, weights], axis=-1), ypred)
        np.testing.assert_allclose(loss.numpy(), loss_v2.numpy(), rtol=1e-4)

    def test_mse_nw(self):
        ytrue = tf.round(tf.random.uniform(shape=(3, 10, 10, 1)))
        ypred = tf.random.normal(shape=(3, 10, 10, 1))
        weights = tf.math.round(tf.random.uniform(shape=(3, 10, 10, 1)))
        mse = tf.losses.MeanSquaredError()
        mse_v2 = MeanSquaredErrorV2()

        loss = mse(ytrue, ypred)
        loss_v2 = mse_v2(tf.concat([ytrue, weights], axis=-1), ypred)
        np.testing.assert_allclose(loss.numpy(), loss_v2.numpy(), rtol=1e-4)


class TestSquaredEarthMoversDistance(unittest.TestCase):
    def setUp(self):
        self.true = tf.constant([[0.0, 0.0, 0.0], [0.1, 0.2, 0.4]])
        self.pred = tf.constant([[0.1, 0.0, 0.0], [0.1, 0.2, 0.4]])

    def test_reduction(self):
        loss = SquaredEarthMoversDistance()
        value = loss(self.true, self.pred)
        self.assertAlmostEqual(value.numpy(), 0.015)

    def test_no_reduction(self):
        loss = SquaredEarthMoversDistance(
            reduction=tf.keras.losses.Reduction.NONE,
        )
        value = loss(self.true, self.pred)
        np.testing.assert_allclose(value.numpy(), np.asarray([0.03, 0.0]))


class TestTversky(unittest.TestCase):
    def setUp(self):
        n = 1000
        self.true = tf.random.uniform((n, 256, 256, 1), maxval=1.0)
        self.true = tf.round(self.true)
        self.pred = tf.random.uniform((n, 256, 256, 1), maxval=1.0)
        weights = tf.random.uniform((n, 256, 256, 1), maxval=1.0)
        weights = tf.round(weights)
        weights = tf.where(weights > 0.0, 10.0, 1.0)
        self.true_w = tf.concat([self.true, weights], axis=-1)
        self.tv = Tversky(alpha=0.6)
        self.tvw = Tversky(alpha=0.6, weights_index=1)

    def test_loss(self):
        loss = self.tv(self.true, self.pred)
        self.assertTrue(not (loss < 0.0))

    def test_loss_w(self):
        loss = self.tvw(self.true_w, self.pred)
        self.assertTrue(not (loss < 0.0))

    def test_ones(self):
        loss = self.tv(tf.ones((8, 256, 256, 1)), tf.ones((8, 256, 256, 1)))
        self.assertAlmostEqual(loss, 0.0)

    def test_zeros(self):
        loss = self.tv(tf.zeros((8, 256, 256, 1)), tf.zeros((8, 256, 256, 1)))
        self.assertAlmostEqual(loss, 0.0)


class TestMSEPlusHistogram(unittest.TestCase):
    def test_fn(self):
        img = tf.random.normal((2, 256, 256, 1))
        loss = MSEPlusHistogram(
            k=256,
            start=-1.0,
            stop=1.0 + 1e-3,
        )
        output = loss(img, img)
        self.assertAlmostEqual(0.0, output.numpy())


class TestTanimotoPlusHistogram(unittest.TestCase):
    def test_fn(self):
        img = tf.random.uniform((2, 256, 256, 1))
        loss = TanimotoPlusHistogram(
            k=256,
            start=0.0,
            stop=1.0 + 1e-3,
        )
        output = loss(img, img)
        self.assertAlmostEqual(0.0, output.numpy())
