import unittest

import numpy as np

from dlc.tools.scalers import normalize_image_np


class TestScalers(unittest.TestCase):
    def setUp(self):
        self.depth = 3
        self.img = (
            np.random.normal(
                loc=4234,
                scale=3.5,
                size=((1000 ** 2) * self.depth),
            )
            .astype("float32")
            .reshape([1000, 1000, self.depth])
        )

    def test_normalize_dtype(self):
        img_scaled = normalize_image_np(self.img)
        self.assertEqual(img_scaled.dtype, self.img.dtype)

    def test_normalize(self):
        img_scaled = normalize_image_np(self.img)
        min_val = np.min(img_scaled)
        max_val = np.max(img_scaled)
        self.assertTrue(not (min_val < 0.0))
        self.assertTrue(not (max_val > 1.0))
