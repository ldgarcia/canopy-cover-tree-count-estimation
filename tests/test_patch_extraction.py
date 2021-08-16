# References:
# [1] Dumoulin, V. & Visin, F. (2018).
#     A guide to convolution arithmetic for deep learning.
#     arXiv: https://arxiv.org/abs/1603.07285
import unittest

import numpy as np

from dlc.tools.patches import compute_patch_slices


class TestPatchExtraction(unittest.TestCase):
    def test_simple_valid_convolution(self):
        """Using example from Fig. 1.1. from [1]."""
        k = np.array(
            [
                [0, 1, 2],
                [2, 2, 0],
                [0, 1, 2],
            ],
            dtype=float,
        )
        img = np.array(
            [
                [3, 3, 2, 1, 0],
                [0, 0, 1, 3, 1],
                [3, 1, 2, 2, 3],
                [2, 0, 0, 2, 2],
                [2, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        patch_specs = compute_patch_slices(
            img.shape,
            k.shape,
            (1, 1),
            (0, 0),
        )
        self.assertEqual(9, len(patch_specs))
        expected_acc = 116.0
        acc = 0.0
        for patch_spec in patch_specs:
            x_slice = slice(*patch_spec["x_slice"])
            y_slice = slice(*patch_spec["y_slice"])
            patch = img[y_slice, x_slice]
            acc += np.sum(patch * k)
        self.assertEqual(expected_acc, acc)

    def test_simple_padding(self):
        k = np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            dtype=float,
        )
        img = np.array(
            [
                [3, 3, 2, 1, 0],
                [0, 0, 1, 3, 1],
                [3, 1, 2, 2, 3],
                [2, 0, 0, 2, 2],
                [2, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        patch_specs = compute_patch_slices(
            img.shape,
            k.shape,
            (1, 1),
            (1, 1),
        )
        self.assertEqual(25, len(patch_specs))
        expected_acc = 225.0
        acc = 0.0
        for patch_spec in patch_specs:
            x_slice = slice(*patch_spec["x_slice"])
            y_slice = slice(*patch_spec["y_slice"])
            tmp = img[y_slice, x_slice]
            patch = np.zeros_like(k)
            patch[0 : tmp.shape[0], 0 : tmp.shape[1]] = tmp
            acc += np.sum(patch * k)
        self.assertEqual(expected_acc, acc)

    def test_simple_padding_striding(self):
        k = np.array(
            [
                [1, 1],
                [1, 1],
            ],
            dtype=float,
        )
        img = np.array(
            [
                [3, 3, 2, 1, 0],
                [0, 0, 1, 3, 1],
                [3, 1, 2, 2, 3],
                [2, 0, 0, 2, 2],
                [2, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        patch_specs = compute_patch_slices(
            img.shape,
            k.shape,
            (2, 2),
            (1, 1),
        )
        self.assertEqual(9, len(patch_specs))
        expected_acc = 34.0
        acc = 0.0
        for patch_spec in patch_specs:
            x_slice = slice(*patch_spec["x_slice"])
            y_slice = slice(*patch_spec["y_slice"])
            # Note that we don't take care of placing the padding
            # "correctly", we just align the patch to a zero-image
            # of the expected patch size using its origin (0,0).
            # We could align it correctly using the information
            # of x and y being 0 or otherwise.
            # This is the reason for this test case using a ones-kernel.
            tmp = img[y_slice, x_slice]
            patch = np.zeros_like(k)
            patch[0 : tmp.shape[0], 0 : tmp.shape[1]] = tmp[
                0 : tmp.shape[0], 0 : tmp.shape[1]
            ]
            acc += np.sum(patch * k)
        self.assertEqual(expected_acc, acc)
