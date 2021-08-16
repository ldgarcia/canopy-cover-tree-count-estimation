"""Augmentation pipelines."""
#
# Implements various augmentation pipelines.
#
# References:
# [1]
#   Brandt, M., Tucker, C., Kariryaa, A., et al. (2020).
#   An unexpectedly large count of trees in the West African Sahara and Sahel.
#   Nature, 587(7832), 78–82.
#   DOI:  https://doi.org/10.1038/s41586-020-2824-5
#   Code: https://git.io/Jsivl
# [2]
#   Yuhong Li and Xiaofan Zhang and Deming Chen (2018).
#   CSRNet: Dilated Convolutional Neural Networks for Understanding
#   the Highly Congested Scenes. CoRR, abs/1802.10062.
#   arXiv: http://arxiv.org/abs/1802.10062
#
# See also:
# - https://albumentations.ai/docs/
# - https://imgaug.readthedocs.io/en/latest/
# - https://albumentations.ai/docs/examples/tensorflow-example/
from .density import DensityAugTransform0
from .segmentation import SegmentationAugTransform0


__all__ = ["DensityAugTransform0", "SegmentationAugTransform0"]
