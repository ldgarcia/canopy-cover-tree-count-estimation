"""Various U-Net architecture blocks."""
#
# Implements U-Net [1] building blocks for solving image segmentation
# tasks. Based on the implementations from [2] and [3].
#
# References:
# [1]
#   Ronneberger, O., Fischer, P., & Brox, T. (2015).
#   U-Net: Convolutional Networks for Biomedical Image Segmentation.
#   arXiv: https://arxiv.org/abs/1505.04597v1
#   Code: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-release-2015-10-02.tar.gz
# [2]
#   Brandt, M., Tucker, C., Kariryaa, A., et al. (2020).
#   An unexpectedly large count of trees in the West African Sahara and Sahel.
#   Nature, 587(7832), 78–82.
#   DOI:  https://doi.org/10.1038/s41586-020-2824-5
#   Code: https://git.io/Jsivl
# [3]
#   Perslev, M., Dam, E., Pai, A., & Igel, C. (2019).
#   One network to segment them all: A general, lightweight system for accurate
#   3d medical image segmentation.
#   In International Conference on Medical Image Computing and
#   Computer-Assisted Intervention (pp. 30–38).
#   DOI:  https://doi.org/10.1007/978-3-030-32245-8_4
#   Code: https://git.io/JsivB
# [4]
#   Odena, A., Dumoulin, V., & Olah, C. (2016).
#   Deconvolution and Checkerboard Artifacts. Distill.
#   DOI: http://doi.org/10.23915/distill.00003
# [5]
#   Oktay, O., Schlemper, J., Le Folgoc, et al. (2018).
#   Attention U-Net: Learning Where to Look for the Pancreas.
#   arXiv: https://arxiv.org/abs/1804.03999
#   Code:  https://git.io/Jsi5H
# [6]
#   Diakogiannis, F. I., Waldner, F., Caccetta, P., & Wu, C. (2020).
#   ResUNet-a: A deep learning framework for semantic segmentation of
#   remotely sensed data. ISPRS Journal of Photogrammetry and Remote Sensing,
#   162, 94-114.
# [7]
#   Shuhang Wang, Szu-Yeu Hu, Eugene Cheah, Xiaohong Wang, Jingchao Wang,
#   Lei Chen, Masoud Baikpour, Arinc Ozturk, Qian Li, Shinn-Huey Chou,
#   Constance D. Lehman, Viksit Kumar, & Anthony Samir. (2020).
#   U-Net Using Stacked Dilated Convolutions for Medical Image Segmentation.
#   arXiv: https://arxiv.org/abs/2004.03466
from dlc.models.unet.model import create_model
from dlc.models.unet.model import load_model
from dlc.models.unet.settings import UNetSettings as Settings

__all__ = ["create_model", "load_model", "Settings"]
