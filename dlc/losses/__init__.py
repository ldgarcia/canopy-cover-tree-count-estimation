"""Loss functions."""
# Implementation note:
#   The loss functions are tailored for 2-D data (e.g., 2-D images).
# References:
# [1]
#   Brandt, M., Tucker, C., Kariryaa, A., et al. (2020).
#   An unexpectedly large count of trees in the West African Sahara and Sahel.
#   Nature, 587(7832), 78–82.
#   DOI:  https://doi.org/10.1038/s41586-020-2824-5
#   Code: https://git.io/Jsivl
# [2]
#   Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, & Piotr Dollár. (2018).
#   Focal Loss for Dense Object Detection.
#   arXiv: https://arxiv.org/pdf/1708.02002.pdf
#   Code: https://git.io/JWMQy
# [3]
#   Abraham, N., & Khan, N. M. (2019).
#   A novel focal tversky loss function with improved attention u-net for
#   lesion segmentation.
#   In 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019)
#   (pp. 683-687). IEEE.
#   arXiv: https://arxiv.org/abs/1810.07842
# [4]
#   Diakogiannis, F. I., Waldner, F., Caccetta, P., & Wu, C. (2020).
#   ResUNet-a: A deep learning framework for semantic segmentation of remotely
#   sensed data. ISPRS Journal of Photogrammetry and Remote Sensing, 162, 94-114.
#   arXiv: https://arxiv.org/abs/1904.00592
# [5]
#   Yeung, M., Sala, E., Schönlieb, C.B., & Rundo., L. (2021).
#   Unified Focal loss: Generalising Dice and cross entropy-based losses to
#   handle class imbalanced medical image segmentation.
#   arXiv: https://arxiv.org/abs/2102.04525
#   Code: https://git.io/J81SJ
# [6]
#   Salehi, A. (2017). Tversky Loss Function for Image Segmentation Using 3D
#   Fully Convolutional Deep Networks.
#   In Machine Learning in Medical Imaging (pp. 379–387).
#   Springer International Publishing.
#   arXiv: https://arxiv.org/abs/1706.05721
#
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import Huber
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import MeanSquaredLogarithmicError

from dlc.losses.emd import SquaredEarthMoversDistance
from dlc.losses.focal import AssymetricFocal
from dlc.losses.histogram import MSEPlusHistogram
from dlc.losses.histogram import TanimotoPlusHistogram
from dlc.losses.mse import MeanSquaredErrorV2
from dlc.losses.tanimoto import Tanimoto
from dlc.losses.tversky import AssymetricFocalTversky
from dlc.losses.tversky import Tversky
from dlc.losses.unified import UnifiedFocal

# from dlc.losses.histogram import MSEPlusHistogram

__all__ = [
    # TF implementations:
    "BinaryCrossentropy",
    "Huber",
    "LogCosh",
    "MeanAbsoluteError",
    "MeanSquaredError",
    "MeanSquaredLogarithmicError",
    # Custom implementations:
    "AssymetricFocal",
    "AssymetricFocalTversky",
    "MeanSquaredErrorV2",
    "Tanimoto",
    "Tversky",
    "UnifiedFocal",
    "SquaredEarthMoversDistance",
    "MSEPlusHistogram",
    "TanimotoPlusHistogram",
]
