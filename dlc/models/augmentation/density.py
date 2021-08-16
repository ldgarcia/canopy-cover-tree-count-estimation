import albumentations

from dlc.models.augmentation.base import (
    AugmentationTransform,
)

__all__ = ["DensityAugTransform0"]


class DensityAugTransform0(AugmentationTransform):
    """Augmentation transform that implements that of [1]."""

    def __init__(self):
        # Transformations B (features only)
        self._transform_b = albumentations.Compose(
            [
                albumentations.RandomBrightnessContrast(
                    brightness_limit=0,
                    contrast_limit=(0.3, 1.2),
                    p=0.3,
                ),
                albumentations.GaussNoise(
                    var_limit=0.1,
                    p=0.8,
                ),
            ]
        )

    def _augment(self, features, annotations):
        # Augmentation
        features = self._transform_b(image=features)["image"]
        return features, annotations
