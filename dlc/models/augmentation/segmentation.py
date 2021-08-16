import albumentations

from dlc.models.augmentation.base import AugmentationTransform

__all__ = ["SegmentationAugTransform0"]


class SegmentationAugTransform0(AugmentationTransform):
    """Augmentation transform that implements that of [1]."""

    def __init__(self):
        # Transformations A (features and annotations).
        self._transform_a = albumentations.Compose(
            [
                albumentations.crops.transforms.CropAndPad(
                    percent=(0, 0.1),
                    p=0.5,
                ),
                albumentations.imgaug.transforms.IAAPiecewiseAffine(
                    0.05,
                    p=0.3,
                ),
                albumentations.imgaug.transforms.Perspective(
                    0.01,
                    p=0.1,
                ),
            ]
        )
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
        augmented_a = self._transform_a(
            image=features,
            mask=annotations,
        )
        features = augmented_a["image"]
        augmented_b = self._transform_b(image=features)
        features = augmented_b["image"]
        annotations = augmented_a["mask"]
        return features, annotations
