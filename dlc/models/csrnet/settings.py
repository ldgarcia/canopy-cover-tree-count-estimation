from typing import Optional
from typing import Tuple

import dlc.models.base.settings


class CSRNetSettings(dlc.models.base.settings.Settings):
    name: str = "csrnet"
    output_activation: str = "linear"
    conv_activation: str = "elu"
    conv_regularizer: Optional[str] = None
    padding: str = "same"
    dilation_rate: Tuple[int, int] = (2, 2)
    interpolation: str = "bilinear"
    use_imagenet_weights: bool = False
    freeze_vgg16_encoders: bool = False
