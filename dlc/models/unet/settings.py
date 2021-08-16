from typing import Optional

from dlc.models.base.settings import Settings


class UNetSettings(Settings):
    name: str = "unet"
    output_activation: str = "sigmoid"
    hidden_activation: str = "elu"
    # Glorot et al. [2011] point that encoder-decoder architectures
    # can benefit from sparse representations and recommend using the
    # L1 for activity regularization to both, prevent numerical issues
    # related to the ReLU's unbounded behaviour and get sparser activations.
    relu_activity_l1_regularizer: Optional[float] = None
    padding: str = "same"
    depth: int = 4
    layer_count: int = 64
    output_resize_interpolation: Optional[str] = None
    encoder_unit: str = "vanilla"
    encoder_pooling: str = "max"
    encoder_pooling_batch_norm: bool = False  # see EddyNet
    decoder_unit: str = "vanilla"
    decoder_upsampling: str = "odena_2016"
    decoder_upsampling_use_batch_norm: bool = False
    decoder_use_attention_gate: bool = False
    decoder_attention_relu_variant: str = "relu"
    decoder_attention_use_batch_norm: bool = False
    decoder_concat_use_batch_norm: bool = False  # see EddyNet
    # The attention gate paper recommends using deep supervision for
    # better training of the attention modules.
    deeply_supervised: bool = False
    # Optional for second output:
    second_output_name: Optional[str] = None
    second_output_activation: Optional[str] = None
    # Optionally, adds a preciding basic unit before the fusing.
    second_output_preceding_unit: Optional[str] = None
