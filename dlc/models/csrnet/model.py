"""A CSR-Net (B-variant) implementation."""
# [1]
#   Yuhong Li and Xiaofan Zhang and Deming Chen (2018).
#   CSRNet: Dilated Convolutional Neural Networks for Understanding
#   the Highly Congested Scenes. CoRR, abs/1802.10062.
#   arXiv: http://arxiv.org/abs/1802.10062
# [2]
#   Odena, A., Dumoulin, V., & Olah, C. (2016).
#   Deconvolution and Checkerboard Artifacts. Distill.
#   DOI: http://doi.org/10.23915/distill.00003
from typing import Optional
from typing import Tuple

import tensorflow as tf

import dlc.models.base.resizing
import dlc.models.vgg16.model
from dlc.initializers import kumar_sigmoid_initialiazer


__all__ = [
    "create_model",
    "load_model",
]


def load_model(input_shape, batch_size, weights_file, **kwargs):
    model = create_model(
        input_shape=input_shape,
        batch_size=batch_size,
        **kwargs,
    )
    model.load_weights(weights_file)
    return model


def create_model(
    *,
    name: str,
    input_shape: Tuple[Optional[int], ...],
    output_channels: int,
    batch_size: int,
    output_name: str,
    output_activation: str,
    conv_activation: str,
    conv_regularizer: Optional[str],
    padding: str,
    dilation_rate: Tuple[int, int],
    interpolation: str,
    seed: Optional[int],
    use_imagenet_weights: bool,
    freeze_vgg16_encoders: bool,
    input_dtype: str,
    **_kwargs,  # catch-all dummy
):
    """Create an instance of the CSRNet B model."""
    assert output_activation in ["linear", "sigmoid", "tanh"]
    assert conv_activation in ["elu", "relu"]

    he_normal_init = tf.keras.initializers.HeNormal(seed=seed)
    glorot_uniform_init = tf.keras.initializers.GlorotUniform(seed=seed)
    kumar_sigmoid_init = kumar_sigmoid_initialiazer(seed=seed)

    input = tf.keras.Input(
        shape=input_shape,
        batch_size=batch_size,
        dtype=input_dtype,
        name="input",
    )
    x = input

    x, _residuals = dlc.models.vgg16.model.create_encoders(
        x,
        padding=padding,
        conv_activation=conv_activation,
        conv_initializer=he_normal_init,
        conv_regularizer=conv_regularizer,
        trainable=(not freeze_vgg16_encoders),
    )

    # Back-end (B variation) but with a different upsampling method.
    x = tf.keras.layers.Conv2D(
        512,
        (3, 3),
        padding=padding,
        dilation_rate=dilation_rate,
        activation=conv_activation,
        name="backend_conv_1",
        kernel_initializer=he_normal_init,
    )(x)
    x = tf.keras.layers.Conv2D(
        512,
        (3, 3),
        padding=padding,
        dilation_rate=dilation_rate,
        activation=conv_activation,
        name="backend_conv_2",
        kernel_initializer=he_normal_init,
    )(x)
    x = tf.keras.layers.Conv2D(
        512,
        (3, 3),
        padding=padding,
        dilation_rate=dilation_rate,
        activation=conv_activation,
        name="backend_conv_3",
        kernel_initializer=he_normal_init,
    )(x)
    x = tf.keras.layers.Conv2D(
        256,
        (3, 3),
        padding=padding,
        dilation_rate=dilation_rate,
        activation=conv_activation,
        name="backend_conv_4",
        kernel_initializer=he_normal_init,
    )(x)
    x = tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding=padding,
        dilation_rate=dilation_rate,
        activation=conv_activation,
        name="backend_conv_5",
        kernel_initializer=he_normal_init,
    )(x)
    x = tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        dilation_rate=dilation_rate,
        activation=conv_activation,
        name="backend_conv_6",
        kernel_initializer=he_normal_init,
    )(x)

    # Output resizing
    x = dlc.models.base.resizing.create_resizing_block(
        x,
        input_shape,
        64,
        "out_up",
        conv_initializer=glorot_uniform_init,
        interpolation=interpolation,
        use_batch_norm=False,
    )

    # Output layers
    fuse_init = glorot_uniform_init
    if output_activation == "sigmoid":
        fuse_init = kumar_sigmoid_init

    x = tf.keras.layers.Conv2D(
        output_channels,
        (1, 1),
        padding="same",
        kernel_initializer=fuse_init,
        kernel_regularizer=conv_regularizer,
        name="out_fusing_conv",
    )(x)

    # Output activation
    x = tf.keras.layers.Activation(
        output_activation,
        name=output_name,
    )(x)

    model = tf.keras.models.Model(inputs=input, outputs=x, name=name)
    if use_imagenet_weights:
        assert input_shape[2] == 3
        dlc.models.vgg16.model.set_encoder_rgb_weights(model)
    return model
