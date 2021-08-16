import tensorflow as tf

from dlc.relu import get_relu_variant_layer

__all__ = ["create_upsampling_block", "UPSAMPLING_VARIANTS"]


UPSAMPLING_VARIANTS = [
    "odena_2016",
    "vanilla",
    "transpose_conv",
]


def _vanilla_up_conv(
    x,
    *,
    num_filters,
    block_name,
    size,
    activation,
    init_container,
    regularizer,
    relu_activity_l1_regularizer,
):
    """Create an up-convolution block as used in [1]."""
    x = tf.keras.layers.UpSampling2D(
        size,
        interpolation="bilinear",
        name=f"{block_name}_bilinear",
    )(x)
    crop_shape = x.shape
    activity_regularizer = None
    if relu_activity_l1_regularizer is not None:
        l1 = relu_activity_l1_regularizer
        activity_regularizer = tf.keras.regularizers.L1(l1=l1)
    x = tf.keras.layers.Conv2D(
        num_filters,
        (3, 3),
        name=f"{block_name}_conv",
        padding="same",
        activation=activation,
        kernel_initializer=init_container.get_kernel_init("relu"),
        bias_initializer=init_container.get_bias_init("relu"),
        kernel_regularizer=regularizer,
        activity_regularizer=activity_regularizer,
    )(x)
    return x, crop_shape


def _odena_up_conv(
    x,
    *,
    num_filters,
    block_name,
    size,
    activation,
    init_container,
    regularizer,
    relu_activity_l1_regularizer,
):
    """Create an up-convolution block as proposed by [4] and recommended \
        by [3] for the U-Net architecture."""
    x = tf.keras.layers.UpSampling2D(
        size=size,
        interpolation="nearest",
        name=f"{block_name}_nearest",
    )(x)
    crop_shape = x.shape
    activity_regularizer = None
    if relu_activity_l1_regularizer is not None:
        l1 = relu_activity_l1_regularizer
        activity_regularizer = tf.keras.regularizers.L1(l1=l1)
    x = tf.keras.layers.Conv2D(
        num_filters,
        (3, 3),
        name=f"{block_name}_conv",
        padding="same",
        activation=activation,
        kernel_initializer=init_container.get_kernel_init("relu"),
        bias_initializer=init_container.get_bias_init("relu"),
        kernel_regularizer=regularizer,
        activity_regularizer=activity_regularizer,
    )(x)
    return x, crop_shape


def _transpose_conv(
    x,
    *,
    num_filters,
    block_name,
    size,
    activation,
    init_container,
    regularizer,
    relu_activity_l1_regularizer,
):
    activity_regularizer = None
    if relu_activity_l1_regularizer is not None:
        l1 = relu_activity_l1_regularizer
        activity_regularizer = tf.keras.regularizers.L1(l1=l1)
    x = tf.keras.layers.Conv2DTranspose(
        num_filters,
        (3, 3),
        strides=size,
        name=f"{block_name}_conv_trans",
        padding="same",
        activation=activation,
        kernel_initializer=init_container.get_kernel_init("relu"),
        bias_initializer=init_container.get_bias_init("relu"),
        kernel_regularizer=regularizer,
        activity_regularizer=activity_regularizer,
    )(x)
    return x, x.shape


def create_upsampling_block(
    x,
    *,
    variant="odena_2016",  # recommended by [3]
    num_filters,
    block_name,
    init_container,
    size=(2, 2),
    activation="relu",
    use_batch_norm=False,
    regularizer=None,
    relu_activity_l1_regularizer=None,
):
    """Create an instance of an up-sampling block."""
    crop_shape = None
    if variant == "odena_2016":
        x, crop_shape = _odena_up_conv(
            x,
            num_filters=num_filters,
            block_name=block_name,
            size=size,
            activation=activation,
            init_container=init_container,
            regularizer=regularizer,
            relu_activity_l1_regularizer=relu_activity_l1_regularizer,
        )
    elif variant == "vanilla":
        x, crop_shape = _vanilla_up_conv(
            x,
            num_filters=num_filters,
            block_name=block_name,
            size=size,
            activation=activation,
            init_container=init_container,
            regularizer=regularizer,
            relu_activity_l1_regularizer=relu_activity_l1_regularizer,
        )
    elif variant == "transpose_conv":
        x, crop_shape = _transpose_conv(
            x,
            num_filters=num_filters,
            block_name=block_name,
            size=size,
            activation=activation,
            init_container=init_container,
            regularizer=regularizer,
            relu_activity_l1_regularizer=relu_activity_l1_regularizer,
        )
    else:
        raise ValueError(f"Invalid up-sampling variant: {variant}")
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(
            name=f"{block_name}_batch_norm",
        )(x)
    return x, crop_shape
