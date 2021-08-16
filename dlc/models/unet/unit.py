import tensorflow as tf

from dlc.relu import get_relu_variant_layer


__all__ = ["create_unit_block", "UNIT_VARIANTS"]


UNIT_VARIANTS = [
    "vanilla",
    "vanilla_bn",
    "sdunet",
]


def _vanilla_unit(
    x,
    *,
    num_filters,
    activation,
    padding,
    block_name,
    init_container,
    kernel_regularizer,
    relu_activity_l1_regularizer,
    output_name,
    **kwargs,  # catch-all dummy
):
    activity_regularizer = None
    if relu_activity_l1_regularizer is not None:
        l1 = relu_activity_l1_regularizer
        activity_regularizer = tf.keras.regularizers.L1(l1=l1)
    for i in range(2):
        x = tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            activation=activation,
            padding=padding,
            name=f"{block_name}_conv_{i}",
            kernel_initializer=init_container.get_kernel_init("relu"),
            bias_initializer=init_container.get_bias_init("relu"),
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
        )(x)
    if output_name is not None:
        x = tf.keras.layers.Activation(
            "linear",
            name=output_name,
        )(x)
    return x


def _vanilla_bn_unit(
    x,
    *,
    num_filters,
    activation,
    padding,
    block_name,
    init_container,
    kernel_regularizer,
    relu_activity_l1_regularizer,
    output_name,
    **kwargs,  # catch-all dummy
):
    activity_regularizer = None
    if relu_activity_l1_regularizer is not None:
        l1 = relu_activity_l1_regularizer
        activity_regularizer = tf.keras.regularizers.L1(l1=l1)
    for i in range(2):
        x = tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            activation=activation,
            padding=padding,
            name=f"{block_name}_conv_{i}",
            kernel_initializer=init_container.get_kernel_init("relu"),
            bias_initializer=init_container.get_bias_init("relu"),
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
        )(x)
        x = tf.keras.layers.BatchNormalization(
            name=f"{block_name}_batch_norm",
        )(x)
    if output_name is not None:
        x = tf.keras.layers.Activation(
            "linear",
            name=output_name,
        )(x)
    return x


def _sdunet_unit(
    x,
    *,
    num_filters,
    activation,
    padding,
    block_name,
    init_container,
    kernel_regularizer,
    relu_activity_l1_regularizer,
    output_name,
    **kwargs,  # catch-all dummy
):
    """Create a block corresponding to the unit used in [7]."""
    assert (num_filters % 16) == 0, "num_filters should be a multiple of 16"
    activity_regularizer = None
    if relu_activity_l1_regularizer is not None:
        l1 = relu_activity_l1_regularizer
        activity_regularizer = tf.keras.regularizers.L1(l1=l1)
    x = tf.keras.layers.Conv2D(
        num_filters // 2,
        (3, 3),
        activation=activation,
        padding=padding,
        name=f"{block_name}_conv",
        kernel_initializer=init_container.get_kernel_init("relu"),
        bias_initializer=init_container.get_bias_init("relu"),
        kernel_regularizer=kernel_regularizer,
        activity_regularizer=activity_regularizer,
    )(x)
    xs = [x]
    # Original pairs used by [7] are: [(3, 4), (6, 8), (9, 16), (12, 16)]
    # Here we use alternative pairs adapted from the dilation factor used by
    # Resunet-a's, which was developed for satellite imagery applications.
    pairs = [(3, 4), (7, 8), (15, 16), (31, 16)]
    for i, pair in enumerate(pairs, 1):
        activity_regularizer = None
        if relu_activity_l1_regularizer is not None:
            l1 = relu_activity_l1_regularizer
            activity_regularizer = tf.keras.regularizers.L1(l1=l1)
        dilation_rate, factor = pair
        x = tf.keras.layers.Conv2D(
            num_filters // factor,
            (3, 3),
            activation=activation,
            padding="same",
            dilation_rate=dilation_rate,
            name=f"{block_name}_dilated_conv_{i}",
            kernel_initializer=init_container.get_kernel_init("relu"),
            bias_initializer=init_container.get_bias_init("relu"),
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
        )(x)
        xs.append(x)
    x = tf.keras.layers.Concatenate(
        axis=-1,
        name=f"{block_name}_unit_cat",
    )(xs)
    if output_name is not None:
        x = tf.keras.layers.Activation(
            "linear",
            name=output_name,
        )(x)
    return x


def create_unit_block(
    x,
    *,
    variant="vanilla",
    num_filters,
    activation,
    padding,
    block_name,
    is_encoder,
    init_container,
    kernel_regularizer,
    relu_activity_l1_regularizer,
    output_name=None,
    **kwargs,
):
    """Create an instance of a U-Net basic unit."""
    if variant == "vanilla":
        return _vanilla_unit(
            x,
            num_filters=num_filters,
            activation=activation,
            padding=padding,
            block_name=block_name,
            is_encoder=is_encoder,
            init_container=init_container,
            kernel_regularizer=kernel_regularizer,
            relu_activity_l1_regularizer=relu_activity_l1_regularizer,
            output_name=output_name,
            **kwargs,
        )
    elif variant == "vanilla_bn":
        return _vanilla_bn_unit(
            x,
            num_filters=num_filters,
            activation=activation,
            padding=padding,
            block_name=block_name,
            is_encoder=is_encoder,
            init_container=init_container,
            kernel_regularizer=kernel_regularizer,
            relu_activity_l1_regularizer=relu_activity_l1_regularizer,
            output_name=output_name,
            **kwargs,
        )
    elif variant == "sdunet":
        return _sdunet_unit(
            x,
            num_filters=num_filters,
            activation=activation,
            padding=padding,
            block_name=block_name,
            is_encoder=is_encoder,
            init_container=init_container,
            kernel_regularizer=kernel_regularizer,
            relu_activity_l1_regularizer=relu_activity_l1_regularizer,
            output_name=output_name,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid unit variant: {variant}")
