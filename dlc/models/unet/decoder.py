import tensorflow as tf

from .attention import create_attention_block
from .crop import add_cropping_if_necessary
from .unit import create_unit_block
from .upsampling import create_upsampling_block


def creade_decoder_block(
    x,
    r,
    *,
    level,
    init_container,
    num_filters=None,
    layer_count=64,
    padding="valid",
    unit="vanilla",
    upsampling="odena_2016",
    upsampling_use_batch_norm=False,
    activation="elu",
    use_attention_gate=False,
    attention_relu_variant="prelu",
    attention_use_batch_norm=False,
    concat_use_batch_norm=False,
    return_skip_connection=False,
    kernel_regularizer=None,
    relu_activity_l1_regularizer=None,
    block_name=None,
):
    """Create an U-Net decoder."""
    if num_filters is None:
        num_filters = (2 ** level) * layer_count
    if block_name is None:
        block_name = f"decoder_{level}"
    else:
        block_name = f"{block_name}_decoder_{level}"

    attention_g = x
    # Up-convolution
    x, crop_shape = create_upsampling_block(
        x,
        num_filters=num_filters,
        block_name=f"{block_name}_up",
        size=(2, 2),
        variant=upsampling,
        use_batch_norm=upsampling_use_batch_norm,
        activation=activation,
        init_container=init_container,
        regularizer=kernel_regularizer,
        relu_activity_l1_regularizer=relu_activity_l1_regularizer,
    )
    # Concatenation with residual connections
    r = add_cropping_if_necessary(r, crop_shape, level)
    # Low-level feature maps are not used (see p. 5 of [5])
    if use_attention_gate and level > 0:
        r = create_attention_block(
            r,
            attention_g,
            in_channels=num_filters,
            level=level,
            block_name=block_name,
            relu_variant=attention_relu_variant,
            init_container=init_container,
            kernel_regularizer=kernel_regularizer,
            relu_activity_l1_regularizer=relu_activity_l1_regularizer,
            use_batch_norm=attention_use_batch_norm,
        )
    x = tf.keras.layers.Concatenate(
        axis=-1,
        name=f"{block_name}_cat",
    )([r, x])
    if concat_use_batch_norm:
        x = tf.keras.layers.BatchNormalization(
            name=f"{block_name}_cat_batch_norm",
        )(x)
    # Unit
    x = create_unit_block(
        x,
        variant=unit,
        num_filters=num_filters,
        activation=activation,
        padding=padding,
        block_name=block_name,
        is_encoder=False,
        init_container=init_container,
        kernel_regularizer=kernel_regularizer,
        relu_activity_l1_regularizer=relu_activity_l1_regularizer,
    )
    if return_skip_connection:
        return x, r
    return x
