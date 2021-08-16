import tensorflow as tf

from .unit import create_unit_block


def create_encoder_block(
    x,
    *,
    level,
    init_container,
    num_filters=None,
    layer_count=64,
    unit="vanilla",
    padding="valid",
    pooling="max",
    pooling_batch_norm=False,
    activation="elu",
    kernel_regularizer=None,
    relu_activity_l1_regularizer=None,
    block_name=None,
):
    """Create an U-Net encoder."""
    if num_filters is None:
        num_filters = (2 ** level) * layer_count

    if block_name is None:
        block_name = f"encoder_{level}"
    else:
        block_name = f"{block_name}_encoder_{level}"

    x = create_unit_block(
        x,
        variant=unit,
        num_filters=num_filters,
        padding=padding,
        activation=activation,
        init_container=init_container,
        block_name=block_name,
        is_encoder=True,
        kernel_regularizer=kernel_regularizer,
        relu_activity_l1_regularizer=relu_activity_l1_regularizer,
    )
    r = x  # the tensor used for the skip/residual connections

    if pooling == "max":
        x = tf.keras.layers.MaxPooling2D(
            (2, 2),
            name=f"{block_name}_max_pool",
        )(x)
    elif pooling == "avg":
        x = tf.keras.layers.AveragePooling2D(
            (2, 2),
            name=f"{block_name}_avg_pool",
        )(x)
    if pooling_batch_norm:
        x = tf.keras.layers.BatchNormalization(
            name=f"{block_name}_max_pool_batch_norm",
        )(x)
    return x, r
