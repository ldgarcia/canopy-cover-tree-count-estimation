"""A resize block implementation."""
import tensorflow as tf


__all__ = ["create_resizing_block"]


def create_resizing_block(
    x,
    *,
    input_shape,
    num_filters,
    block_name,
    use_batch_norm,
    glorot_uniform_init,
    interpolation="nearest",
):
    if interpolation not in ["nearest", "bilinear"]:
        raise ValueError(f"Invalid interpolation: {interpolation}")

    x = tf.keras.layers.experimental.preprocessing.Resizing(
        input_shape[0],
        input_shape[1],
        interpolation=interpolation,
        name=f"{block_name}_up_resize",
    )(x)
    x = tf.keras.layers.Conv2D(
        num_filters,
        (2, 2),
        name=f"{block_name}_up_conv",
        padding="same",
        activation=None,
        kernel_initializer=glorot_uniform_init,
    )(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(
            name=f"{block_name}_batch_norm",
        )(x)
    return x
