import tensorflow as tf

import dlc.models.unet0.model

__all__ = ["load_model"]


def load_model(input_shape, batch_size, weights_file, **kwargs):
    unet = dlc.models.unet0.model.load_model(
        input_shape=input_shape,
        batch_size=batch_size,
        weights_file=weights_file,
        name="unet",
        **kwargs,
    )
    unet.trainable = False
    x = unet.output
    x = tf.keras.layers.Lambda(
        lambda x: tf.expand_dims(tf.cast(x[..., 0] >= 0.5, dtype=tf.float32), -1),
    )(x)
    x = tf.keras.layers.Lambda(
        lambda x: tf.reduce_mean(
            x,
            axis=(1, 2, 3),
        ),
        name="cover",
    )(x)
    model = tf.keras.models.Model(
        inputs=unet.inputs,
        outputs=x,
        name="cover0b",
    )
    return model
