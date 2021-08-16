import tensorflow as tf

import dlc.models.unet0.model


__all__ = ["create_model", "load_model"]


def load_model(input_shape, model_file):
    model = create_model(input_shape)
    model.load_weights(model_file)
    return model


def create_model(input_shape, batch_size, base_weights, **kwargs):
    unet = dlc.models.unet0.model.load_model(
        input_shape=input_shape,
        batch_size=batch_size,
        weights_file=base_weights,
        name="unet",
        **kwargs,
    )
    unet.trainable = False
    unet.get_layer("c9_1").trainable = True
    unet.get_layer("c9_2").trainable = True
    unet.get_layer("out").trainable = True

    x = unet.output
    x = tf.keras.layers.GlobalAveragePooling2D(
        name="cover",
    )(x)
    x = tf.keras.layers.Reshape(())(x)
    model = tf.keras.models.Model(inputs=unet.inputs, outputs=x, name="cover0d")
    return model
