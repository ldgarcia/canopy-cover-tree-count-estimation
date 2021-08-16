import tensorflow as tf

import dlc.models.unet0.model

__all__ = ["load_model"]

# Modified this model so that we can compute the segmentation map's
# MCC, Accuracy, DSC, Precision, Recall.
# We also simulate the post-processing thresholding.
# Basically, we just use it to load the weights and compute metrics.


def load_model(input_shape, batch_size, weights_file, **kwargs):
    unet = dlc.models.unet0.model.load_model(
        input_shape=input_shape,
        batch_size=batch_size,
        weights_file=weights_file,
        **kwargs,
    )
    unet.trainable = False
    sm = unet.output

    # Simulate the thresholding step.
    x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.round(x))(sm)
    x = tf.keras.layers.GlobalAveragePooling2D(
        name="cover",
    )(x)
    model = tf.keras.models.Model(
        inputs=unet.inputs,
        outputs=[sm, x],
        name="cover0c",
    )
    return model
