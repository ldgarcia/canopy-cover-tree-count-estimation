# Code for working with the published U-Net model v1.0.0 from the GitHub repo.
import tensorflow as tf

import dlc.losses
import dlc.metrics


__all__ = [
    "load_model",
    "create_model",
    "retrieve_model",
]


# Assumes we're using an H5 file that only stores the weights.
def load_model(
    *,
    input_shape,
    batch_size,
    weights_file=None,
    name=None,
    **kwargs,
):
    model = create_model(
        input_shape=input_shape,
        batch_size=batch_size,
        name=name,
        **kwargs,
    )
    if weights_file is not None:
        model.load_weights(weights_file)
    return model


# From: https://git.io/JsiUc
def create_model(
    *,
    input_shape,
    output_channels,
    output_name,
    batch_size,
    name,
    input_dtype,
    layer_count,
    **_kwargs,
):
    input = tf.keras.Input(
        shape=input_shape,
        batch_size=batch_size,
        dtype=input_dtype,
        name="input",
    )
    x = input

    gaussian_noise = 0.1
    x = tf.keras.layers.GaussianNoise(gaussian_noise)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    c1 = tf.keras.layers.Conv2D(
        1 * layer_count,
        (3, 3),
        activation="relu",
        padding="same",
        name="c1_1",
    )(x)
    c1 = tf.keras.layers.Conv2D(
        1 * layer_count,
        (3, 3),
        activation="relu",
        padding="same",
        name="c1_2",
    )(c1)
    n1 = tf.keras.layers.BatchNormalization(name="n1")(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2), name="p1")(n1)

    c2 = tf.keras.layers.Conv2D(
        2 * layer_count,
        (3, 3),
        activation="relu",
        padding="same",
        name="c2_1",
    )(p1)
    c2 = tf.keras.layers.Conv2D(
        2 * layer_count,
        (3, 3),
        activation="relu",
        padding="same",
        name="c2_2",
    )(c2)
    n2 = tf.keras.layers.BatchNormalization(name="n2")(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2), name="p2")(n2)

    c3 = tf.keras.layers.Conv2D(
        4 * layer_count,
        (3, 3),
        activation="relu",
        padding="same",
        name="c3_1",
    )(p2)
    c3 = tf.keras.layers.Conv2D(
        4 * layer_count,
        (3, 3),
        activation="relu",
        padding="same",
        name="c3_2",
    )(c3)
    n3 = tf.keras.layers.BatchNormalization(name="n3")(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2), name="p3")(n3)

    c4 = tf.keras.layers.Conv2D(
        8 * layer_count,
        (3, 3),
        activation="relu",
        padding="same",
        name="c4_1",
    )(p3)
    c4 = tf.keras.layers.Conv2D(
        8 * layer_count,
        (3, 3),
        activation="relu",
        padding="same",
        name="c4_2",
    )(c4)
    n4 = tf.keras.layers.BatchNormalization(name="n4")(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="p4")(n4)

    c5 = tf.keras.layers.Conv2D(
        16 * layer_count,
        (3, 3),
        activation="relu",
        padding="same",
        name="c5_1",
    )(p4)
    c5 = tf.keras.layers.Conv2D(
        16 * layer_count,
        (3, 3),
        activation="relu",
        padding="same",
        name="c5_2",
    )(c5)

    u6 = tf.keras.layers.UpSampling2D((2, 2), name="u6")(c5)
    n6 = tf.keras.layers.BatchNormalization(name="n6")(u6)
    u6 = tf.keras.layers.concatenate([n6, n4], name="concat6")
    c6 = tf.keras.layers.Conv2D(
        8 * layer_count,
        (3, 3),
        activation="relu",
        padding="same",
        name="c6_1",
    )(u6)
    c6 = tf.keras.layers.Conv2D(
        8 * layer_count,
        (3, 3),
        activation="relu",
        padding="same",
        name="c6_2",
    )(c6)

    u7 = tf.keras.layers.UpSampling2D((2, 2), name="u7")(c6)
    n7 = tf.keras.layers.BatchNormalization(name="n7")(u7)
    u7 = tf.keras.layers.concatenate([n7, n3], name="concat7")
    c7 = tf.keras.layers.Conv2D(
        4 * layer_count,
        (3, 3),
        activation="relu",
        padding="same",
        name="c7_1",
    )(u7)
    c7 = tf.keras.layers.Conv2D(
        4 * layer_count,
        (3, 3),
        activation="relu",
        padding="same",
        name="c7_2",
    )(c7)

    u8 = tf.keras.layers.UpSampling2D((2, 2), name="u8")(c7)
    n8 = tf.keras.layers.BatchNormalization(name="n8")(u8)
    u8 = tf.keras.layers.concatenate([n8, n2], name="concat8")
    c8 = tf.keras.layers.Conv2D(
        2 * layer_count,
        (3, 3),
        activation="relu",
        padding="same",
        name="c8_1",
    )(u8)
    c8 = tf.keras.layers.Conv2D(
        2 * layer_count,
        (3, 3),
        activation="relu",
        padding="same",
        name="c8_2",
    )(c8)

    u9 = tf.keras.layers.UpSampling2D((2, 2), name="u9")(c8)
    n9 = tf.keras.layers.BatchNormalization(name="n9")(u9)
    u9 = tf.keras.layers.concatenate([n9, n1], axis=3, name="concat9")
    c9 = tf.keras.layers.Conv2D(
        1 * layer_count,
        (3, 3),
        activation="relu",
        padding="same",
        name="c9_1",
    )(u9)
    c9 = tf.keras.layers.Conv2D(
        1 * layer_count,
        (3, 3),
        activation="relu",
        padding="same",
        name="c9_2",
    )(c9)

    d = tf.keras.layers.Conv2D(
        output_channels,
        (1, 1),
        activation="sigmoid",
        name=output_name,
    )(c9)

    model = tf.keras.models.Model(inputs=input, outputs=d, name=name)
    return model


# Function to retrieve model weights from the H5 file published in GitHub:
# - https://git.io/JnckF (Sudan and Sahel)
# - https://git.io/JncIg (Sahara)
# See also:
# - scripts/retrieve_model_weights.py
# - retrieve_model_weights.yml
def retrieve_model(model_file, compile=False, is_sudan_h5=False):
    custom_objects = {
        "tversky": dlc.losses.tversky,
        "dice_coef": dlc.metrics.dice_coef,
        "dice_loss": dlc.metrics.dice_loss,
        "accuracy": dlc.metrics.accuracy,
        "specificity": dlc.metrics.specificity,
        "sensitivity": dlc.metrics.sensitivity,
        "tf": tf,
    }
    model = tf.keras.models.load_model(
        model_file,
        custom_objects=custom_objects,
        compile=compile,
    )
    if is_sudan_h5:
        model = model.layers[3]
    return model
