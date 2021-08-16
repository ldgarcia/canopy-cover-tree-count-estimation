from collections import ChainMap

import tensorflow as tf

import dlc.models.unet.model

__all__ = ["create_model", "load_model"]


def load_model(input_shape, batch_size, weights_file, **kwargs):
    model = create_model(
        input_shape=input_shape,
        batch_size=batch_size,
        **kwargs,
    )
    model.load_weights(weights_file)
    return model


def create_model(*, deeply_supervised, **kwargs):
    unet = dlc.models.unet.model.create_model(
        **dict(
            ChainMap(
                dict(
                    name="unet",
                    output_name="segmentation_map",
                    output_activation="sigmoid",
                    second_output_name="density_map",
                    second_output_activation="linear",
                    deeply_supervised=deeply_supervised,
                ),
                kwargs,
            )
        )
    )
    outputs = unet.output
    sm_out = outputs[0]
    dm_out = outputs[1]

    # Computing cover scalar
    cover = tf.keras.layers.GlobalAveragePooling2D(name="cover")(sm_out)
    # Computing count scalar
    count = tf.keras.layers.Lambda(
        lambda x: tf.expand_dims(
            tf.reduce_sum(
                x,
                axis=(1, 2, 3),
            ),
            -1,
        ),
        name="count",
    )(dm_out)

    outputs += [cover, count]
    model = tf.keras.models.Model(
        inputs=unet.inputs,
        outputs=outputs,
        name="multi0",
    )
    return model
