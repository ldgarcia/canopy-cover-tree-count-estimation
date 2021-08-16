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


def create_model(deeply_supervised, **kwargs):
    unet = dlc.models.unet.model.create_model(
        **dict(
            ChainMap(
                dict(
                    name="unet",
                    output_name="density_map",
                    deeply_supervised=deeply_supervised,
                ),
                kwargs,
            )
        )
    )
    dm = unet.output
    dm_out = dm if not deeply_supervised else dm[0]

    c = tf.keras.layers.Lambda(
        lambda x: tf.expand_dims(
            tf.reduce_sum(
                x,
                axis=(1, 2, 3),
            ),
            -1,
        ),
        name="count",
    )(dm_out)

    outputs = [dm, c] if not deeply_supervised else dm + [c]
    model = tf.keras.models.Model(
        inputs=unet.inputs,
        outputs=outputs,
        name="density1",
    )
    return model
