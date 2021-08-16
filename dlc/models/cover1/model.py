from collections import ChainMap

import tensorflow as tf
from tensorflow import keras

import dlc.models.unet.model

__all__ = ["create_model", "load_model"]


def load_model(
    input_shape,
    batch_size,
    weights_file,
    enable_eval_output=False,
    **kwargs,
):
    model = create_model(
        input_shape=input_shape,
        batch_size=batch_size,
        enable_eval_output=enable_eval_output,
        **kwargs,
    )
    model.load_weights(weights_file)
    return model


# Note: enable_eval_output is for performing the thresholding by 0.5
#       before computing the scalar. We do this to take advantage
#       of the existing evaluation pipeline.
def create_model(
    *,
    deeply_supervised,
    enable_eval_output=False,
    **kwargs,
):
    unet = dlc.models.unet.model.create_model(
        **dict(
            ChainMap(
                dict(
                    name="unet",
                    output_name="segmentation_map",
                    output_activation="sigmoid",
                    deeply_supervised=deeply_supervised,
                ),
                kwargs,
            )
        )
    )
    sm = unet.output
    sm_out = sm if not deeply_supervised else sm[0]

    # Computing cover scalar
    c = tf.keras.layers.GlobalAveragePooling2D(name="cover")(sm_out)

    outputs = [sm, c] if not deeply_supervised else sm + [c]

    if enable_eval_output:
        print("Will enable additional, evaluation-step, output.")
        # Simulate the thresholding step.
        x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.round(x))(sm_out)
        bin_c = tf.keras.layers.GlobalAveragePooling2D(
            name="cover_after_threshold",
        )(x)
        outputs.append(bin_c)

    model = tf.keras.models.Model(
        inputs=unet.inputs,
        outputs=outputs,
        name="cover1",
    )
    return model
