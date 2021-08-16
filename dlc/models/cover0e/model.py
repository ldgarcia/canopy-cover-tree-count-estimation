import tensorflow as tf

from dlc.models.base.unet0 import load_unet0_model


__all__ = ["create_model", "load_model"]


def load_model(input_shape, model_weights_file):
    model = create_model(input_shape)
    model.load_weights(model_weights_file)
    return model


def create_model(
    input_shape,
    unet0_weights_file=None,
    finetune_unet=False,
    seed=None,
):
    initializer_dense = tf.keras.initializers.glorot_uniform(seed=seed)
    unet = load_unet0_model(
        weights_file=unet0_weights_file,
        input_shape=input_shape,
        name="unet",
    )
    unet.trainable = False
    unet.get_layer("out").trainable = finetune_unet
    unet_features = []
    for i in range(9, 5, -1):
        unet.get_layer(f"c{i}_1").trainable = finetune_unet
        x = unet.get_layer(f"c{i}_2")
        x.trainable = finetune_unet
        x = tf.keras.layers.GlobalAveragePooling2D(
            name=f"avg_pooling_{i}",
        )(x.output)
        x = tf.keras.layers.Dense(
            1,
            name=f"dense_64_{i}",
            kernel_initializer=initializer_dense,
        )(x)
        unet_features.append(x)
    x = tf.keras.layers.concatenate(
        unet_features,
        name="cat",
    )
    x = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="cover",
        kernel_initializer=initializer_dense,
    )(x)
    x = tf.keras.layers.Reshape(())(x)

    model = tf.keras.models.Model(
        inputs=unet.inputs,
        outputs=x,
        name="cover0e",
    )
    return model
