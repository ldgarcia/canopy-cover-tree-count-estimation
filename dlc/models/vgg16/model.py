"""A VGG16 (D-configuration) front-end implementation."""
#
# [1]
#   Simonyan, K. & Zisserman, A. (2015).
#   Very Deep Convolutional Networks for Large-Scale Image Recognition.
#   In International Conference on Learning Representations.
#   arXiv: https://arxiv.org/pdf/1409.1556.pdf
# See also:
# - https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16
import tensorflow as tf

from dlc.initializers import InitializerContainer

__all__ = ["create_encoders", "set_encoder_rgb_weights"]


def set_encoder_rgb_weights(model):
    vgg16 = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
    )
    weights = model.get_weights()
    vgg16_weights = vgg16.get_weights()
    weights[0 : len(vgg16_weights)] = vgg16_weights
    model.set_weights(weights)


def create_encoders(
    x,
    *,
    init_container: InitializerContainer,
    padding="valid",
    activation="relu",
    conv_regularizer=None,
    trainable=True,
):
    residuals = []
    # Levels 0 and 1
    for i, num_filters in ((0, 64), (1, 128)):
        block_name = f"encoder_{i}"
        x = tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            activation=activation,
            padding=padding,
            kernel_initializer=init_container.get_kernel_init(activation),
            kernel_regularizer=conv_regularizer,
            name=f"{block_name}_conv_1",
        )(x)
        x.trainable = trainable
        x = tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            activation=activation,
            padding=padding,
            kernel_initializer=init_container.get_kernel_init(activation),
            kernel_regularizer=conv_regularizer,
            name=f"{block_name}_conv_2",
        )(x)
        x.trainable = trainable
        residuals.append(x)
        x = tf.keras.layers.MaxPool2D(
            (2, 2),
            strides=2,
            name=f"{block_name}_max_pool",
        )(x)
        x.trainable = trainable
    # Level 2, 3 and 4
    for i, num_filters in ((2, 256), (3, 512), (4, 512)):
        block_name = f"encoder_{i}"
        x = tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            activation=activation,
            padding=padding,
            kernel_initializer=init_container.get_kernel_init(activation),
            kernel_regularizer=conv_regularizer,
            name=f"{block_name}_conv_1",
        )(x)
        x.trainable = trainable
        x = tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            activation=activation,
            padding=padding,
            kernel_initializer=init_container.get_kernel_init(activation),
            kernel_regularizer=conv_regularizer,
            name=f"{block_name}_conv_2",
        )(x)
        x.trainable = trainable
        x = tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            activation=activation,
            padding=padding,
            kernel_initializer=init_container.get_kernel_init(activation),
            kernel_regularizer=conv_regularizer,
            name=f"{block_name}_conv_3",
        )(x)
        x.trainable = trainable
        residuals.append(x)
        x = tf.keras.layers.MaxPool2D(
            (2, 2),
            name=f"{block_name}_max_pool",
        )(x)
        x.trainable = trainable
    return x, residuals
