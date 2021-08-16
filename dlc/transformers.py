import tensorflow as tf

from dlc.layers import histogram

# As per the note in:
#   https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16
# Pre-processing consists in:
#   (1) convert the image from RGB to BGR
#   (2) zero-centering each color channel with respect the ImageNet dataset
#       (but without scaling)
@tf.function
def preprocess_rbg_input(x, y):
    x_new = tf.keras.applications.vgg16.preprocess_input(
        x,
        data_format="channels_last",
    )
    return x_new, y


@tf.function
def to_cover0_annotation(x, y):
    y_new = tf.expand_dims(tf.reduce_mean(y[..., 0]), -1)
    return x, y_new


@tf.function
def to_cover1_annotation(x, y):
    cover = tf.expand_dims(tf.reduce_mean(y[..., 0]), -1)
    y_new = dict(segmentation_map=y, cover=cover)
    return x, y_new


@tf.function
def to_cover1_annotation_deeply_supervised(x, y):
    cover = tf.expand_dims(tf.reduce_mean(y[..., 0]), -1)
    # Reference: https://git.io/JBFfR
    y_hidden_1 = y[::2, ::2, :]
    y_hidden_2 = y[::4, ::4, :]
    y_hidden_3 = y[::8, ::8, :]
    y_new = dict(
        segmentation_map=y,
        segmentation_map_hidden_1=y_hidden_1,
        segmentation_map_hidden_2=y_hidden_2,
        segmentation_map_hidden_3=y_hidden_3,
        cover=cover,
    )
    return x, y_new


DENSITY_SCALE_FACTOR = 1e2


@tf.function
def to_multi0_annotation_deeply_supervised(x, y):
    # Assumes y has the following layout:
    # channel 0: segmentation map
    # channel 1: density map
    # channel 2: boundary weights
    # Fix boundary weights
    weights = tf.where(y[..., 2] >= 0.5, 10.0, 1.0)
    y_a = tf.stack([y[..., 0], weights], -1)
    y_b_scaled = tf.expand_dims(
        tf.where(y[..., 1] > 0.0, y[..., 1] * DENSITY_SCALE_FACTOR, y[..., 1]),
        -1,
    )
    cover = tf.expand_dims(tf.reduce_mean(y[..., 0]), -1)
    count = tf.expand_dims(tf.reduce_sum(y_b_scaled), -1)
    # Deep supervision
    # Reference: https://git.io/JBFfR
    y_a_hidden_1 = y_a[::2, ::2, :]
    y_a_hidden_2 = y_a[::4, ::4, :]
    y_a_hidden_3 = y_a[::8, ::8, :]
    # Convert to multi-output dict
    y_new = dict(
        segmentation_map=y_a,
        segmentation_map_hidden_1=y_a_hidden_1,
        segmentation_map_hidden_2=y_a_hidden_2,
        segmentation_map_hidden_3=y_a_hidden_3,
        density_map=y_b_scaled,
        count=count,
        cover=cover,
    )
    return x, y_new


@tf.function
def to_density1_annotation_scaled(x, y):
    # Assumes:
    # y[..., 0] contains density map ground truth
    y_scaled = tf.expand_dims(
        tf.where(y[..., 0] > 0.0, y[..., 0] * DENSITY_SCALE_FACTOR, y[..., 0]),
        -1,
    )
    count = tf.expand_dims(tf.reduce_sum(y_scaled), -1)
    y_new = tf.concat([y_scaled, y[..., 1:]], axis=-1)
    outputs = dict(density_map=y_new, count=count)
    return x, outputs


@tf.function
def to_density1_annotation_scaled_deeply_supervised(x, y):
    # Assumes:
    # y[..., 0] contains density map ground truth
    y_scaled = tf.expand_dims(
        tf.where(y[..., 0] > 0.0, y[..., 0] * DENSITY_SCALE_FACTOR, y[..., 0]),
        -1,
    )
    count = tf.expand_dims(tf.reduce_sum(y_scaled), -1)
    y_new = tf.concat([y_scaled, y[..., 1:]], axis=-1)
    # Reference: https://git.io/JBFfR
    y_hidden_1 = y_new[::2, ::2, :]
    y_hidden_2 = y_new[::4, ::4, :]
    y_hidden_3 = y_new[::8, ::8, :]
    outputs = dict(
        density_map=y_new,
        density_map_hidden_1=y_hidden_1,
        density_map_hidden_2=y_hidden_2,
        density_map_hidden_3=y_hidden_3,
        count=count,
    )
    return x, outputs


@tf.function
def to_density2_annotation_scaled(x, y):
    # Assumes:
    # y[..., 0] contains density map ground truth
    # y[..., 1] contains segmentation ground truth
    # y[..., 2] contains boundary weigths
    y_scaled = tf.expand_dims(
        tf.where(y[..., 0] > 0.0, y[..., 0] * DENSITY_SCALE_FACTOR, y[..., 0]),
        -1,
    )
    count = tf.expand_dims(tf.reduce_sum(y_scaled), -1)
    cover = tf.expand_dims(tf.reduce_mean(y[..., 1]), -1)
    y_new = tf.concat([y_scaled, y[..., 1:]], axis=-1)
    outputs = dict(density_map=y_new, count=count, cover=cover)
    return x, outputs


@tf.function
def to_density2_annotation_scaled_deeply_supervised(x, y):
    # Assumes:
    # y[..., 0] contains density map ground truth
    # y[..., 1] contains segmentation ground truth
    # y[..., 2] contains boundary weigths
    y_scaled = tf.expand_dims(
        tf.where(y[..., 0] > 0.0, y[..., 0] * DENSITY_SCALE_FACTOR, y[..., 0]),
        -1,
    )
    count = tf.expand_dims(tf.reduce_sum(y_scaled), -1)
    cover = tf.expand_dims(tf.reduce_mean(y[..., 1]), -1)
    y_new = tf.concat([y_scaled, y[..., 1:]], axis=-1)
    # Reference: https://git.io/JBFfR
    y_hidden_1 = y_new[::2, ::2, :]
    y_hidden_2 = y_new[::4, ::4, :]
    y_hidden_3 = y_new[::8, ::8, :]
    outputs = dict(
        density_map=y_new,
        density_map_hidden_1=y_hidden_1,
        density_map_hidden_2=y_hidden_2,
        density_map_hidden_3=y_hidden_3,
        count=count,
        cover=cover,
    )
    return x, outputs


@tf.function
def fix_density2_boundary_weights(x, y):
    # Assumes:
    # y[..., 0:1] contains ground truths
    # y[..., 2] contains boundary weigths
    weights = tf.expand_dims(y[..., 2], -1)
    weights = tf.where(weights >= 0.5, 10.0, 1.0)
    y_new = tf.concat([y[..., 0:-1], weights], -1)
    return x, y_new


@tf.function
def fix_boundary_weights(x, y):
    # Assumes that channel 0 has the ground truth and that
    # channel 1 has the boundary weights.
    weights = y[..., 1]
    weights = tf.where(weights >= 0.5, 10.0, 1.0)
    y_new = tf.stack([y[..., 0], weights], -1)
    return x, y_new


@tf.function
def fix_outlier_weights(x, y):
    # Assumes that channel 0 has the ground truth and that
    # channel 1 has the outlier weights.
    weights = y[..., 1]
    weights = tf.where(
        weights == 1.0,  # non-outliers
        1.0,
        tf.where(
            weights < 0.0,  # no-data
            0.1,
            0.50,
        ),
    )
    y_new = tf.stack([y[..., 0], weights], -1)
    return x, y_new


@tf.function
def fix_fuse_boundary_and_outlier_weights(x, y):
    # Assumes that channel 0 has the ground truth and that
    # channel 1 has the boundary weights and channel 2 has the outlier weights.
    boundary_weights = y[..., 1]
    boundary_weights = tf.where(boundary_weights >= 0.5, 10.0, 1.0)
    outlier_weights = y[..., 2]
    # The outliers mask comes with outliers as 0, non-outliers as 1
    # and no-data areas due to patch creation with -1.
    outlier_weights = tf.where(
        outlier_weights > 0.5,  # non-outliers
        1.0,
        tf.where(
            outlier_weights < 0.0,  # no-data
            0.1,
            0.50,
        ),
    )
    weights = boundary_weights * outlier_weights
    y_new = tf.stack([y[..., 0], weights], -1)
    return x, y_new
