"""Metrics related to the confusion matrix for binary classification."""
import tensorflow as tf

# TP, TN, FP, FN helpers:


@tf.function
def _true_positives(y_true, y_pred):
    return y_true * y_pred


@tf.function
def _true_negatives(y_true, y_pred):
    return (1.0 - y_true) * (1.0 - y_pred)


@tf.function
def _false_positives(y_true, y_pred):
    return (1.0 - y_true) * y_pred


@tf.function
def _false_negatives(y_true, y_pred):
    return y_true * (1.0 - y_pred)


# Classwise:


@tf.function
def precision(y_true, y_pred):
    y_true_0 = y_true[..., 0]
    y_pred = tf.keras.backend.round(tf.squeeze(y_pred))
    tp = tf.reduce_sum(_true_positives(y_true_0, y_pred))
    fp = tf.reduce_sum(_false_positives(y_true_0, y_pred))
    return tf.math.divide_no_nan(tp, tp + fp)


@tf.function
def recall(y_true, y_pred):
    y_true_0 = y_true[..., 0]
    y_pred = tf.keras.backend.round(tf.squeeze(y_pred))
    tp = tf.reduce_sum(_true_positives(y_true_0, y_pred))
    fn = tf.reduce_sum(_false_negatives(y_true_0, y_pred))
    return tf.math.divide_no_nan(tp, tp + fn)


# Global:


@tf.function
def accuracy(y_true, y_pred):
    """Compute the accuracy."""
    y_true_0 = y_true[..., 0]
    y_pred = tf.keras.backend.round(tf.squeeze(y_pred))
    tp = tf.reduce_sum(_true_positives(y_true_0, y_pred))
    tn = tf.reduce_sum(_true_negatives(y_true_0, y_pred))
    fp = tf.reduce_sum(_false_positives(y_true_0, y_pred))
    fn = tf.reduce_sum(_false_negatives(y_true_0, y_pred))
    num = tp + tn
    denom = num + fp + fn
    return tf.math.divide_no_nan(num, denom)


@tf.function
def dsc(y_true, y_pred):
    """Compute Sørensen–Dice coefficient."""
    y_true_0 = y_true[..., 0]
    y_pred = tf.keras.backend.round(tf.squeeze(y_pred))
    tp = tf.reduce_sum(_true_positives(y_true_0, y_pred))
    fp = tf.reduce_sum(_false_positives(y_true_0, y_pred))
    fn = tf.reduce_sum(_false_negatives(y_true_0, y_pred))
    num = 2.0 * tp
    denom = num + fp + fn
    return tf.math.divide_no_nan(num, denom)


@tf.function
def mcc(y_true, y_pred):
    """Compute the Matthews correlation coefficient."""
    y_true_0 = y_true[..., 0]
    y_pred = tf.keras.backend.round(tf.squeeze(y_pred))
    tp = tf.reduce_sum(_true_positives(y_true_0, y_pred))
    tn = tf.reduce_sum(_true_negatives(y_true_0, y_pred))
    fp = tf.reduce_sum(_false_positives(y_true_0, y_pred))
    fn = tf.reduce_sum(_false_negatives(y_true_0, y_pred))
    num = tp * tn - fp * fn
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    denom = tf.math.sqrt(denom_sq)
    return tf.math.divide_no_nan(num, denom)
