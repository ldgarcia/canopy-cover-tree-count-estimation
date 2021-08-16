import numpy as np
import tensorflow as tf

# The function below is adapted from: https://git.io/JsiZn
def add_cropping_if_necessary(r, x_shape, level=None):
    """Ensure that the shape of r agrees with that of x by \
            adding an optional cropping layer."""
    name = None
    if level is not None:
        name = f"cropping2d_{level}"
    r_shape = np.array(r.shape.as_list()[1:-1])
    x_shape = np.array(x_shape.as_list()[1:-1])
    if np.any(r_shape != x_shape):
        tmp = (r_shape - x_shape).astype(int)
        cr = np.array([tmp // 2, tmp // 2]).T
        cr[:, 1] += tmp % 2
        cropping2d = tf.keras.layers.Cropping2D(cr, name=name)
        return cropping2d(r)
    return r
