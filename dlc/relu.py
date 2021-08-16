"""Module to support exploring the various ReLU alternatives."""
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import PReLU

__all__ = ["get_relu_variant_layer", "RELU_VARIANTS"]

RELU_VARIANTS = [
    "relu",
    "leaky_relu",
    "prelu",
    "elu",
    "selu",
]


def get_relu_variant_layer(
    *,
    variant,
    **kwargs,
):
    if variant == "relu":
        return Activation("relu", **kwargs)
    elif variant == "leaky_relu":
        return LeakyReLU(alpha=0.55, **kwargs)
    elif variant == "prelu":
        return PReLU(**kwargs)
    elif variant == "elu":
        return Activation("elu", **kwargs)
    elif variant == "selu":
        return Activation("selu", **kwargs)
    else:
        raise ValueError(f"Invalid ReLU variant: {variant}")
