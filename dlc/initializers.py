"""Implements custom initializers."""
#
# References:
# [1]
#   Siddharth Krishna Kumar. (2017).
#   On weight initialization in deep neural networks.
#   arXiv: https://arxiv.org/abs/1704.08863
#   Code: https://git.io/JCfea
# [2]
#   Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He,
#   & Piotr Dollár. (2018).
#   Focal Loss for Dense Object Detection.
#   arXiv: https://arxiv.org/pdf/1708.02002.pdf
# [3]
#   Mishkin, D., & Matas, J. (2015).
#   All you need is a good init.
#   arXiv preprint arXiv:1511.06422.
#   arXiv: https://arxiv.org/abs/1511.06422
#   Code: https://git.io/JBWzj
#
# See also TF's implementation of He's (Kaiming's) initialization:
#   - https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal
#   - https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling
import math
from typing import Optional

import numpy as np
import tensorflow as tf

SIGMOID_SCALE = 12.8  # from [1]

SELU_ALPHA = 1.67326324
SELU_LAMBDA = 1.05070098
# Not to confuse with the lambda param above:
SELU_SCALE = 1.0 / (SELU_ALPHA * SELU_LAMBDA) ** 2


def kumar_initializer(scale, seed=None):
    """Return a scaled normal weight initializer (adapted from [1])."""
    return tf.keras.initializers.VarianceScaling(
        scale=scale,  # stddev = sqrt(scale / n)
        mode="fan_in",
        distribution="truncated_normal",  # as in TF's implementation of He's
        seed=seed,
    )


class InitializerContainer:
    """Encapsulates all the logic related to weight initialization.

    By default, uses the scaled-normal initialization, distinguishing
    by activation function (as indicated by the user) to select the best
    scale for each according to the literature. If LSUV initialization
    is to be used, then weights are orthogonal-initialized. This
    assumes the second part of the LSUV algorithm is applied afterwards
    on the whole model (see 'lsuv_initialization' below).

    Parameters
    ----------
    seed
        Seed to pass onto the random initializers.
    use_orthogonal_initialization
        Indicates that random orthogonal initialization should be used instead
        of random scaled-normal initialization.
    target_class_prior
        In the case of binary segmentation tasks, specifies the probability
        with which, at the start of the training, the pixels are classified
        as belonging to the target class (e.g., canopy cover).
    """

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        use_orthogonal_initialization: bool = False,
        target_class_prior: Optional[float] = None,
        target_class_prior_scale: Optional[float] = None,
        **kwargs,
    ):
        self.seed = seed
        self.use_orthogonal_initialization = use_orthogonal_initialization
        self.target_class_prior = target_class_prior
        self.target_class_prior_scale = target_class_prior_scale
        self.lecun_normal_init = tf.keras.initializers.LecunNormal(seed=seed)
        self.sigmoid_init = kumar_initializer(SIGMOID_SCALE, seed=seed)
        self.alt_selu_init = kumar_initializer(SELU_SCALE, seed=seed)
        self.he_init = tf.keras.initializers.HeNormal(seed=seed)
        self.orthogonal_init = tf.keras.initializers.Orthogonal(seed=seed)

    def get_kernel_init(
        self,
        activation: Optional[str] = None,
        is_output: bool = False,
        force_use_gaussian: bool = False,
    ):
        if not force_use_gaussian and self.use_orthogonal_initialization:
            return self.orthogonal_init
        elif activation is None:
            return self.lecun_normal_init
        elif activation == "elu":
            return self.lecun_normal_init
        elif activation == "selu":
            return self.alt_selu_init
        # elif activation == "selu":
        #    return self.lecun_normal_init
        elif activation == "relu":
            return self.he_init
        elif activation == "prelu":
            return self.he_init
        elif activation == "sigmoid":
            return self.sigmoid_init
        elif is_output and activation == "linear":
            # Need to revise this.
            return tf.keras.initializers.TruncatedNormal(
                mean=0.0,
                stddev=0.005,
                seed=self.seed,
            )

    def get_bias_init(
        self,
        activation: Optional[str] = None,
        is_output: bool = False,
    ):
        # See:
        # - section 3.3, page 4 of [2]
        # - https://karpathy.github.io/2019/04/25/recipe/#2-set-up-the-end-to-end-trainingevaluation-skeleton--get-dumb-baselines
        if is_output and activation == "sigmoid":
            prior = 0.5
            if self.target_class_prior is not None:
                prior = self.target_class_prior
            b = math.log(prior / (1.0 - prior))
            return tf.initializers.Constant(value=b)
        elif is_output and activation == "linear":
            prior = 0.5
            if self.target_class_prior is not None:
                prior = self.target_class_prior
            factor = 1.0
            if self.target_class_prior_scale is not None:
                factor = self.target_class_prior_scale
            b = prior * factor
            return tf.initializers.Constant(value=b)
        else:
            return "zeros"


# This function is adapted from: https://git.io/JBbrW
# See also: https://stackoverflow.com/questions/68146669/get-values-of-kerastensor
def _get_activations(
    model: tf.keras.models.Model,
    layer: tf.keras.layers.Layer,
    xs,
):
    tmp_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=layer.output,
    )
    tmp_model.build(xs.shape)
    activations = tmp_model.predict(xs)
    return activations


def lsuv_initialization(
    model: tf.keras.models.Model,
    batch: tf.data.Dataset,
    tol_var: float = 1e-5,
    target_variance: float = 1.0,
    max_iter: int = 25,
    verbose: bool = False,
):
    """LSUV (layer sequential unit-variance) initialization algoritm from [3].

    Parameters
    ----------
    model
        A keras model that has been pre-initialized using orthogonal
        initialization.

    Notes
    -----
    Adapted from [3] for Tensorflow and our pipeline.
    """
    # The following layers have trainable parameters and should be skipped.
    SKIP_TYPES = (
        tf.keras.layers.LayerNormalization,
        tf.keras.layers.BatchNormalization,
        tf.keras.layers.PReLU,
    )
    try:
        layer = None
        batch_iterator = iter(batch)
        xs, _ = next(batch_iterator)
        print(f"LSUV: got input with shape: {xs.shape}")
        # Need to run the model at least once:
        _ys = model.predict(xs)
        assert _ys is not None

        n_layers = len(model.layers)
        target_variance_sqrt = math.sqrt(target_variance)
        for i, layer in enumerate(model.layers):
            if (
                layer.trainable
                and len(layer.trainable_weights) > 0
                and not isinstance(layer, SKIP_TYPES)
            ):
                if verbose:
                    print(f"LSUV: processing layer '{layer.name}'")

                # Some layers have a decoupled activation, which
                # is handled in another layer. We mark those with
                # a '__lsuv' suffix on their name so that we can
                # track them here and perform "look-ahead" when computing
                # the corresponding activation.
                # This works because the layer list is sorted topologically.
                lookahead = None
                if layer.name.endswith("__lsuv"):
                    j = i + 1
                    while j < n_layers and lookahead is None:
                        tmp = model.layers[j]
                        if tmp.name.endswith("__lsuv_activation"):
                            lookahead = tmp
                            print(f"LSUV: using look-ahead: {lookahead.name}")
                        j += 1
                # Do-while:
                output = _get_activations(model, layer, xs)
                if lookahead is not None:
                    output = _get_activations(model, lookahead, xs)
                variance = np.var(output)
                k = 0
                while not math.isclose(
                    variance,
                    target_variance,
                    rel_tol=tol_var,
                ):
                    variance_sqrt = math.sqrt(variance)
                    if np.abs(variance_sqrt) < 1e-7:
                        break  # avoids division by zero
                    WEIGHTS_IDX = 0
                    parameters = layer.get_weights()
                    scaling_factor = variance_sqrt / target_variance_sqrt
                    parameters[WEIGHTS_IDX] /= scaling_factor
                    layer.set_weights(parameters)
                    k += 1
                    if k >= max_iter:
                        break
                    output = _get_activations(model, layer, xs)
                    if lookahead is not None:
                        output = _get_activations(model, lookahead, xs)
                    variance = np.var(output)
                if verbose:
                    print(f"LSUV: got variance of {variance:0.2f} after {k} iterations")
            else:
                if verbose:
                    print(f"LSUV: skipping layer '{layer.name}'")
    except RuntimeError as e:
        layer_name = layer.name if layer is not None else None
        print(f"Error ocurred while processing layer: {layer_name}")
        raise e
