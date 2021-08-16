# References:
# - https://www.tensorflow.org/api_docs/python/tf/random/set_seed
# - https://github.com/albumentations-team/albumentations/issues/93
# - https://github.com/NVIDIA/framework-determinism
import os
import random
from typing import Tuple

import dill as pickle
import imgaug
import numpy as np
import tensorflow as tf


def load_state(state_file: str) -> Tuple[int, ...]:

    # Load state for the following RNG objects
    with open(state_file, "rb") as src:
        state = pickle.load(src)
    st0, st1, st2, seed, splitter_seed = state

    random.seed(seed)
    imgaug.random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    random.setstate(st0)
    imgaug.random.get_global_rng().state = st1
    np.random.set_state(st2)

    return seed, splitter_seed


def save_state(seeds: Tuple[int, ...], state_file: str) -> None:
    st0 = random.getstate()
    st1 = imgaug.random.get_global_rng().state
    st2 = np.random.get_state()
    seed, splitter_seed = seeds
    state = (st0, st1, st2, seed, splitter_seed)
    with open(state_file, "wb") as dst:
        pickle.dump(state, dst)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    imgaug.random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def check_environment() -> None:
    if os.environ.get("TF_DETERMINISTIC_OPS", None) is None:
        print("TF_DETERMINISTIC_OPS is not set")

    if os.environ.get("PYTHONHASHSEED", None) is None:
        print("PYTHONHASHSEED is not set")

    gpu_dev = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if gpu_dev is not None:
        if os.environ.get("TF_CUDNN_DETERMINISTIC", None) is None:
            print("TF_CUDNN_DETERMINISTIC is not set")
