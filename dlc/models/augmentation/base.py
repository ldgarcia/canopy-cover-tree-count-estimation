import abc

import tensorflow as tf


__all__ = ["AugmentationTransform"]


class AugmentationTransform(metaclass=abc.ABCMeta):
    """Define the interface of an augmentation transform."""

    @abc.abstractmethod
    def _augment(self, features, annotations):
        pass

    @tf.function
    def _augment_tf(self, features, annotations):
        return tf.numpy_function(
            self._augment, [features, annotations], [tf.float32, tf.float32]
        )

    @tf.function
    def __call__(self, features, annotations):
        return self._augment_tf(features, annotations)
