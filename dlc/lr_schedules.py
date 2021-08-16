import tensorflow as tf

__all__ = ["LRSchedule1"]

# Scaffold for custom LR schedule
class LRSchedule1(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, **kwargs):
        super(LRSchedule1, self).__init__(**kwargs)
        pass

    def __call__(self, step):
        pass

    def get_config(self):
        config = dict()
        return config
