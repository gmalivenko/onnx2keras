from typing import List

import keras.backend as K
from keras.layers import Layer


class OnnxReduceMean(Layer):
    def __init__(self, axes: List[int], keepdims: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.axes = axes
        self.keepdims = keepdims

    def call(self, inputs, **kwargs):
        tensor = K.mean(inputs, keepdims=self.keepdims, axis=self.axes)
        return tensor

    def get_config(self):
        config = super().get_config()
        config.update({
            "axes": self.axes,
            "keepdims": self.keepdims,
        })
        return config
