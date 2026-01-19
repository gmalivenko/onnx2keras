from typing import Any, Optional, List

from keras.layers import Layer, TFOpLambda
import tensorflow as tf
import numpy as np


# this custom layer needed because of a tensorflow bug on einsum serielization
class OnnxEinsumLayer(Layer):
    """

    Args:
        equation: str
        constant_input: Optional[List[float]]
        constant_place: Optional[int]
    """

    def __init__(self, equation: str, constant_input: Optional[List[float]], constant_place: Optional[int], **kwargs):
        super().__init__(**kwargs)
        self.equation = equation
        if constant_input is not None:
            if hasattr(constant_input, 'numpy'):
                constant_input = constant_input.numpy()
            if not isinstance(constant_input, np.ndarray):
                constant_input = np.array(constant_input)
            self.constant_input = constant_input
        else:
            self.constant_input = None
        self.constant_place = constant_place

    def call(self, inputs, *args, **kwargs):
        if self.constant_input is not None:
            if self.constant_place == 1:
                inputs = [inputs, self.constant_input]
            else:
                inputs = [self.constant_input, inputs]

        return tf.einsum(self.equation, *inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "equation": self.equation,
            "constant_input": self.constant_input,
            "constant_place": self.constant_place
        })
        return config
