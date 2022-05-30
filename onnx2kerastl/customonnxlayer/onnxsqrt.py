import tensorflow as tf
from keras.layers import Layer


class OnnxSqrt(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = tf.math.sqrt(inputs)
        return x
