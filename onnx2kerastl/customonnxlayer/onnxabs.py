from keras.layers import Layer
import tensorflow as tf


class OnnxAbs(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = tf.math.abs(inputs)
        return x
