from keras.layers import Layer
import tensorflow as tf


class OnnxHardSigmoid(Layer):
    def __init__(self, alpha: float = 0.2, beta: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def call(self, inputs, **kwargs):
        x = tf.multiply(inputs, self.alpha)
        x = tf.add(x, self.beta)
        x = tf.clip_by_value(x, 0., 1.)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "beta": self.beta,
        })
        return config
