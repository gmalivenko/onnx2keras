from keras.layers import Layer
import tensorflow as tf


class OnnxLSTM(Layer):
    """

    Args:
        units: int
        return_sequences: bool
        return_lstm_state: bool
        **kwargs:
    """

    def __init__(self, units: int, return_sequences: bool, return_lstm_state: bool,  **kwargs):
        super().__init__(**kwargs)
        self.lstm_layer = tf.keras.layers.LSTM(units, return_sequences=return_sequences,
                                               return_state=return_lstm_state)
        self.return_lstm_state = return_lstm_state
        self.return_sequences = return_sequences
        self.units = units

    def call(self, inputs, initial_h_state=None, initial_c_state=None, **kwargs):
        if initial_h_state is not None and initial_c_state is not None:
            initial_states = [initial_h_state, initial_c_state]
        else:
            initial_states = None
        res = self.lstm_layer(inputs, initial_state=initial_states, **kwargs)
        if self.return_lstm_state:
            lstm_tensor, h_out, c_out = res
            lstm_flat = tf.keras.layers.Flatten()(lstm_tensor)
            h_flat = tf.keras.layers.Flatten()(h_out)
            c_flat = tf.keras.layers.Flatten()(c_out)
            concat_output = tf.keras.layers.Concatenate()([lstm_flat, h_flat, c_flat])
            return concat_output
        else:
            return res

    def build(self, input_shape):
        self.lstm_layer.build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "return_sequences": self.return_sequences,
            "return_lstm_state": self.return_lstm_state,
            "units": self.units
        })
        return config
