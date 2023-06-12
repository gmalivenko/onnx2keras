import logging
import tensorflow as tf
import numpy as np
from onnx2kerastl.customonnxlayer.onnxlstm import OnnxLSTM

from .exceptions import UnsupportedLayer
from .utils import ensure_tf_type, ensure_numpy_type


def convert_lstm(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert convolution layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.conv')

    if node.input[4] != '':
        raise UnsupportedLayer('LSTM with non default sequence_lens')
    if 'direction' in params:
        direction = params['direction']
        if isinstance(direction, bytes):
            direction = direction.decode("utf-8")
        if direction != 'forward':
            raise UnsupportedLayer(f"LSTM with {direction} direction")
    should_return_state = len(node.output) == 3
    input_tensor = tf.transpose(ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name[0]), perm=[1, 0, 2])
    weights_w = ensure_numpy_type(layers[node.input[1]])[0]
    weights_r = ensure_numpy_type(layers[node.input[2]])[0]
    weights_b = ensure_numpy_type(layers[node.input[3]])[0]

    initial_h_state = tf.cast(tf.squeeze(ensure_tf_type(layers[node.input[5]]), axis=1), input_tensor.dtype)
    initial_c_state = tf.cast(tf.squeeze(ensure_tf_type(layers[node.input[6]]), axis=1), input_tensor.dtype)

    tf.keras.backend.set_image_data_format("channels_last")
    hidden_size = params['hidden_size']
    lstm_layer = OnnxLSTM(hidden_size, return_sequences=True, return_lstm_state=should_return_state)
    res = lstm_layer(input_tensor, initial_h_state, initial_c_state)
    # prepare the keras lstm weights from the onnx inputs:
    w1 = np.concatenate([weights_w[0:hidden_size, :], weights_w[2 * hidden_size:3 * hidden_size, :],
                         weights_w[3 * hidden_size:4 * hidden_size, :],
                         weights_w[hidden_size:2 * hidden_size, :]]).transpose()
    w2 = np.concatenate([weights_r[0:hidden_size, :], weights_r[2 * hidden_size:3 * hidden_size, :],
                         weights_r[3 * hidden_size:4 * hidden_size, :],
                         weights_r[hidden_size:2 * hidden_size, :]]).transpose()
    weights_b_part1 = weights_b[:w2.shape[1]]
    weights_b_part2 = weights_b[w2.shape[1]:]
    bias1 = np.concatenate([weights_b_part1[0:hidden_size], weights_b_part1[2 * hidden_size:3 * hidden_size],
                            weights_b_part1[3 * hidden_size:4 * hidden_size],
                            weights_b_part1[hidden_size:2 * hidden_size]]).transpose()
    bias2 = np.concatenate([weights_b_part2[0:hidden_size], weights_b_part2[2 * hidden_size:3 * hidden_size],
                            weights_b_part2[3 * hidden_size:4 * hidden_size],
                            weights_b_part2[hidden_size:2 * hidden_size]]).transpose()
    bias = bias1 + bias2
    res.node.layer.set_weights([w1, w2, bias])
    tf.keras.backend.set_image_data_format("channels_first")
    if should_return_state:
        c_out = res[:, -1, :]
        h_out = res[:, 0, :]
        lstm_tensor = res[:, 1:-1, :]
        layers[node.output[1]] = c_out
        layers[node.output[2]] = h_out
    else:
        lstm_tensor = res
    lstm_tensor_in_onnx_order = tf.transpose(lstm_tensor, perm=[1, 0, 2])
    lstm_tensor_in_onnx_order = tf.expand_dims(lstm_tensor_in_onnx_order, axis=1)
    layers[node_name] = lstm_tensor_in_onnx_order

