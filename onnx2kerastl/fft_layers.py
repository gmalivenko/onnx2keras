import tensorflow as tf


def convert_dft(node, params, layers, lambda_func, node_name, keras_name):
    # Currently, there's no way to run this - pytorch export not supported + Onnx Runtime version is too advanced.
    raise AttributeError("DFT could not yet be converted - contact Tensorleap for support")
    axis = params.get('axis', 1)
    inverse = bool(params.get('inverse', 0))
    onesided = bool(params.get('onesided', 0))
    input_0 = layers[node.input[0]]
    if len(node.input[0]) == 2:
        fft_length = layers[node.input[0]]
    else:
        fft_length = None
    rank_tensor = len(input_0.shape)
    rotated_axis = False
    if axis != rank_tensor-1 and axis != -1: # tf.fft works only on last dimension - permuting
        if axis == 0:
            raise AttributeError("FFT on the batch dimension isn't convertable")
        output = tf.keras.layers.Permute((axis, rank_tensor-1))(input_0)
        rotated_axis = True
    else:
        output = input_0
    if inverse and onesided:
        output = tf.signal.irfft(input_0, fft_length=fft_length)
    if inverse and not onesided:
        output = tf.signal.ifft(input_0, fft_length=fft_length)
    if onesided:
        output = tf.signal.rfft(input_0, fft_length=fft_length)
    if not onesided:
        output = tf.signal.fft(input_0, fft_length=fft_length)
    if rotated_axis:
        layers[node_name] = tf.keras.layers.Permute((rank_tensor-1, axis))(output)
    else:
        layers[node_name] = output
