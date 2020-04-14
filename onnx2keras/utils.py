import numpy as np
from tensorflow import keras


def is_numpy(obj):
    """
    Check of the type is instance of numpy array
    :param obj: object to check
    :return: True if the object is numpy-type array.
    """
    return isinstance(obj, (np.ndarray, np.generic))


def ensure_numpy_type(obj):
    """
    Raise exception if it's not a numpy
    :param obj: object to check
    :return: numpy object
    """
    if is_numpy(obj):
        return obj
    else:
        raise AttributeError('Not a numpy type.')


def ensure_tf_type(obj, fake_input_layer=None, name=None):
    """
    Convert to Keras Constant if needed
    :param obj: numpy / tf type
    :param fake_input_layer: fake input layer to add constant
    :return: tf type
    """
    if is_numpy(obj):
        if obj.dtype == np.int64:
            obj = np.int32(obj)

        def target_layer(_, inp=obj, dtype=obj.dtype.name):
            import numpy as np
            import tensorflow as tf
            if not isinstance(inp, (np.ndarray, np.generic)):
                inp = np.array(inp, dtype=dtype)
            return tf.constant(inp, dtype=inp.dtype)

        lambda_layer = keras.layers.Lambda(target_layer, name=name)
        return lambda_layer(fake_input_layer)
    else:
        return obj


def check_torch_keras_error(model, k_model, input_np, epsilon=1e-5, change_ordering=False):
    """
    Check difference between Torch and Keras models
    :param model: torch model
    :param k_model: keras model
    :param input_np: input data as numpy array or list of numpy array
    :param epsilon: allowed difference
    :param change_ordering: change ordering for keras input
    :return: actual difference
    """
    from torch.autograd import Variable
    import torch

    if isinstance(input_np, np.ndarray):
        input_np = [input_np]

    input_var = [Variable(torch.FloatTensor(i)) for i in input_np]
    pytorch_output = model(*input_var)
    if not isinstance(pytorch_output, tuple):
        pytorch_output = [pytorch_output.data.numpy()]
    else:
        pytorch_output = [p.data.numpy() for p in pytorch_output]

    if change_ordering:
        # change image data format
        _input_np = []
        for i in input_np:
            axes = list(range(len(i.shape)))
            axes = axes[0:1] + axes[2:] + axes[1:2]
            _input_np.append(np.transpose(i, axes))
        input_np = _input_np

        # run keras model
        keras_output = k_model.predict(input_np)
        if not isinstance(keras_output, list):
            keras_output = [keras_output]

        # change image data format
        _koutput = []
        for k in keras_output:
            axes = list(range(len(k.shape)))
            axes = axes[0:1] + axes[-1:] + axes[1:-1]
            _koutput.append(np.transpose(k, axes))
        keras_output = _koutput
    else:
        keras_output = k_model.predict(input_np)
        if not isinstance(keras_output, list):
            keras_output = [keras_output]

    max_error = 0
    for p, k in zip(pytorch_output, keras_output):
        error = np.max(np.abs(p - k))
        if error > max_error:
            max_error = error

    assert max_error < epsilon
    return max_error
