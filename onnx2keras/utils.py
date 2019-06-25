import numpy as np
import keras


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


def ensure_tf_type(obj, fake_input_layer=None):
    """
    Convert to Keras Constant if needed
    :param obj: numpy / tf type
    :param fake_input_layer: fake input layer to add constant
    :return: tf type
    """
    if is_numpy(obj):
        if obj.dtype == np.int64:
            obj = np.int32(obj)

        def target_layer(_, inp=obj):
            import tensorflow as tf
            return tf.constant(inp, dtype=inp.dtype, verify_shape=True)

        lambda_layer = keras.layers.Lambda(target_layer)
        return lambda_layer(fake_input_layer)
    else:
        return obj


def check_torch_keras_error(model, k_model, input_np, epsilon=1e-5):
    """
    Check difference between Torch and Keras models
    :param model: torch model
    :param k_model: keras model
    :param input_np: input data
    :param epsilon: allowed difference
    :return: actual difference
    """
    from torch.autograd import Variable
    import torch

    input_var = Variable(torch.FloatTensor(input_np))
    pytorch_output = model(input_var).data.numpy()
    keras_output = k_model.predict(input_np)

    error = np.max(pytorch_output - keras_output)

    assert error < epsilon
    return error