import numpy as np
from tensorflow import keras
from keras_data_format_converter import convert_channels_first_to_last


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


def check_torch_keras_error(model, k_model, input_np, epsilon=1e-5, change_ordering=False,
                            should_transform_inputs=False):
    """
    Check difference between Torch and Keras models
    :param model: torch model
    :param k_model: keras model
    :param input_np: input data as numpy array or list of numpy array
    :param epsilon: allowed difference
    :param change_ordering: change ordering for keras input
    :param should_transform_inputs: default False, set to True for converting channel first inputs to  channel last format
    :return: actual difference

    """
    from torch.autograd import Variable
    import torch

    if isinstance(input_np, np.ndarray):
        input_np = [input_np.astype(np.float32)]

    input_var = [Variable(torch.FloatTensor(i)) for i in input_np]
    pytorch_output = model(*input_var)
    if isinstance(pytorch_output, dict):
        pytorch_output = [p.data.numpy() for p in list(pytorch_output.values())]
    elif isinstance(pytorch_output, (tuple, list)):
        pytorch_output = [p.data.numpy() for p in pytorch_output]
    else:
        pytorch_output = [pytorch_output.data.numpy()]

    if change_ordering:
        # change image data format

        # to proper work with Lambda layers that transpose weights based on image_data_format
        keras.backend.set_image_data_format("channels_last")

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

        # change image data format if output shapes are different (e.g. the same for global_avgpool2d)
        _koutput = []
        for i, k in enumerate(keras_output):
            if k.shape != pytorch_output[i].shape:
                axes = list(range(len(k.shape)))
                axes = axes[0:1] + axes[-1:] + axes[1:-1]
                k = np.transpose(k, axes)
            _koutput.append(k)
        keras_output = _koutput
    else:
        inputs_to_transpose = []
        if should_transform_inputs:
            inputs_to_transpose = [k_input.name for k_input in k_model.inputs]

            _input_np = []
            for i in input_np:
                axes = list(range(len(i.shape)))
                axes = axes[0:1] + axes[2:] + axes[1:2]
                _input_np.append(np.transpose(i, axes))
            input_np = _input_np

        k_model = convert_channels_first_to_last(k_model, inputs_to_transpose)
        keras_output = k_model(*input_np)
        if not isinstance(keras_output, list):
            keras_output = [keras_output]

        _koutput = []
        for i, k in enumerate(keras_output):
            if k.shape != pytorch_output[i].shape:
                axes = list(range(len(k.shape)))
                axes = axes[0:1] + axes[-1:] + axes[1:-1]
                k = np.transpose(k, axes)
            _koutput.append(k)
        keras_output = _koutput

    max_error = 0
    for p, k in zip(pytorch_output, keras_output):
        error = np.max(np.abs(p - k))
        np.testing.assert_allclose(p, k, atol=epsilon, rtol=0.0)
        if error > max_error:
            max_error = error

    return max_error
