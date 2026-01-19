from typing import List, Union, Callable

import numpy as np
import keras
from keras.engine.keras_tensor import KerasTensor
from keras_data_format_converter import convert_channels_first_to_last
import tensorflow as tf
from.tfops_funcs import tf_reshape

ONNX_ELEM_TO_TF_TYPE = {
    1: tf.float32,
    2: tf.uint8,
    3: tf.int8,
    4: tf.uint16,
    5: tf.int16,
    6: tf.int32,
    7: tf.int64,
    8: tf.string,
    9: tf.bool,
    10: tf.float16,
    11: tf.double,
    12: tf.uint32,
    13: tf.uint64,
    14: tf.complex64,
    15: tf.complex128,
    16: tf.bfloat16
}


def is_numpy(obj):
    """
    Check of the type is instance of numpy array
    :param obj: object to check
    :return: True if the object is numpy-type array.
    """
    return isinstance(obj, (np.ndarray, np.generic))


def ensure_tf_type(obj, name="Const"):
    import numpy as np
    import tensorflow as tf
    """
    Convert to Keras Constant if needed
    :param obj: numpy / tf type
    :param fake_input_layer: fake input layer to add constant
    :return: tf type
    """
    if is_numpy(obj): # TF < v1.16 assumes all ints are int32 and all floats are float32
        if obj.dtype == np.int64:
            obj = np.int32(obj)
        return tf.constant(obj, name=name)
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
        if len(input_np) == 1:
            input_np = input_np[0]
        keras_output = k_model(input_np)
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


def unsqueeze_tensors_of_rank_one(tensor_list, axis: int, name: str):
    """
    Adjusts the ranks of tensors of rank 1 in a given list to match the maximum rank by adding dummy dimensions
    e.g., for input tensors shapes [(2,), (1, 4)] the unsqueezed tensors are [(1, 2), (1, 4)]

    Args:
        tensor_list (list): A list of tensors.

    Returns:
        list: A new list of tensors with adjusted ranks to match the maximum rank.
              If all tensors in the input list already have the same rank, the original list is returned.
    """
    ranks = [tensor.shape.rank for tensor in tensor_list]
    max_rank = max(ranks)

    if len(set(ranks)) == 1:
        return tensor_list
    elif len(set(ranks)) > 2:
        raise ValueError(f"More than 2 different ranks detected, broadcasting is ambiguous.\n"
                         f"Check the outputs of layers: \n" + '\n'.join([tensor.name for tensor in tensor_list]))

    unsqueezed_tensors = []
    for tensor in tensor_list:
        tensor_rank = tensor.shape.rank
        if tensor_rank == 1:
            rank_diff = max_rank - 1
            new_shape = [1] * axis + list(tensor.shape) + [1] * (rank_diff - axis)
            unsqueezed_tensor = tf_reshape(tensor, new_shape, tf_name=f"{name}_rank_one_reshape")
            unsqueezed_tensors.append(unsqueezed_tensor)
        else:
            unsqueezed_tensors.append(tensor)

    return unsqueezed_tensors


def ensure_float(value):
    if isinstance(value, (list, np.ndarray)):
        return float(value[0])
    elif isinstance(value, tf.Tensor):
        return float(value.numpy().item())
    else:
        return float(value)
