import io

import onnx
import torch
from keras.layers import Lambda
from keras.models import Model

from onnx2kerastl import onnx_to_keras
from onnx2kerastl.utils import check_torch_keras_error

NP_SEED = 42

class LambdaLayerException(Exception):
    pass


def torch2keras(model: torch.nn.Module, input_variable, verbose=True, change_ordering=False):
    if isinstance(input_variable, (tuple, list)):
        input_variable = tuple(torch.FloatTensor(var) for var in input_variable)
        input_names = [f'test_in{i}' for i, _ in enumerate(input_variable)]
    else:
        input_variable = torch.FloatTensor(input_variable)
        input_names = ['test_in']

    temp_f = io.BytesIO()
    torch.onnx.export(model, input_variable, temp_f, verbose=verbose, input_names=input_names,
                      output_names=['test_out'])
    temp_f.seek(0)
    onnx_model = onnx.load(temp_f)
    k_model = onnx_to_keras(onnx_model, input_names, change_ordering=change_ordering)
    return k_model


def convert_and_test(model: torch.nn.Module,
                     input_variable,
                     verbose=True,
                     change_ordering=False,
                     epsilon=1e-5,
                     should_transform_inputs=False):
    k_model = torch2keras(model, input_variable, verbose=verbose, change_ordering=change_ordering)
    error = test_conversion(model, k_model, input_variable, change_ordering=change_ordering, epsilon=epsilon,
                            should_transform_inputs=should_transform_inputs)
    return error


def test_conversion(onnx_model, k_model, input_variable, change_ordering=False, epsilon=1e-5,
                    should_transform_inputs=False):
    error = check_torch_keras_error(onnx_model, k_model, input_variable, change_ordering=change_ordering, epsilon=epsilon,
                                    should_transform_inputs=should_transform_inputs)
    if is_lambda_layers_exist(k_model):
        raise LambdaLayerException("Found Lambda layers")
    return error


def is_lambda_layers_exist(model: Model):
    return any(isinstance(layer, Lambda) for layer in model.layers)
