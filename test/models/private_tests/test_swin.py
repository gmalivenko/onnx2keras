import onnxruntime as ort
import numpy as np
import onnx
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import tensorflow as tf
from test.models.private_tests.aws_utils import aws_s3_download
import pytest


@pytest.mark.parametrize('aws_s3_download', [["swin/", "swin/", False]], indirect=True)
def test_swin(aws_s3_download):
    model_path = f'{aws_s3_download}/swin_v2_t.onnx'
    inpt = np.load(f'{aws_s3_download}/input.npy')
    result = np.load(f'{aws_s3_download}/output.npy')
    onnx_model = onnx.load(model_path)
    keras_model = onnx_to_keras(onnx_model, ['input'], name_policy='attach_weights_name',
                                allow_partial_compilation=False).converted_model
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
    res = final_model(inpt)
    mean_error = (res-result).numpy().__abs__().mean()
    max_error = (res-result).numpy().__abs__().max()
    eps = 5e-6
    assert mean_error < eps
    assert max_error < eps
