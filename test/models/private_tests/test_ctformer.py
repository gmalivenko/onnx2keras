# code to proprely load data here: https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/
import onnx
import onnxruntime as ort
import numpy as np
from test.models.private_tests.aws_utils import aws_s3_download

import pytest
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last

@pytest.mark.parametrize('aws_s3_download', [["ctformer/", "ctformer/", False]], indirect=True)
def test_ctformer(
    aws_s3_download) -> None:
    onnx_path = f'{aws_s3_download}/ctformer.onnx'
    onnx_model = onnx.load(onnx_path)

    rng = np.random.default_rng(seed=42)
    input_array = np.random.rand(1, 1, 64, 64).astype(np.float32)
    input_tensor = input_array

    keras_model = onnx_to_keras(onnx_model, input_names=['input'], name_policy='attach_weights_name'
                                , allow_partial_compilation=False)
    keras_model = keras_model.converted_model
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=False)
    ort_session = ort.InferenceSession(onnx_path )
    onnx_res = ort_session.run(
        None,
        {"input": input_array},
    )[0]
    keras_preds = final_model(input_array)[0]
    assert np.abs(keras_preds - onnx_res).max() < 1e-04
