# code to proprely load data here: https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/
import onnx
import onnxruntime as ort
import numpy as np
import pytest
from test.models.private_tests.aws_utils import aws_s3_download

from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last

@pytest.mark.parametrize('aws_s3_download', [["interfuser/", "interfuser/", False]], indirect=True)
def test_interfuser(aws_s3_download):
    onnx_path = f'{aws_s3_download}/interfuser_planKD_26_3M_256.onnx'
    onnx_model = onnx.load(onnx_path)
    output_names = ['out0', 'out1', 'out2', 'out3', 'out4', 'out5']
    input_keys = ['lidar', 'measurements',  'rgb','rgb_center','rgb_left', 'rgb_right', 'target_point']
    shapes = [
        (1, 3, 224, 224),
        (1, 7),
        (1, 3, 256, 256),
        (1, 3, 128, 128),
        (1, 3, 256, 256),
        (1, 3, 256, 256),
        (1, 2)
    ]

    inputs = tuple(np.random.rand(*shape).astype(np.float32) for shape in shapes)
    inputs = {key: inp_ for key, inp_ in zip(input_keys, inputs)}
    keras_model = onnx_to_keras(onnx_model, input_keys, name_policy='attach_weights_name'
                                , allow_partial_compilation=False)
    keras_model = keras_model.converted_model
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=False)
    ort_session = ort.InferenceSession(onnx_path)
    onnx_res = ort_session.run(
        output_names,
        input_feed=inputs)[0]
    keras_preds = final_model(inputs)[0]
    assert np.abs(keras_preds - onnx_res).max() < 1e-04
