import onnxruntime as ort
import numpy as np
import onnx
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import tensorflow as tf
from test.models.private_tests.aws_utils import aws_s3_download
import pytest
import onnxruntime as rt


@pytest.mark.parametrize('aws_s3_download', [["gps/", "gps/", False]], indirect=True)
def test_gps(aws_s3_download):
    onnx_model_path = f'{aws_s3_download}/gps_750_v1.onnx'
    onnx_model = onnx.load(onnx_model_path)
    keras_model = onnx_to_keras(onnx_model, ['images', 'gps', 'masks'], name_policy='attach_weights_name',
                                allow_partial_compilation=False).converted_model
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
    data = np.random.random((1, 11, 3, 224, 224))
    gps = np.random.random((1, 10, 2))
    masks = np.ones((1, 11))
    masks[:, :8] = 0
    res = final_model([data.transpose([0, 2, 3, 4, 1]), gps.transpose([0,2,1]), masks])
    sess = rt.InferenceSession(onnx_model_path)
    input_name_1 = sess.get_inputs()[0].name
    input_name_2 = sess.get_inputs()[1].name
    input_name_3 = sess.get_inputs()[2].name
    label_name = sess.get_outputs()[0].name
    pred = sess.run([label_name],
                    {input_name_1: data.astype(np.float32), input_name_2: gps.astype(np.float32),
                     input_name_3: masks.astype(np.float32)})
    assert (pred[0] - res).numpy().__abs__().max() < 2e-5
