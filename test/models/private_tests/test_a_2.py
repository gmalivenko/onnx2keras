import onnxruntime as ort
import numpy as np
import onnx
from PIL import Image
import  tensorflow as tf
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
from test.models.private_tests.aws_utils import aws_s3_download
import pytest


@pytest.mark.parametrize('aws_s3_download', [["pick_2/", "pick_2/", False]], indirect=True)
def test_pick_2(aws_s3_download):
    model_path = f'{aws_s3_download}/new_2.onnx'

    session = ort.InferenceSession(model_path)

    # Get the names of the input and output nodes
    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]

    onnx_model = onnx.load(model_path)
    keras_model = onnx_to_keras(onnx_model, input_names, name_policy='attach_weights_name',
                                allow_partial_compilation=False).converted_model
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=False)
    img =  np.random.random((1, 3, 512, 512)).astype(np.float32)
    pick_param = np.random.random((13)).astype(np.float32)
    pick_parameters = np.expand_dims(pick_param, axis=0)
    res = final_model([tf.convert_to_tensor(pick_parameters), tf.convert_to_tensor(img)])

    res_onnx = session.run(output_names, {input_names[0]: img, input_names[1]: pick_parameters})

    assert abs((res[0].numpy() - res_onnx[0])[0][0]) < 5e-2
    assert np.sum(np.abs(res[1].numpy() - res_onnx[1])) < 5e-2
    assert np.sum(np.abs(res[2].numpy() - res_onnx[2])) < 0.5

