import onnxruntime as ort
import numpy as np
import onnx
from PIL import Image

from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
from test.models.private_tests.aws_utils import aws_s3_download
import pytest


@pytest.mark.parametrize('aws_s3_download', [["pick/", "pick/", False]], indirect=True)
def test_pick_1(aws_s3_download):
    model_path = f'{aws_s3_download}/sample-gen-grasp-v2.onnx'
    image_path = f'{aws_s3_download}/pick_image_downsample_4x.jpg'
    image = Image.open(image_path)

    # Resize the image to 512x512
    image_resized = image.resize((512, 512))

    # Convert the image to a NumPy array
    image_array = np.array(image_resized)

    image_array = np.transpose(image_array, (2, 0, 1)).astype(np.float32) / 255
    rgb_input = np.expand_dims(image_array, axis=0)

    pick_parameters = np.array([200, 300, 0, 0, 1, 0.2, 1, 0, 0, 0, 0, 0, 0]).astype(np.float32)
    pick_parameters = np.expand_dims(pick_parameters, axis=0)

    session = ort.InferenceSession(model_path)

    # Get the names of the input and output nodes
    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]

    onnx_model = onnx.load(model_path)
    keras_model = onnx_to_keras(onnx_model, input_names, name_policy='attach_weights_name',
                                allow_partial_compilation=False).converted_model
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=False)
    res = final_model([pick_parameters, rgb_input])

    res_onnx = session.run(output_names, {input_names[0]: rgb_input, input_names[1]: pick_parameters})

    assert abs((res[0].numpy() - res_onnx[0])[0][0]) < 1e-3
    assert np.sum(np.abs(res[1].numpy() - res_onnx[1])) < 2e-3
    assert np.sum(np.abs(res[2].numpy() - res_onnx[2])) < 4e-3

