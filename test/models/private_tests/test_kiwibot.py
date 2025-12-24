import numpy as np
import onnx
import onnxruntime as ort
from onnx2kerastl import onnx_to_keras
import tensorflow as tf
from keras_data_format_converter import convert_channels_first_to_last
from onnx2kerastl.customonnxlayer import onnx_custom_objects_map
import pytest

from test.models.private_tests.aws_utils import aws_s3_download

@pytest.mark.parametrize('aws_s3_download', [["kiwibot/", "kiwibot/", False]], indirect=True)
def test_kiwibot(aws_s3_download):
    onnx_model_path = f'{aws_s3_download}/model.onnx'
    save_model_path = f'{aws_s3_download}/model.h5'

    input_data = np.random.uniform(0, 255, (1, 360, 640, 3)).astype(np.uint8)
    # load onnx model
    onnx_model = onnx.load(onnx_model_path)
    # extract feature names from the model
    input_features = [inp.name for inp in onnx_model.graph.input]
    # convert onnx model to keras
    keras_model = onnx_to_keras(onnx_model, input_names=input_features,
                                name_policy='attach_weights_name', allow_partial_compilation=False).converted_model

    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True,
                                                 verbose=True)

    final_model.save(save_model_path)

    loaded_keras_model = tf.keras.models.load_model(save_model_path, custom_objects=onnx_custom_objects_map)
    keras_output = loaded_keras_model(input_data)
    keras_output_np = [output.numpy().transpose((0, 2, 1)) for output in keras_output]

    onnx_session = ort.InferenceSession(onnx_model_path)
    onnx_output = onnx_session.run(None, {'input_0': input_data.transpose((0, 3, 1, 2)).astype(np.float32)})

    # masks after softmax
    assert np.abs(keras_output_np[1] - onnx_output[1]).max() < 1e-1
    assert np.abs(keras_output_np[2] - onnx_output[2]).max() < 1e-1
    # mask after pixel class prediction
    assert np.abs(keras_output_np[0] == onnx_output[0]).sum() / np.prod(keras_output_np[0].shape) > 0.99
