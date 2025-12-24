import numpy as np
import onnx
from onnx2kerastl import onnx_to_keras
import tensorflow as tf
from keras_data_format_converter import convert_channels_first_to_last
from onnx2kerastl.customonnxlayer import onnx_custom_layers
import pytest
from test.models.private_tests.aws_utils import aws_s3_download


@pytest.mark.parametrize('aws_s3_download', [["mmdet_convnext/", "mmdet_convnext/", False]], indirect=True)
def test_mmdet_convnext(aws_s3_download):
    onnx_model_path = f'{aws_s3_download}/simplified_por_convnext.onnx'
    save_model_path = f'{aws_s3_download}/simplified_por_convnext.h5'

    input_data = np.random.uniform(0, 1, (1, 480, 640, 3)).astype(np.float32)
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

    loaded_keras_model = tf.keras.models.load_model(save_model_path, custom_objects=onnx_custom_layers)
    keras_output = loaded_keras_model(input_data)
