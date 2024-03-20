import numpy as np
import onnx
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import tensorflow as tf
import onnxruntime as ort
from onnx2kerastl.customonnxlayer import onnx_custom_objects_map
from test.models.private_tests.aws_utils import aws_s3_download
import pytest


@pytest.mark.parametrize('aws_s3_download', [["clip/", "clip/", False]], indirect=True)
def test_clip_model(aws_s3_download):
    # declare paths
    onnx_model_path = f'{aws_s3_download}/clip.onnx'
    save_model_path = f'{aws_s3_download}/clip.h5'
    token_ids = [320 for _ in range(75)]
    token_ids.insert(0, 49406)
    token_ids.append(49407)
    input_data = {'input_ids': np.asarray(token_ids).astype(np.int64).reshape(1, -1),
                  'attention_mask': np.ones((1, 77)).astype(np.int64)}
    # load onnx model
    onnx_model = onnx.load(onnx_model_path)
    # extract feature names from the model
    input_features = list(input_data.keys())
    # convert onnx model to keras
    keras_model = onnx_to_keras(onnx_model, input_names=input_features, name_policy='attach_weights_name',
                                allow_partial_compilation=False).converted_model
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True, verbose=True)
    # final_model = tf.keras.models.Model(inputs=final_model.input, outputs=final_model.layers[-2].output)
    keras_output = keras_model(input_data)
    final_model.save(save_model_path)
    ort_session = ort.InferenceSession(onnx_model_path, providers=ort.get_available_providers()[0])
    onnx_outputs = ort_session.run(None, input_data)
    loaded_keras_model = tf.keras.models.load_model(save_model_path, custom_objects=onnx_custom_objects_map)
    loaded_keras_outputs = loaded_keras_model(input_data)
    onnx_embedding = onnx_outputs[1]
    keras_embedding = loaded_keras_outputs[1].numpy()
    is_same = np.allclose(onnx_embedding, keras_embedding, 1e-3)
