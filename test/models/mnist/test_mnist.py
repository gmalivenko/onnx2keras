import numpy as np
import onnx
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import tensorflow as tf
import onnxruntime as ort
import pytest
import pathlib

def test_mnist_mode():
    # load onnx model
    dir = pathlib.Path(__file__).parent.resolve()
    model_path = f"{dir}/mnist-12.onnx"
    onnx_model = onnx.load(model_path)
    # convert onnx model to keras
    keras_model = onnx_to_keras(onnx_model, input_names=["Input3"], name_policy='attach_weights_name',
                                allow_partial_compilation=False).converted_model
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=False, verbose=True)
    ort_session = ort.InferenceSession(model_path)
    img = np.random.random((1,1,28,28))
    keras_output = final_model(img)
    onnx_outputs = ort_session.run(None, {"Input3": img.astype(np.float32)})
    is_same = np.allclose(onnx_outputs, keras_output, 1e-6)
