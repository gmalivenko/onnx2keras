# code to proprely load data here: https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/
import onnx
import numpy as np
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import urllib


def test_raft_ci():
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/example-datasets-47ml982d/raft/raft.onnx",
        "raft.onnx")
    onnx_model = onnx.load('raft.onnx')
    keras_model = onnx_to_keras(onnx_model, ['onnx::Div_0', 'onnx::Div_1'], name_policy='attach_weights_name',
                                allow_partial_compilation=False)
    keras_model = keras_model.converted_model
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
    first_im = np.random.random((440, 1024, 3))[None, ...]
    second_im = np.random.random((440, 1024, 3))[None, ...]
    tf_preds = final_model([first_im, second_im])
