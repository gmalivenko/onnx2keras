# code to proprely load data here: https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/
import onnx
import numpy as np
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import urllib


def test_openclip():
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/example-datasets-47ml982d/openclip/openclip.onnx",
        "openclip.onnx")
    onnx_model = onnx.load('openclip.onnx')
    keras_model = onnx_to_keras(onnx_model, ["pixel_values"], name_policy='attach_weights_name',
                                allow_partial_compilation=False)
    keras_model = keras_model.converted_model
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
    tf_preds = final_model(np.random.random((1, 224, 224, 3)))

