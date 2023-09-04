import numpy as np
import onnx
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import urllib


def test_nms():
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/example-datasets-47ml982d/onnx2kerras/nms/nms_v2.onnx",
        "nms_v2.onnx")
    onnx_model = onnx.load('nms_v2.onnx')
    keras_model = onnx_to_keras(onnx_model, ['boxes', 'scores'], name_policy='attach_weights_name')
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/example-datasets-47ml982d/onnx2kerras/nms/nms_in_boxes.npy",
        "nms_in_boxes.npy")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/example-datasets-47ml982d/onnx2kerras/nms/nms_in_scores.npy",
        "nms_in_scores.npy")
    boxes = np.load('nms_in_boxes.npy')
    scores = np.load('nms_in_scores.npy')
    keras_res = final_model([boxes.swapaxes(1, 2), scores])
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/example-datasets-47ml982d/onnx2kerras/nms/nms_out.npy",
        "nms_out.npy")
    results = np.load('nms_out.npy')
    assert len(set(keras_res[..., 0].numpy()).intersection(set(results))) == len(results) == keras_res.shape[0]
