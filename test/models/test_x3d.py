# code to proprely load data here: https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/
import onnx
# from transformers.onnx import export, OnnxConfig
import numpy as np
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
from packaging import version
from collections import OrderedDict
from typing import Mapping
import urllib


def test_x3d():
    import torch
    model_name = 'x3d_s'
    model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
    model = model.eval()
    # torch.onnx.export(n_model, torch.ones(1, 3, 13, 182, 182), 'x3d.onnx', export_params=True, input_names=['input'],
    #                   output_names=['output'],
    #                   dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
    #                                 'output': {0: 'batch_size'}}) - this requires an earlier version of tensorflow
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/example-datasets-47ml982d/x3d_tests/x3d.onnx",
        "x3d_s.onnx")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/example-datasets-47ml982d/x3d_tests/inputs.npy",
        "x3d_input.npy")
    onnx_model = onnx.load('x3d_s.onnx')
    inputs = np.load('x3d_input.npy')
    keras_model = onnx_to_keras(onnx_model, ['input'], name_policy='attach_weights_name'
                                , allow_partial_compilation=False)
    keras_model = keras_model.converted_model
    permuted_inputs = np.swapaxes(np.swapaxes(np.swapaxes(inputs, 0, 1), 1, 2), 2, 3)
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
    model = model.eval()
    this_pred = model(torch.Tensor(inputs)[None, ...])
    keras_preds = final_model(permuted_inputs[None, ...])
    assert np.abs(keras_preds - this_pred.detach().numpy()).max() < 1e-04
