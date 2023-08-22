# code to proprely load data here: https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/
from transformers import AutoImageProcessor, GLPNForDepthEstimation, DPTForDepthEstimation
import onnx
import torch
import requests
from PIL import Image
import numpy as np
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
from packaging import version
from collections import OrderedDict
from typing import Mapping
import urllib


def test_glpn():
    model_name = 'glpn'
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti")
    model = model.eval()
    # torch.onnx.export(n_model, torch.ones(1, 3, 13, 182, 182), 'x3d.onnx', export_params=True, input_names=['input'],
    #                   output_names=['output'],
    #                   dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
    #                                 'output': {0: 'batch_size'}}) - this requires an earlier version of tensorflow
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image_processor = AutoImageProcessor.from_pretrained("vinvino02/glpn-kitti")
    inputs = image_processor(images=image, return_tensors="pt")
    x = inputs.data['pixel_values']
    print(x.shape)
    torch.onnx.export(model, x, 'glpn.onnx')
    onnx_model = onnx.load('glpn.onnx')

    keras_model = onnx_to_keras(onnx_model, ['input.1'], name_policy='attach_weights_name')
    permuted_inputs = np.swapaxes(np.swapaxes(np.swapaxes(inputs, 0, 1), 1, 2), 2, 3)
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
    model = model.eval()
    this_pred = model(torch.Tensor(inputs)[None, ...])
    keras_preds = final_model(permuted_inputs[None, ...])

    assert np.abs(keras_preds - this_pred.detach().numpy()).max() < 1e-04


# def test_dpt():
#     model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
#     model = model.eval()
#     # torch.onnx.export(n_model, torch.ones(1, 3, 13, 182, 182), 'x3d.onnx', export_params=True, input_names=['input'],
#     #                   output_names=['output'],
#     #                   dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
#     #                                 'output': {0: 'batch_size'}}) - this requires an earlier version of tensorflow
#     url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#     image = Image.open(requests.get(url, stream=True).raw)
#
#     image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
#     inputs = image_processor(images=image, return_tensors="pt")
#     x = inputs.data['pixel_values']
#
#
#     torch.onnx.export(model, x, 'dpt.onnx')
#     onnx_model = onnx.load('dpt.onnx')
#
#     keras_model = onnx_to_keras(onnx_model, ['input.1'], name_policy='attach_weights_name')
#     permuted_inputs = np.swapaxes(np.swapaxes(np.swapaxes(inputs, 0, 1), 1, 2), 2, 3)
#     final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
#     model = model.eval()
#     this_pred = model(torch.Tensor(inputs)[None, ...])
#     keras_preds = final_model(permuted_inputs[None, ...])
#
#     assert np.abs(keras_preds - this_pred.detach().numpy()).max() < 1e-04
