import io
from transformers import AutoImageProcessor, GLPNForDepthEstimation, DPTForDepthEstimation
import onnx
import torch
import requests
from PIL import Image
import numpy as np
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last



def test_glpn():
    import sys
    sys.setrecursionlimit(10000)

    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti")
    model = model.eval()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image_processor = AutoImageProcessor.from_pretrained("vinvino02/glpn-kitti")
    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs.data['pixel_values']

    # torch.onnx.export(model, pixel_values, 'glpn.onnx')
    # onnx_model = onnx.load('glpn.onnx')

    temp_f = io.BytesIO()
    torch.onnx.export(model, pixel_values, temp_f)
    temp_f.seek(0)
    onnx_model = onnx.load(temp_f)

    keras_model = onnx_to_keras(onnx_model, ['input.1'], name_policy='attach_weights_name')

    permuted_inputs = np.swapaxes(np.swapaxes(pixel_values, 1, 2), 2, 3)
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
    model = model.eval()
    this_pred = model(torch.Tensor(pixel_values))
    keras_preds = final_model(permuted_inputs)
    keras_preds = np.swapaxes(keras_preds, 1, 2)
    torch_pred = this_pred['predicted_depth']
    assert np.abs(keras_preds - torch_pred.detach().numpy()).max() < 1e-04

    final_model.save('glpn.h5')