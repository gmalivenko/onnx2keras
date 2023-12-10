import io
from transformers import AutoImageProcessor, GLPNForDepthEstimation, DPTForDepthEstimation
import onnx
import torch
import requests
from PIL import Image
import numpy as np
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last



def remove_last_two_nodes(onnx_model):

    # Remove the last two nodes from the graph
    for i in range(6):
        onnx_model.graph.node.pop()

    # Set the third-to-last node as the new output
    new_output_name = onnx_model.graph.node[-1].output[0]

    # Extend the graph's output with the new output
    onnx_model.graph.output.extend(onnx_model.graph.value_info)
    onnx_model.graph.output[-1].name = new_output_name
    return onnx_model


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
    onnx_model = remove_last_two_nodes(onnx_model)
    keras_model = onnx_to_keras(onnx_model, ['input.1'], name_policy='attach_weights_name')
    keras_model = keras_model.converted_model
    permuted_inputs = np.swapaxes(np.swapaxes(pixel_values, 1, 2), 2, 3)
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
    model = model.eval()
    this_pred = model(torch.Tensor(pixel_values))
    keras_preds = final_model(permuted_inputs)
    torch_pred = this_pred['predicted_depth']
    assert np.abs(keras_preds[..., 0] - torch_pred.detach().numpy()).max() < 1e-04