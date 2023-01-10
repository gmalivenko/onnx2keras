import onnx
import tensorflow as tf
from transformers.onnx import FeaturesManager
from pathlib import Path
from transformers import ViTFeatureExtractor, ViTModel
from transformers.onnx import export, OnnxConfig
import numpy as np
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
from packaging import version
from collections import OrderedDict
from typing import Mapping
from functools import partial


class ViTOnnxConfig(OnnxConfig):

    torch_onnx_minimum_version = version.parse("1.11")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {}),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        return 1e-4


def test_vit_huggingface():
    save_model = True
    onnx_path = 'vit_huggingface.onnx'
    model_name = "google/vit-base-patch16-224-in21k"
    model = ViTModel.from_pretrained(model_name)
    tokenizer = ViTFeatureExtractor.from_pretrained(model_name)
    OnnxConfig.default_fixed_batch = 1
    OnnxConfig.default_fixed_sequence = 3
    OnnxConfig.default_batch_size = 1
    OnnxConfig.default_sequence_length = 3
    if save_model:
        onnx_path = Path(onnx_path)
        model_onnx_config = partial(ViTOnnxConfig.from_model_config, task='default')
        onnx_config = model_onnx_config(model.config)
        onnx_inputs, onnx_outputs = export(tokenizer, model, onnx_config, onnx_config.default_onnx_opset, onnx_path)
    onnx_model = onnx.load(onnx_path)
    # keras_model = onnx_to_keras(onnx_model, ['pixel_values'], batch_size=1)
    keras_model = onnx_to_keras(onnx_model, ['pixel_values'])
    final_model = convert_channels_first_to_last(keras_model, ['pixel_values'])
    tokens = tokenizer(np.ones([300, 300, 3], dtype=np.uint8), return_tensors="pt")
    pt_res = model(**tokens)
    keras_res = final_model(tokens['pixel_values'].numpy().reshape(1, 224, 224, 3))
    assert np.abs(pt_res[1].detach().numpy()-keras_res[1]).max() < 1e-04