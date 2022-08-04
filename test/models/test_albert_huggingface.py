import onnx
import pytest
import tensorflow as tf
from transformers import AlbertTokenizer, TFAlbertModel
from transformers.onnx import FeaturesManager
from pathlib import Path
from transformers.onnx import export, OnnxConfig
import numpy as np
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last


@pytest.mark.slow
def test_albert_huggingface():
    onnx_path = 'model.onnx'
    model_name = "albert-base-v2"
    model_name_for_features = "albert"
    model = TFAlbertModel.from_pretrained(model_name)
    tokenizer = AlbertTokenizer.from_pretrained(model_name)
    real_inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    OnnxConfig.default_fixed_batch = 1
    albert_features = list(FeaturesManager.get_supported_features_for_model_type(model_name_for_features).keys())
    print(albert_features)
    onnx_path = Path(onnx_path)
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature='default')
    onnx_config = model_onnx_config(model.config)
    onnx_inputs, onnx_outputs = export(tokenizer, model, onnx_config, onnx_config.default_onnx_opset, onnx_path)
    onnx_model = onnx.load(onnx_path)
    keras_model = onnx_to_keras(onnx_model, ['input_ids', 'token_type_ids', 'attention_mask'],
                                input_types=[tf.int32, tf.int32, tf.float32])
    input_np = [real_inputs['input_ids'],
                real_inputs['token_type_ids'],
                real_inputs['attention_mask']]
    out = model(real_inputs)
    flipped_model = convert_channels_first_to_last(keras_model, [])
    flipped_otpt = flipped_model(input_np)
    assert np.abs((out['last_hidden_state']-flipped_otpt[0])).max() < 1e-04
    assert np.abs((out['pooler_output']-flipped_otpt[1])).max() < 1e-04