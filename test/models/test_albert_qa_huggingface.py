import onnx
import pytest
import tensorflow as tf
from transformers import AlbertTokenizer, TFAlbertForQuestionAnswering
from transformers.onnx import FeaturesManager
from pathlib import Path
from transformers.onnx import export, OnnxConfig
import numpy as np
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last


@pytest.mark.slow
def test_albert_qa_huggingface():
    save_model = True
    onnx_path = 'model.onnx'
    model_name_for_features = "albert"
    tokenizer = AlbertTokenizer.from_pretrained("vumichien/albert-base-v2-squad2")
    model = TFAlbertForQuestionAnswering.from_pretrained("vumichien/albert-base-v2-squad2")
    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    inputs = tokenizer(question, text, return_tensors="tf")
    outputs = model(**inputs)
    answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
    answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
    predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
    answer = tokenizer.decode(predict_answer_tokens)
    OnnxConfig.default_fixed_batch = 1
    OnnxConfig.default_fixed_sequence = 14
    albert_features = list(FeaturesManager.get_supported_features_for_model_type(model_name_for_features).keys())
    print(albert_features)
    onnx_path = Path(onnx_path)
    if save_model == True:
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature='question-answering')
        onnx_config = model_onnx_config(model.config)
        onnx_inputs, onnx_outputs = export(tokenizer, model, onnx_config, onnx_config.default_onnx_opset, onnx_path)
    onnx_model = onnx.load(onnx_path)
    keras_model = onnx_to_keras(onnx_model, ['input_ids', 'token_type_ids', 'attention_mask'],
                                input_types=[tf.int32, tf.int32, tf.float32])
    input_np = [inputs['input_ids'],
                inputs['token_type_ids'],
                inputs['attention_mask']]
    out = model(inputs)
    flipped_model = convert_channels_first_to_last(keras_model, [])
    flipped_otpt = flipped_model(input_np)
    assert np.abs((out[0]-flipped_otpt[1])).max() < 1e-04
    assert np.abs((out[1]-flipped_otpt[0])).max() < 1e-04
