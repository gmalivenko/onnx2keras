import onnx
import pytest
import tensorflow as tf
# import torch
from transformers import BertTokenizer, BertModel
from transformers.onnx import FeaturesManager
from pathlib import Path
from transformers.onnx import export, OnnxConfig
import numpy as np
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
from onnx2kerastl.customonnxlayer import onnx_custom_objects_map
from transformers import AutoModelForSequenceClassification
import torch

# @pytest.mark.skip(reason="Fails on CI but works locally (might be too big?)")
def test_bert_huggingface_classifcation():
    onnx_path = 'bert_huggingface.onnx'
    model_name = "bert-base-uncased"
    model_name_for_features = "bert"
    id2label = {0: "IS_DAMAGED", 1: "NOT_DAMAGED"}
    label2id = {"IS_DAMAGED": 0, "NOT_DAMAGED": 1}
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    real_inputs = tokenizer("Hello, my dog is cute", return_tensors="pt" , padding='max_length', max_length=100)
    OnnxConfig.default_fixed_sequence = 8  # this does nothing here, serves as a reminder
    OnnxConfig.default_fixed_batch = 2  # this does nothing here, serves as a reminder
    albert_features = list(FeaturesManager.get_supported_features_for_model_type(model_name_for_features).keys())
    onnx_path = Path(onnx_path)
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature='sequence-classification')
    onnx_config = model_onnx_config(model.config)
    onnx_inputs, onnx_outputs = export(tokenizer, model, onnx_config, onnx_config.default_onnx_opset, onnx_path)
    onnx_model = onnx.load(onnx_path)
    keras_model = onnx_to_keras(onnx_model, ['input_ids', 'token_type_ids', 'attention_mask'],
                                input_types=[tf.int32, tf.int32, tf.float32])
    keras_model = keras_model.converted_model
    input_np = [real_inputs['input_ids'].numpy(),
                real_inputs['token_type_ids'].numpy(),
                real_inputs['attention_mask'].numpy()]
    with torch.no_grad():
        out = model(**real_inputs)
    flipped_model = convert_channels_first_to_last(keras_model, [])
    flipped_model.save('temp.h5')
    model = tf.keras.models.load_model('temp.h5', custom_objects=onnx_custom_objects_map)
    flipped_otpt = model(input_np)
    assert ((flipped_otpt-out.logits.detach().numpy()).__abs__().numpy().max() < 5*10**-6)

if __name__ == "__main__":
    test_bert_huggingface_classifcation()