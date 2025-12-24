import os.path

import onnx
import pytest
import tensorflow as tf
from transformers import AutoTokenizer
import numpy as np
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
from onnx2kerastl.customonnxlayer import onnx_custom_layers
#from test.utils import export_torch_to_onnx_optimum


@pytest.mark.skip(reason="Fails on CI but works locally (might be too big?)")
def test_llama_32_1b_inst():
    onnx_model_folder = 'onnx_model'
    onnx_path = os.path.join(onnx_model_folder, 'model.onnx')
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # --------------------------------- Export to ONNX -------------------------------------
    export_torch_to_onnx_optimum(model_name, model_output_path=onnx_model_folder)
    # ----------------------------------------- Input Preparation --------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    text = "i love this movie!"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user",
          "content": f"What is the sentiment of this sentence: \"{text}\"? Respond with 'positive' or 'negative' only."}],
        add_generation_prompt=True,
        return_tensors="np"
    )
    input_ids = prompt
    attention_mask = (input_ids != tokenizer.pad_token_id).astype(np.int64)
    position_ids = np.arange(input_ids.shape[1])[None, :]
    model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }
    keras_inputs = {k: tf.convert_to_tensor(v) for k, v in model_inputs.items()}
    # --------------------------------- Export to Keras -------------------------------------
    onnx_model = onnx.load(onnx_path)  # TODO: add to requirements, updated onnx==1.17.0 ()
    keras_model = onnx_to_keras(onnx_model, ['input_ids', 'attention_mask', 'position_ids'],
                                allow_partial_compilation=False)
    keras_model = keras_model.converted_model
    flipped_model = convert_channels_first_to_last(keras_model, [])
    flipped_model.save('temp.h5')
    model = tf.keras.models.load_model('temp.h5', custom_objects=onnx_custom_layers)
    # --------------------------------- Evaluating Inference -------------------------------------
    outputs = model(keras_inputs)
    last_token_logits = outputs[0, -1]
    pred_token_id = np.argmax(last_token_logits)
    pred_token = tokenizer.decode([pred_token_id]).strip().lower()

    assert pred_token=='positive'

if __name__ == "__main__":
    test_llama_32_1b_inst()