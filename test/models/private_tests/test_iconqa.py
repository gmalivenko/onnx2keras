import onnxruntime as ort
import numpy as np
import onnx
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import tensorflow as tf
from test.models.private_tests.aws_utils import aws_s3_download
import pytest


def get_keras_layer_model(keras_model, layer_name):
    idx = np.argmax([keras_model.layers[i].name == layer_name for i in range(len(keras_model.layers))])
    return tf.keras.Model(keras_model.input, keras_model.layers[idx].output)


# res_torch = np.load('logits.npy')
@pytest.mark.parametrize('aws_s3_download', [["iconqa/", "iconqa/", False]], indirect=True)
def test_iconqa(aws_s3_download):
    img = np.load(f'{aws_s3_download}/torch_img.npy')
    c = np.load(f'{aws_s3_download}/c.npy')
    q = np.load(f'{aws_s3_download}/q.npy')
    onnx_model = onnx.load(f'{aws_s3_download}/complete_model.onnx')
    keras_model = onnx_to_keras(onnx_model, ['img', 'question', 'choices'], name_policy='attach_weights_name',
                                allow_partial_compilation=False).converted_model
    final_k = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
    final_k.save('temp.h5')
    random_d = np.random.random(size=img.shape)*50
    loaded_model = tf.keras.models.load_model('temp.h5')
    res_perm = loaded_model([q, np.transpose(img, [0, 2, 3, 1]), c.swapaxes(1,2)])
    res_perm_2 = loaded_model([q, np.transpose(img+random_d, [0, 2, 3, 1]), c.swapaxes(1,2)])

    sess = ort.InferenceSession(f'{aws_s3_download}/complete_model.onnx')
    res_onnx = sess.run(
        ['logits'],
        input_feed={'choices': c, 'question':q, 'img':img.astype(np.float32)}
    )
    res_2_onnx = sess.run(
        ['logits'],
        input_feed={'choices': c, 'question':q, 'img':img.astype(np.float32)+random_d.astype(np.float32)}
    )
    diff_res = res_perm-res_onnx[0]
    assert diff_res.numpy().mean() < 0.1
    assert diff_res.numpy().max() < 0.2

    diff_res_2 = res_perm_2-res_2_onnx[0]
    assert diff_res_2.numpy().mean() < 0.1
    assert diff_res_2.numpy().max() < 0.2
