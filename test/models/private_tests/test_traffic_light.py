import onnxruntime as ort
import numpy as np
import onnx
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import tensorflow as tf
from test.models.private_tests.aws_utils import aws_s3_download
import pytest


@pytest.mark.parametrize('aws_s3_download', [["traffic_light/", "traffic_light/", False]], indirect=True)
def test_traffic_light(aws_s3_download):
    model_path = f'{aws_s3_download}/model.onnx'
    img = np.load(f'{aws_s3_download}/traffic_input.npy')
    onnx_model = onnx.load(model_path)
    keras_model = onnx_to_keras(onnx_model, ['image'], name_policy='attach_weights_name',
                                allow_partial_compilation=False).converted_model
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
    res = final_model(np.transpose(img, [0, 2, 3, 1]))
    ort_session = ort.InferenceSession(model_path)

    res_onnx = ort_session.run(
        ['bbox', 'scores', 'classes', 'cls_probabilities', 'rois', 'objects_idx_2d', 'scores_filtered', 'bulb_state',
         'tcd_face'],
        input_feed={"image": img}
    )
    eps_mean = 1e-6
    eps_max = 1e-5
    # These are really really close

    assert (res[0] - res_onnx[0]).__abs__().numpy().mean() < 2.5 * eps_mean
    assert (res[0] - res_onnx[0]).__abs__().numpy().max() < 7 * eps_max
    assert (res[1] - res_onnx[1]).__abs__().numpy().mean() < eps_mean
    assert (res[1] - res_onnx[1]).__abs__().numpy().max() < eps_mean

    assert (res[2] - res_onnx[2]).__abs__().numpy().mean() < eps_mean
    assert (res[2] - res_onnx[2]).__abs__().numpy().max() < eps_max
    assert (res[3] - res_onnx[3]).__abs__().numpy().mean() < eps_mean
    assert (res[3] - res_onnx[3]).__abs__().numpy().max() < eps_max

    assert (res[4] - res_onnx[4]).__abs__().numpy().mean() < 1.5*eps_mean
    assert (res[4] - res_onnx[4]).__abs__().numpy().max() < 7 * eps_max

    assert (res[5][:, 0] - res_onnx[5]).__abs__().numpy().mean() < eps_mean
    assert (res[5][:, 0] - res_onnx[5]).__abs__().numpy().max() < eps_max

    assert (res[6] - res_onnx[6]).__abs__().numpy().mean() < eps_mean
    assert (res[6] - res_onnx[6]).__abs__().numpy().max() < eps_max

    # These two have lower accuracy but are still acceptable

    (tf.nn.softmax(res[7][:4, :]) - tf.nn.softmax(res_onnx[7][:4, :])).numpy().__abs__().mean() < 5e-3
    (tf.nn.softmax(res[7][:4, :]) - tf.nn.softmax(res_onnx[7][:4, :])).numpy().__abs__().max() < 5e-2

    (tf.nn.softmax(res[8][:4, :]) - tf.nn.softmax(res_onnx[8][:4, :])).numpy().__abs__().mean() < 5e-4
    (tf.nn.softmax(res[8][:4, :]) - tf.nn.softmax(res_onnx[8][:4, :])).numpy().__abs__().max() < 1e-2
