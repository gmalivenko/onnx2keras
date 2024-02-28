import numpy as np
import onnx
import pytest

from onnx2kerastl import onnx_to_keras
import tensorflow as tf
from keras_data_format_converter import convert_channels_first_to_last
from test.models.private_tests.aws_utils import aws_s3_download


def find_best_matches(A, B):
    n = A.shape[0]
    best_matches = np.ones(n, dtype=int)*(-1)
    for i in range(n):
        best_distance = float('inf')
        best_index = -1
        for j in range(n):
            if j not in best_matches:
                distance = np.sum((A[i] - B[j]) ** 2)
                if distance < best_distance:
                    best_distance = distance
                    best_index = j
        if best_index == -1:
            raise Exception("Could not greedily match the order of the two arrays")
        best_matches[i] = best_index
    return best_matches


def error_test(a, b, epsilon=0.04):
    abs_difference = np.sqrt((a-b)**2)
    assert abs_difference.max() < epsilon
    assert abs_difference.mean() < epsilon


@pytest.mark.parametrize('aws_s3_download', [["maskrcnn/", "maskrcnn/", False]], indirect=True)
def test_maskrcnn_eff(aws_s3_download):
    onnx_model_path = f'{aws_s3_download}/maskrcnn.onnx'
    # save_model_path = f'effizency_models/mod_efficiency.h5'
    real_img = np.load(f'{aws_s3_download}/img.npy')
    real_img = np.transpose(real_img, [1, 2, 0])[None, ...]
    real_img = real_img - np.array([103.53, 116.28, 123.675])
    # load onnx model
    onnx_model = onnx.load(onnx_model_path)
    # extract feature names from the model
    input_features = [inp.name for inp in onnx_model.graph.input]
    # # convert onnx model to keras
    keras_model = onnx_to_keras(onnx_model, input_names=input_features, name_policy='attach_weights_name')
    final_model = convert_channels_first_to_last(keras_model.converted_model, should_transform_inputs_and_outputs=True, verbose=True)
    # final_model.save('temp.h5')
    # final_model = tf.keras.models.load_model('temp.h5')
    keras_output = final_model(real_img)
    fin = []
    for i in range(12):
        fin.append(np.load(f'{aws_s3_download}/orig_outputs/out_{i}.npy'))
    fin = [*fin[:4], *fin[5:]]
    instance_idx_match = find_best_matches(np.transpose(keras_output[0].numpy()[0],[1,0]), fin[0])
    bb_loc_pixel_threshold = 8  #allow a movement of up to 8 pixels, 1% of image
    error_test(np.transpose(keras_output[0].numpy()[0],[1,0]), fin[0][instance_idx_match], epsilon=bb_loc_pixel_threshold)
    mask_prob_th = 0.025
    assert (fin[3][instance_idx_match] - np.transpose(keras_output[3][0], [3,0,1,2])).__abs__().mean()\
           < mask_prob_th
    prob_th = 0.015
    prob_diff = (fin[1][instance_idx_match] - keras_output[1][0]).__abs__().numpy()
    assert prob_diff.max() < prob_th
    assert prob_diff.mean() < prob_th
    assert (fin[4] - keras_output[4]).numpy().max() < 3e-5
    assert (fin[5]-np.transpose(keras_output[5].numpy(), [0, 2, 1])).max() < 3e-6
    assert (keras_output[6][0]-np.transpose(fin[6], [1, 0])).__abs__().numpy().mean() < 0.21
    assert (fin[7]-np.transpose(keras_output[7][0], [1, 0])).__abs__().mean() < 0.26
    assert (fin[8][instance_idx_match]-np.transpose(keras_output[8].numpy()[0], [3, 0, 1, 2])).__abs__().mean() < 0.29
    assert (fin[9]-np.transpose(keras_output[9].numpy()[0],[1,0])).__abs__().max() < 4e-4
    assert (fin[10]-keras_output[10][0]).numpy().__abs__().max() < 1.4e-5
    print(1)
    # dummy_input = np.ones((1, 800, 800, 3))+np.random.random((1, 800, 800, 3))
    # res = final_model(dummy_input) # be sure we are able to
