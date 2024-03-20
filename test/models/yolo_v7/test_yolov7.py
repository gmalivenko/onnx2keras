import pathlib

import numpy as np
import onnx
import pytest

from onnx2kerastl import onnx_to_keras
from test.utils import NP_SEED


@pytest.mark.slow
@pytest.mark.parametrize('pretrained', [True])
def test_yolov7(pretrained):
    np.random.seed(seed=NP_SEED)

    dir = pathlib.Path(__file__).parent.resolve()
    yolov7_model_path = f"{dir}/yolov7-tiny.onnx"
    onnx_model = onnx.load(yolov7_model_path)

    input_all = [_input.name for _input in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    input_names = list(set(input_all) - set(input_initializer))
    k_model = onnx_to_keras(onnx_model, input_names, name_policy='attach_weights_name', allow_partial_compilation=False)
    # input_np = np.random.uniform(0, 1, (1, 1, 32, 64, 64))
    # error = test_conversion(onnx_model, k_model, input_np, epsilon=2 * 10 ** (-5))
