import numpy as np
import torch
import torch.nn as nn
import pytest
from keras_data_format_converter import convert_channels_first_to_last
from onnx import helper, TensorProto, save
import onnxruntime as rt
from onnx2kerastl import onnx_to_keras
from test.utils import convert_and_test
import tensorflow as tf

class FConstant(nn.Module):
    def __init__(self, constant):
        super(FConstant, self).__init__()
        self.constant = constant

    def forward(self, x):
        return x + nn.functional.one_hot()


class OneHot():
    def __init__(self, depth, values, axis):
        super(OneHot, self).__init__()
        self.depth = depth
        self.values = values
        self.axis = axis

    def get_onnx(self):
        named_input = ['batch', 'h', 'w']
        if self.axis == -1:
            output_shape = named_input + [self.depth]
        else:
            if self.axis < 0:
                add = 1
            else:
                add = 0
            output_shape = named_input[:self.axis+add] + [self.depth] + named_input[self.axis+add:]
        model = helper.make_model(helper.make_graph(
            nodes=[
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["depth"],
                    value=helper.make_tensor('depth_tensor', TensorProto.INT64, [], np.array(self.depth).tobytes(),
                                             raw=True),
                ),
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["values"],
                    value=helper.make_tensor('values_tesnor', TensorProto.INT64, [2],
                                             np.array([self.values[0], self.values[1]]).tobytes(), raw=True),
                ),
                helper.make_node(
                    "OneHot",
                    inputs=["indices", "depth", "values"],
                    outputs=["one_hot_encoded"],
                    axis=self.axis,
                ),
            ],

            name="test-model",
            inputs=[
                helper.make_tensor_value_info("indices", TensorProto.FLOAT, named_input),
            ],
            outputs=[
                helper.make_tensor_value_info("one_hot_encoded", TensorProto.INT64, output_shape)
            ]

        ))
        return model


class TorchOneHot(nn.Module):
    def __init__(self):
        super(TorchOneHot, self).__init__()

    def forward(self, x):
        return torch.nn.functional.one_hot(x, 12)


@pytest.mark.parametrize('constant', [-1.0, 0.0, 1.0])
def test_constant(constant):
    model = FConstant(constant)
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)

@pytest.mark.parametrize('depth', [10])
@pytest.mark.parametrize('values', [[1,3]])
@pytest.mark.parametrize('axis', [-1])
def test_one_hot(depth, values, axis):
    onnx_one_hot = OneHot(depth=depth, values=values, axis=axis).get_onnx()
    indices = np.array([[[1, 9, 3], [2, 4, 5]]], dtype=np.float32)
    keras_model = onnx_to_keras(onnx_one_hot, ['indices'], name_policy='attach_weights_name')
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
    keras_res = final_model(indices)
    sess = rt.InferenceSession(onnx_one_hot.SerializeToString())
    input_name_1 = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred = sess.run([label_name], {input_name_1: indices,})[0]
    assert tf.reduce_max(tf.abs(tf.transpose(keras_res,[0,1,3,2])-pred)).numpy() < 10**(-4)
    assert tf.reduce_mean(tf.abs(tf.transpose(keras_res, [0, 1, 3, 2]) - pred)).numpy() < 10**(-4)

