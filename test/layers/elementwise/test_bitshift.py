import torch
import numpy as np
from test.utils import convert_and_test, torch2keras
from keras_data_format_converter import convert_channels_first_to_last
from onnx import helper, TensorProto, save
import onnxruntime as rt
from onnx2kerastl import onnx_to_keras


class BitShift():
    def __init__(self, direction):
        super(BitShift, self).__init__()
        self.direction = direction

    def get_onnx(self):
        named_input = ['batch', 'h', 'w']
        model = helper.make_model(helper.make_graph(
            nodes=[
                helper.make_node(
                    "BitShift",
                    inputs=["x", "y"],
                    outputs=["bitshifted"],
                    direction=self.direction
                ),
            ],

            name="test-model",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.UINT64, named_input),
                helper.make_tensor_value_info("y", TensorProto.UINT64, named_input),
            ],
            outputs=[
                helper.make_tensor_value_info("bitshifted", TensorProto.UINT64, named_input)
            ]

        ))
        return model


def test_bitshift():
    onnx_model = BitShift(direction='LEFT').get_onnx()
    x = np.random.randint(low=-100, high=100, size=(1, 50, 50))
    y = np.random.randint(low=1, high=5, size=(1, 50, 50))
    sess = rt.InferenceSession(onnx_model.SerializeToString())
    input_name_1 = sess.get_inputs()[0].name
    input_name_2 = sess.get_inputs()[1].name
    label_name = sess.get_outputs()[0].name
    pred = sess.run([label_name], {input_name_1: x.astype(np.uint64), input_name_2: y.astype(np.uint64)})[0]
    keras_model = onnx_to_keras(onnx_model, ['x', 'y'], name_policy='attach_weights_name')
    final_k = convert_channels_first_to_last(keras_model, ['x', 'y'])
    assert (final_k([x, y])-pred).numpy().max() < 10**(-5)


