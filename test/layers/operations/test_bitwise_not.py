from onnx import helper, TensorProto
import numpy as np
from keras_data_format_converter import convert_channels_first_to_last
from onnx2kerastl import onnx_to_keras


class BitwiseNot():
    def __init__(self):
        super(BitwiseNot, self).__init__()

    def get_onnx(self):
        model = helper.make_model(helper.make_graph(
            nodes=[
                helper.make_node(
                    "BitwiseNot",
                    inputs=["x"],
                    outputs=["biwise_not"],
                )],

            name="test-model",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.INT32, ["B", "N", "C"]),
            ],
            outputs=[
                helper.make_tensor_value_info("biwise_not", TensorProto.INT32, ["B", "N", "C"])
            ]

        ))
        return model


def test_bitwise_not():
    np_input = np.random.randint(low=1, high=2000, size=(1, 8, 3), dtype=np.int32) # onnx export only supports bool input
    onnx_bitwise = BitwiseNot().get_onnx()
    keras_model = onnx_to_keras(onnx_bitwise, ['x'], name_policy='attach_weights_name')
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
    keras_res = final_model(np_input)
    assert (np.abs((np.bitwise_not(np_input) - keras_res).numpy()) < 1).all()