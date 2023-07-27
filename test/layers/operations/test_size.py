import numpy as np
from keras_data_format_converter import convert_channels_first_to_last
from onnx import helper, TensorProto
import onnxruntime as rt
from onnx2kerastl import onnx_to_keras


class OnnxSize():
    def __init__(self):
        super(OnnxSize, self).__init__()

    def get_onnx(self):
        model = helper.make_model(helper.make_graph(
            nodes=[
                helper.make_node(
                    "Size",
                    inputs=["test_in"],
                    outputs=["test_out"],
                )],

            name="test-model",
            inputs=[
                helper.make_tensor_value_info("test_in", TensorProto.FLOAT, ["B", "N", "C"]),
            ],
            outputs=[
                helper.make_tensor_value_info("test_out", TensorProto.INT64, [])
            ]

        ))
        return model


def test_size():
    np_input = np.random.random((1, 8, 3)) # onnx export only supports bool input
    onnx_size = OnnxSize().get_onnx()
    keras_model = onnx_to_keras(onnx_size, ['test_in'], name_policy='attach_weights_name')
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
    keras_res = final_model(np_input)
    sess = rt.InferenceSession(onnx_size.SerializeToString())
    input_name_1 = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred = sess.run([label_name], {input_name_1: np_input.astype(np.float32), })[0]
    assert (keras_res-pred) < 1

