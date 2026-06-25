import numpy as np
from onnx import helper, TensorProto
import onnxruntime as rt
import pytest

from onnx2kerastl import onnx_to_keras


def _shape_model(attrs):
    # A symbolic first dim forces the converter down the dynamic tf_shape path,
    # so the Shape output is a real tensor (and a valid Keras model output).
    node = helper.make_node("Shape", inputs=["test_in"], outputs=["test_out"], **attrs)
    graph = helper.make_graph(
        nodes=[node],
        name="test-model",
        inputs=[helper.make_tensor_value_info("test_in", TensorProto.FLOAT, ["B", 3, 4, 5])],
        outputs=[helper.make_tensor_value_info("test_out", TensorProto.INT64, None)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


@pytest.mark.parametrize("attrs", [
    {},                       # no start/end -> full shape (regression: unchanged behavior)
    {"start": 1},             # start only
    {"start": 1, "end": 3},   # start + end
    {"start": -2},            # negative start
    {"end": -1},              # negative end
])
def test_shape_start_end(attrs):
    onnx_model = _shape_model(attrs)
    np_input = np.random.random((1, 3, 4, 5)).astype(np.float32)

    sess = rt.InferenceSession(onnx_model.SerializeToString())
    onnx_out = sess.run([sess.get_outputs()[0].name],
                        {sess.get_inputs()[0].name: np_input})[0]

    keras_model = onnx_to_keras(onnx_model, ["test_in"],
                                name_policy="attach_weights_name").converted_model
    keras_out = keras_model(np_input)
    keras_out = keras_out.numpy() if hasattr(keras_out, "numpy") else np.asarray(keras_out)

    assert np.array_equal(keras_out.reshape(-1).astype(np.int64),
                          onnx_out.reshape(-1).astype(np.int64))
