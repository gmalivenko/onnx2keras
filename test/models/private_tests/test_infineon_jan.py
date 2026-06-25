import onnxruntime as ort
import numpy as np
import onnx
import tensorflow as tf
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
from test.models.private_tests.aws_utils import aws_s3_download
import pytest


# ONNX TensorProto elem_type -> numpy / tensorflow dtype (only what this model uses)
_ONNX_TO_NP = {1: np.float32, 6: np.int32, 7: np.int64, 9: bool, 11: np.float64}
_ONNX_TO_TF = {1: tf.float32, 6: tf.int32, 7: tf.int64, 9: tf.bool, 11: tf.float64}

# Concrete value used for the dynamic sequence-length dimension during the test.
SEQ_LEN = 64


@pytest.mark.parametrize('aws_s3_download', [["infineon-jan/", "infineon-jan/", False]], indirect=True)
def test_infineon_jan_pinn_model(aws_s3_download):
    """Convert + numerically verify the Infineon dual-decoder PiNN model (model.onnx).

    This is a variable-length sequence model (Conv1d encoders + physics/free
    decoders) with two inputs (`x_norm` float32 [batch, seq_len, 5] and
    `seq_lengths` int64 [batch]) and six outputs, one of which (`mask`) is bool.
    It needs two things the generic image-model template does not handle:

      * It has an int64 input (`seq_lengths`); `input_types` is passed so the
        converter preserves the integer dtype instead of defaulting to float32.
      * `seq_lengths` is rank-1 ([batch]) in ONNX but the converter emits a
        rank-2 Keras input, so every input is reshaped to the converted model's
        expected rank before inference.

    The model on S3 is a self-contained ONNX IR v9 export (onnxruntime<=1.17.3
    only supports IR<=9). Requires the `Shape` op `start`/`end` attributes to be
    honored by the converter (otherwise the seq-length extraction / Range
    subgraph fails).
    """
    model_path = f'{aws_s3_download}/model.onnx'

    onnx_model = onnx.load(model_path)
    assert onnx_model.ir_version <= 9, (
        f"expected IR<=9 but got IR {onnx_model.ir_version}; "
        f"re-export the model with ONNX IR version 9"
    )

    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_infos = session.get_inputs()
    output_info = session.get_outputs()

    onnx_input_elem = {i.name: i.type.tensor_type.elem_type for i in onnx_model.graph.input}

    # Create test inputs for all model inputs, respecting each input's dtype.
    rng = np.random.default_rng(seed=42)
    input_arrays = {}
    input_names = []
    input_types = []

    for input_info in input_infos:
        input_name = input_info.name
        input_names.append(input_name)
        elem_type = onnx_input_elem[input_name]
        np_dtype = _ONNX_TO_NP.get(elem_type, np.float32)
        input_types.append(_ONNX_TO_TF.get(elem_type, tf.float32))

        # Resolve dynamic dims: a 'seq'-named dim -> SEQ_LEN, everything else -> 1.
        test_shape = []
        for dim in input_info.shape:
            if isinstance(dim, str) or dim is None or dim == -1:
                test_shape.append(SEQ_LEN if isinstance(dim, str) and 'seq' in dim.lower() else 1)
            else:
                test_shape.append(dim)

        if np.issubdtype(np_dtype, np.integer):
            # Integer inputs (e.g. seq_lengths) drive masking/Range; use the full length.
            input_arrays[input_name] = np.full(test_shape, SEQ_LEN, dtype=np_dtype)
        else:
            input_arrays[input_name] = rng.random(test_shape).astype(np_dtype)

    # Get ONNX outputs
    output_names = [o.name for o in output_info]
    onnx_outputs = session.run(output_names, input_arrays)

    # Convert to Keras, preserving integer input dtypes.
    keras_model = onnx_to_keras(
        onnx_model,
        input_names,
        name_policy='attach_weights_name',
        allow_partial_compilation=False,
        input_types=input_types,
    ).converted_model

    # Convert data format from channels-first to channels-last
    final_model = convert_channels_first_to_last(
        keras_model,
        should_transform_inputs_and_outputs=False
    )

    # Reshape each input to the rank the converted Keras model expects (the rank-1
    # ONNX `seq_lengths` becomes a rank-2 Keras input). Element count is preserved.
    keras_input_list = []
    for input_name, keras_in in zip(input_names, final_model.inputs):
        arr = input_arrays[input_name]
        while arr.ndim < len(keras_in.shape):
            arr = arr[..., None]
        keras_input_list.append(arr)

    keras_outputs = final_model(keras_input_list)

    # Handle single or multiple outputs
    if not isinstance(keras_outputs, (list, tuple)):
        keras_outputs = [keras_outputs]

    # Compare outputs. z_physics has large magnitude (~1e1), so verify on a
    # relative basis rather than the strict absolute thresholds used for images.
    for i, (keras_out, onnx_out) in enumerate(zip(keras_outputs, onnx_outputs)):
        keras_np = keras_out.numpy() if hasattr(keras_out, 'numpy') else np.asarray(keras_out)
        keras_np = keras_np.astype(np.float64)
        onnx_np = np.asarray(onnx_out).astype(np.float64)

        diff = np.abs(keras_np - onnx_np)
        mean_error = diff.mean()
        max_error = diff.max()
        rel_error = max_error / (np.abs(onnx_np).max() + 1e-9)

        print(f"Output {i} ({output_names[i]}): mean_error={mean_error:.6e}, "
              f"max_error={max_error:.6e}, rel_error={rel_error:.6e}")

        assert rel_error < 1e-3, f"Output {i} relative error {rel_error} exceeds threshold"
