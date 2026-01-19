import onnxruntime as ort
import numpy as np
import onnx
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
from test.models.private_tests.aws_utils import aws_s3_download
import pytest


@pytest.mark.parametrize('aws_s3_download', [["infineon/", "infineon/", False]], indirect=True)
def test_infineon_ts_model(aws_s3_download):
    """Test conversion of Infineon enhanced_trial_19_full_model_complete.onnx model"""
    model_path = f'{aws_s3_download}/enhanced_trial_19_full_model_complete.onnx'

    # Load ONNX model
    onnx_model = onnx.load(model_path)

    # Create ONNX Runtime session to get input/output info
    session = ort.InferenceSession(model_path)
    input_infos = session.get_inputs()
    output_info = session.get_outputs()

    # Create random test inputs for all model inputs
    rng = np.random.default_rng(seed=42)
    input_arrays = {}
    input_names = []

    for input_info in input_infos:
        input_name = input_info.name
        input_names.append(input_name)
        input_shape = input_info.shape

        # If any dimension is dynamic (None or -1), use a reasonable value
        test_shape = []
        for dim in input_shape:
            if isinstance(dim, str) or dim is None or dim == -1:
                test_shape.append(1)  # Use batch size of 1 for dynamic dimensions
            else:
                test_shape.append(dim)

        input_arrays[input_name] = rng.random(test_shape).astype(np.float32)

    # Get ONNX outputs
    output_names = [o.name for o in output_info]
    onnx_outputs = session.run(output_names, input_arrays)

    # Convert to Keras
    keras_model = onnx_to_keras(
        onnx_model,
        input_names,
        name_policy='attach_weights_name',
        allow_partial_compilation=False
    ).converted_model

    # Convert data format from channels-first to channels-last
    final_model = convert_channels_first_to_last(
        keras_model,
        should_transform_inputs_and_outputs=False
    )

    # Get Keras outputs - pass inputs in the same order as input_names
    keras_input_list = [input_arrays[name] for name in input_names]
    keras_outputs = final_model(keras_input_list)

    # Handle single or multiple outputs
    if not isinstance(keras_outputs, (list, tuple)):
        keras_outputs = [keras_outputs]

    # Compare outputs with tolerance
    for i, (keras_out, onnx_out) in enumerate(zip(keras_outputs, onnx_outputs)):
        keras_np = keras_out.numpy() if hasattr(keras_out, 'numpy') else keras_out
        diff = np.abs(keras_np - onnx_out)
        mean_error = diff.mean()
        max_error = diff.max()

        print(f"Output {i}: mean_error={mean_error:.6e}, max_error={max_error:.6e}")

        # Assert with reasonable tolerance
        assert mean_error < 1e-4, f"Output {i} mean error {mean_error} exceeds threshold"
        assert max_error < 1e-3, f"Output {i} max error {max_error} exceeds threshold"
