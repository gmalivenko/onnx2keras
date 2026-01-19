import argparse
import numpy as np
import onnx
from onnx2kerastl import onnx_to_keras
import tensorflow as tf
from keras_data_format_converter import convert_channels_first_to_last

def convert_onnx_to_keras(onnx_model_path, transform_io:bool = True):
    # Load ONNX model
    save_model_path = onnx_model_path.replace('.onnx', '.h5')
    onnx_model = onnx.load(onnx_model_path)
    
    # Extract input feature names from the model
    input_features = [inp.name for inp in onnx_model.graph.input]
    
    # Convert ONNX model to Keras
    keras_model = onnx_to_keras(onnx_model, input_names=input_features,
                                name_policy='attach_weights_name', allow_partial_compilation=False).converted_model

    # Convert from channels-first to channels-last format
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=transform_io,
                                                 verbose=True)

    # Save the final Keras model
    final_model.save(save_model_path)
    print(f"Model saved to {save_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert ONNX model to Keras')
    parser.add_argument('onnx_path', type=str, help='Path to the input ONNX model')    
    parser.add_argument('transform_input_output', type=bool, help='Whether to transform input and output data format')
    args = parser.parse_args()
    
    # Convert input_shape string to tuple of integers
    
    convert_onnx_to_keras(args.onnx_path, args.transform_input_output)