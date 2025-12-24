import onnx
from onnx2kerastl import onnx_to_keras


def main():
    onnx_file_path = "onnx_models/model.onnx"

    # Load ONNX model
    onnx_model = onnx.load(onnx_file_path)

    # Call the converter (input - is the main model input name, can be different for your model)
    k_model = onnx_to_keras(onnx_model, ['input_1'])


if __name__ == '__main__':
    main()
