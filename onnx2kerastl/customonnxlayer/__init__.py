from onnx2kerastl.customonnxlayer.onnxhardsigmoid import OnnxHardSigmoid
from onnx2kerastl.customonnxlayer.onnxsqrt import OnnxSqrt
from onnx2kerastl.customonnxlayer.onnxreducemean import OnnxReduceMean
from onnx2kerastl.customonnxlayer.onnxerf import OnnxErf
from onnx2kerastl.customonnxlayer.onnxabs import OnnxAbs
from onnx2kerastl.customonnxlayer.onnxlstm import OnnxLSTM

onnx_custom_objects_map = {
    "OnnxHardSigmoid": OnnxHardSigmoid,
    "OnnxSqrt": OnnxSqrt,
    "OnnxReduceMean": OnnxReduceMean,
    "OnnxAbs": OnnxAbs,
    "OnnxErf": OnnxErf,
    "OnnxLSTM": OnnxLSTM
}
