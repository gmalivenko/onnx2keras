from onnx2kerastl.customonnxlayer.onnxhardsigmoid import OnnxHardSigmoid
from onnx2kerastl.customonnxlayer.onnxsqrt import OnnxSqrt
from onnx2kerastl.customonnxlayer.onnxreducemean import OnnxReduceMean
from onnx2kerastl.customonnxlayer.onnxerf import OnnxErf

onnx_custom_objects_map = {
    "OnnxHardSigmoid": OnnxHardSigmoid,
    "OnnxSqrt": OnnxSqrt,
    "OnnxReduceMean": OnnxReduceMean,
    "OnnxErf": OnnxErf
}
