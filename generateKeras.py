import onnx
import keras
import keras.backend as K
from onnx2keras import onnx_to_keras
import tensorflow as tf

# ONNX_MODEL_NAME = "espnetv2_W512_H336_C3"
ONNX_MODEL_NAME = "deeplabv3+_W512_H336_C3"

# Load ONNX model
# onnx_model = onnx.load('mnist.onnx')
# onnx_model = onnx.load('espnetv2_W512_H336_C3.onnx')
onnx_model = onnx.load("{}.onnx".format(ONNX_MODEL_NAME))

# # Call the converter (input - is the main model input name, can be different for your model)
m = onnx_to_keras(onnx_model, ['actual_input_1'])
m.summary()

# Cannot save due to `symbolic tensor` issue: 711/Identity

tf.saved_model.save( m, "saved_model_{}".format(ONNX_MODEL_NAME) )

m.save("{}.h5".format(ONNX_MODEL_NAME))

# with open("{}.json".format(ONNX_MODEL_NAME), 'w') as f:
#     f.write(m.to_json())

# m.save_weights('mnist.h5')
