import numpy as np
import torch
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torchvision.models as models

input_np = np.random.uniform(0, 1, (1, 3, 320, 320))
input_var = Variable(torch.FloatTensor(input_np))
model = models.resnet18()
model.eval()
k_model = pytorch_to_keras(model, input_var, [(3, 320, 320,)], verbose=True, change_ordering=True)

for i in range(3):
    input_np = np.random.uniform(0, 1, (1, 3, 320, 320))
    input_var = Variable(torch.FloatTensor(input_np))
    output = model(input_var)
    pytorch_output = output.data.numpy()
    keras_output = k_model.predict(np.transpose(input_np, [0, 2, 3, 1]))
    error = np.max(pytorch_output - keras_output)
    print('error -- ', error)
