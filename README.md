# onnx2keras

ONNX to Keras deep neural network converter.

## API

`onnx_to_keras(onnx_model, input_names, input_shapes=None, name_policy=None, verbose=True, change_ordering=False)`

`onnx_model`: ONNX model to convert

`input_names`: list with graph input names

`input_shapes`: override input shapes (experimental)

`name_policy`: override layer names (experimental)

`verbose`: detailed output

`change_ordering` change ordering to HWC (experimental)

Return: Keras model


## Getting started

### ONNX model
```
import onnx
from onnx2keras import onnx_to_keras

# Load ONNX model
onnx_model = onnx.load('resnet18.onnx')

# Call the converter (input - is the main model input name, can be different for your model)
k_model = onnx_to_keras(onnx_model, ['input'])
```

Keras model will be stored to the `k_model` variable. So simple, isn't it?


### PyTorch model

Using ONNX as intermediate format, you can convert PyTorch model as well.

```
import numpy as np
import torch
import onnx
from torchvision.models.resnet import resnet18
from onnx2keras import onnx_to_keras, check_torch_keras_error


if __name__ == '__main__':
    model = resnet18()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    input_var = torch.FloatTensor(input_np)
    output = model(input_var)

    torch.onnx.export(model, (input_var), "resnet18.onnx",
                      verbose=True,
                      input_names=['input'],
                      output_names=['output']
    )

    onnx_model = onnx.load('resnet18.onnx')
    k_model = onnx_to_keras(onnx_model, ['input'])

    error = check_torch_keras_error(model, k_model, input_np)

    print('Error: {0}'.format(error))  #  1e-6 :)
```


## License
This software is covered by MIT License.
