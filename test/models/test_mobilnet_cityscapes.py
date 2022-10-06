import numpy as np
import pytest
from test.utils import convert_and_test, NP_SEED
import torch
from torch import nn
from torchvision import transforms as T
from test.models.cityscape_semseg.cityscapes import Cityscapes
from test.models.cityscape_semseg import network
import io
import onnx
from onnx2kerastl import onnx_to_keras
from onnx2kerastl.utils import check_torch_keras_error
import urllib.request

DATASET = "KITTI" # KITTI

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def torch2keras(model, img, device='cpu'):
    temp_f = io.BytesIO()
    input_names = ['img']
    torch_input = torch.randn(2, *np.array(img.shape)[1:], device=device)
    np_input = torch_input.cpu().numpy()
    torch.onnx.export(model, torch_input, temp_f,
                      training=torch.onnx.TrainingMode.TRAINING, input_names=input_names,
                      output_names=['segmenation'])

    temp_f.seek(0)
    onnx_model = onnx.load(temp_f)
    k_model = onnx_to_keras(onnx_model, input_names, change_ordering=False)
    error = check_torch_keras_error(model, k_model, np_input, change_ordering=False, epsilon=5*10**(-3),
                                    should_transform_inputs=True)


@pytest.mark.slow
@pytest.mark.parametrize('model', ['deeplabv3plus_mobilenet'])
@pytest.mark.parametrize('num_classes', [19])
@pytest.mark.parametrize('output_strides', [16])
def test_mobile_net_cityscape(model, num_classes, output_strides):
    np.random.seed(seed=NP_SEED)
    LOAD_WEIGHTS = True
    device='cpu'
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/example-datasets-47ml982d/Cityscapes_weights/best_deeplabv3plus_mobilenet_cityscapes_os16.pth",
        "tmp.pth")
    model_weights_loc = "tmp.pth"
    model = network.modeling.__dict__[model](num_classes=num_classes, output_stride=output_strides).cpu()
    model.eval()
    set_bn_momentum(model.backbone, momentum=0.01)
    if LOAD_WEIGHTS:
        checkpoint = torch.load(model_weights_loc, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        del checkpoint
    model.to(device)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    #Input size MUST be divisble by 16 in both axis (this restriction could be removed if we find a way for fractional upsample in keras)
    if DATASET == "KITTI":
        img = np.random.randint(0, 255, (1232, 384, 3), dtype=np.uint8)

    else:
        img = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    np.random.seed(42)
    img = transform(img).unsqueeze(0)  # To tensor of NCHW
    img = img.to(device)
    torch2keras(model, img)
