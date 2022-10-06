import numpy as np
import pytest
from test.utils import convert_and_test, NP_SEED
from torchvision.models.segmentation import deeplabv3_resnet50


@pytest.mark.slow
@pytest.mark.parametrize('model_class', [deeplabv3_resnet50])
@pytest.mark.parametrize('pretrained', [True])
def test_deeplab(pretrained, model_class):
    np.random.seed(seed=NP_SEED)
    model = model_class(pretrained=pretrained)
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 256, 256))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True, epsilon=2*10**(-5))
