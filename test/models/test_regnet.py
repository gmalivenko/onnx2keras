import numpy as np
import pytest
from test.utils import convert_and_test
from torchvision.models import regnet_y_128gf, regnet_x_8gf



@pytest.mark.slow
@pytest.mark.parametrize('model_class', [regnet_x_8gf])
@pytest.mark.parametrize('pretrained', [True])
def test_regnet(pretrained, model_class):
    model = model_class(pretrained=pretrained)
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)