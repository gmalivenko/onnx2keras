import numpy as np
import pytest
from test.utils import convert_and_test
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32



@pytest.mark.slow
@pytest.mark.parametrize('model_class', [vit_b_16, vit_b_32, vit_l_16, vit_l_32])
@pytest.mark.parametrize('pretrained', [True, False])
def test_vit(pretrained, model_class):
    model = model_class(pretrained=pretrained)
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
