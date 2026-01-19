import numpy as np
import pytest
from test.utils import convert_and_test, NP_SEED
from torchvision.models import resnext50_32x4d



@pytest.mark.slow
@pytest.mark.parametrize('model_class', [resnext50_32x4d])
@pytest.mark.parametrize('pretrained', [True])
def test_resnext(pretrained, model_class):
    np.random.seed(seed=NP_SEED)
    model = model_class(pretrained=pretrained)
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
