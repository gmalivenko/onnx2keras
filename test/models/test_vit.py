import numpy as np
import pytest
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32

from test.utils import convert_and_test


@pytest.mark.slow
@pytest.mark.parametrize('model_class', [vit_b_16, vit_b_32, vit_l_16, vit_l_32])
def test_vit(model_class):
    model = model_class()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
