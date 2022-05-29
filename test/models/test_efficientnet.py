import numpy as np
import pytest
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, \
    efficientnet_b5, efficientnet_b6, efficientnet_b7

from test.utils import convert_and_test


@pytest.mark.slow
@pytest.mark.parametrize('model_class', [efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
                                         efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7])
def test_efficientnet(model_class):
    model = model_class()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
