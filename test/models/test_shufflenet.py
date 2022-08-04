import numpy as np
import pytest
from test.utils import convert_and_test
from torchvision.models import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0



@pytest.mark.slow
@pytest.mark.parametrize('model_class', [shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0])
@pytest.mark.parametrize('pretrained', [True, False])
def test_shufflenet(pretrained, model_class):
    model = model_class(pretrained=pretrained)
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
