import numpy as np
import pytest
from torchvision.models import regnet_y_400mf, regnet_y_800mf, regnet_y_8gf, regnet_y_128gf, regnet_x_8gf, \
    regnet_y_1_6gf, regnet_x_400mf, regnet_x_800mf, regnet_x_1_6gf, regnet_x_3_2gf, regnet_y_3_2gf, regnet_x_16gf, \
    regnet_x_32gf, regnet_y_16gf, regnet_y_32gf

from test.utils import convert_and_test


@pytest.mark.slow
@pytest.mark.parametrize('model_class', [regnet_y_400mf, regnet_y_800mf, regnet_y_8gf, regnet_x_8gf,
                                         regnet_y_1_6gf, regnet_x_400mf, regnet_x_800mf, regnet_x_1_6gf, regnet_x_3_2gf,
                                         regnet_y_3_2gf, regnet_x_16gf, regnet_x_32gf, regnet_y_16gf, regnet_y_32gf])
@pytest.mark.parametrize('pretrained', [True, False])
def test_regnet(pretrained, model_class):
    model = model_class(pretrained=pretrained)
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)


@pytest.mark.parametrize('model_class', [regnet_y_128gf])
@pytest.mark.parametrize('pretrained', [False])
def test_regnet_untrained(pretrained, model_class):
    model = model_class(pretrained=pretrained)
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
