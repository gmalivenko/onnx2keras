import io
from typing import Callable

import numpy as np
import onnx
import pytest
import torch
from torch import nn
from torchvision.models import resnet18

from onnx2keras import check_torch_keras_error, onnx_to_keras
from onnx2keras.utils import count_params_keras, count_params_torch


def _convert_and_test_model_training_mode(
    model_class: Callable, include_fc: bool = True
) -> None:

    # instanciate model and optionally discard fc layer
    model = model_class()
    if not include_fc:
        model.fc = nn.Identity()
    model.train()

    print(
        "Created PyTorch model '{}'; param count: {}, trainable param count: {}".format(
            model_class,
            count_params_torch(model),
            count_params_torch(model, trainable_only=True),
        )
    )

    # create placeholder input tensors
    input_np = np.random.randn(1, 3, 224, 224)
    input_t = torch.from_numpy(input_np).float()

    # export PyTorch model to ONNX
    input_names = ["input_1"]
    output_names = ["output_1"]

    temp_f = io.BytesIO()
    torch.onnx.export(
        model,
        input_t,
        temp_f,
        verbose=True,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=14,
        do_constant_folding=False,
        dynamic_axes={
            "input_1": {0: "batch_size"},
            "output_1": {0: "batch_size"},
        },
        training=torch.onnx.TrainingMode.TRAINING,
    )
    temp_f.seek(0)
    onnx_model = onnx.load(temp_f)

    # check the ONNX model
    onnx.checker.check_model(onnx_model)

    # export ONNX model to TF Keras
    change_ordering = True
    k_model = onnx_to_keras(
        onnx_model,
        input_names=input_names,
        input_shapes=[(3, None, None)],
        name_policy=None,
        verbose=True,
        change_ordering=change_ordering,
    )

    # Assert outputs are all close up to an absolute tolerance or 1e-04
    # Note: max absolute difference is expected to be around 5e-05 if `include_fc=False`, that's
    # why we increase `epsilon` from 1e-5 to 1e-4
    model.eval()

    error = check_torch_keras_error(
        model,
        k_model,
        input_np,
        epsilon=1e-4,
        change_ordering=change_ordering,
    )
    print(f"Outputs Max Error: {error}")

    # Verify the number of trainable parameters is the same
    # Note: we don't verify that the number of total parameters is the same
    # because there are possible differences:
    # for particular layers such as BatchNorm, some variables are not considered parameters
    # in PyTorch ('buffers', instead), while they are in Keras.

    n_trainable_params = count_params_torch(model, trainable_only=True)
    k_n_trainable_params = count_params_keras(k_model, trainable_only=True)

    assert n_trainable_params == k_n_trainable_params

    k_model.summary()


@pytest.mark.parametrize("include_fc", [True, False])
def test_resnet18(include_fc: bool):

    _convert_and_test_model_training_mode(resnet18, include_fc=include_fc)
