"""Loads edadaltocg/resnet34_cifar100 without the `detectors` package.

The architecture is taken directly from detectors/models/resnet.py: a
standard timm resnet34, with conv1 replaced by a 3x3 stride-1 conv (no
downsampling) and maxpool removed (nn.Identity) - the standard CIFAR
adaptation. The `detectors` package itself is not used: importing it
breaks on an incompatibility between an old timm ViT config API and
current timm (1.0.29).

Fetch the weights first:
  curl -sL -A "Mozilla/5.0" \
    "https://huggingface.co/edadaltocg/resnet34_cifar100/resolve/main/pytorch_model.bin" \
    -o resnet34_cifar100.bin
"""

import torch
import torch.nn as nn
import timm

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def build_resnet34_cifar100(pretrained_path: str) -> nn.Module:
    model = timm.create_model("resnet34", pretrained=False, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()

    state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"

    model.eval()
    return model
