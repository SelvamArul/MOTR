from typing import Dict
from collections import OrderedDict

import torch
from torch import nn
import torchvision

from util.misc import is_main_process


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class ResNet(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
        "num_channels": int
    }

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool) -> None:
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}

        excludes = ("avgpool", "fc")
        layers = OrderedDict()
        for name, module in backbone.named_children():
            if name not in excludes:
                layers[name] = module

        super(ResNet, self).__init__(layers)
        self.return_layers = return_layers
        self.num_channels = num_channels
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layers = {}
        x1 = self.layer1(x)
        layers["layer1"] = x1
        x2 = self.layer2(x1)
        layers["layer2"] = x2
        x3 = self.layer3(x2)
        layers["layer3"] = x3
        x4 = self.layer4(x3)
        layers["layer4"] = x4

        out = OrderedDict()
        for name in self.return_layers:
            out_name = self.return_layers[name]
            out[out_name] = layers[name]
        return out
