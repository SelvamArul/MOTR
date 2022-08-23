# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F

from .resnet import ResNet
from .refinenet import RefineNet
from models.position_encoding import build_position_encoding
from util.misc import NestedTensor


class Backbone(nn.Module):

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        super().__init__()
        if 'resnet' in name:
            self.body = ResNet(name, train_backbone, return_interm_layers, dilation)
        elif 'refinenet' in name:
            self.body = RefineNet(name, train_backbone, return_interm_layers, dilation)
        self.num_channels = self.body.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Joiner(nn.Sequential):

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    return model
