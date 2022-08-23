from typing import Dict
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, conv1x1
from torchvision.models.utils import load_state_dict_from_url

from .resnet import FrozenBatchNorm2d
from util.misc import is_main_process


model_cfgs = {
    'refinenetlw50': [Bottleneck, [3, 4, 6, 3], 'https://download.pytorch.org/models/resnet50-19c8e357.pth'],  # 'https://cloudstor.aarnet.edu.au/plus/s/xp7GcVKC0GbxhTv/download'
    'refinenetlw101': [Bottleneck, [3, 4, 23, 3], 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'],  # 'https://cloudstor.aarnet.edu.au/plus/s/CPRKWiaCIDRdOwF/download'
    'refinenetlw152': [Bottleneck, [3, 8, 36, 3], 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'],  # 'https://cloudstor.aarnet.edu.au/plus/s/2w8bFOd45JtPqbD/download'
}


class CRPBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(
                self,
                "{}_{}".format(i + 1, "outvar_dimred"),
                conv1x1(in_planes if (i == 0) else out_planes, out_planes),
            )
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, "{}_{}".format(i + 1, "outvar_dimred"))(top)
            x = top + x

        return x


class ResNetLW(ResNet):

    def __init__(
        self,
        block,
        layers,
        num_classes = 21,
        zero_init_residual = False,
        groups = 1,
        width_per_group = 64,
        replace_stride_with_dilation = None,
        norm_layer = None
    ) -> None:
        super(ResNetLW, self).__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)

        self.dropout = nn.Dropout(p=0.5)

        self.p_ims1d2_outl1_dimred = conv1x1(2048, 512)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256)

        self.p_ims1d2_outl2_dimred = conv1x1(1024, 256)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256)

        self.p_ims1d2_outl3_dimred = conv1x1(512, 256)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256)

        self.p_ims1d2_outl4_dimred = conv1x1(256, 256)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)

        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1, bias=True)

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = self.dropout(l4)
        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = F.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.shape[2:], mode="bilinear", align_corners=True)(x4)

        l3 = self.dropout(l3)
        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.shape[2:], mode="bilinear", align_corners=True)(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.shape[2:], mode="bilinear", align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)

        out = self.clf_conv(x1)
        return out


def _refinenet(arch, pretrained=False, progress=True, **kwargs):
    block, layers, url = model_cfgs[arch]
    model = ResNetLW(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(url, map_location='cpu', progress=progress)
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        model.load_state_dict(state_dict, strict=False)
    return model


class RefineNet(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
        "num_channels": int
    }

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool) -> None:
        backbone = _refinenet(name, pretrained=is_main_process(), replace_stride_with_dilation=[False, False, dilation],
            norm_layer=None)
        for name, parameter in backbone.named_parameters():
            if not train_backbone or name == 'conv1.weight' or name == 'bn1.weight' or name == 'bn1.bias' or 'layer1' in name or 'fc' in name or 'clf_conv' in name:  # or 'outl1' in name or 'g1' in name
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}

        excludes = ("avgpool", "fc", "dropout", "clf_conv")
        layers = OrderedDict()
        for name, module in backbone.named_children():
            if name not in excludes:
                layers[name] = module

        super(RefineNet, self).__init__(layers)
        self.return_layers = return_layers
        self.num_channels = 256
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        layers = {}
        # x4 = self.dropout(l4)
        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = F.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        layers["layer4"] = x4
        x4 = nn.Upsample(size=l3.shape[2:], mode="bilinear", align_corners=True)(x4)

        # x3 = self.dropout(l3)
        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        layers["layer3"] = x3
        x3 = nn.Upsample(size=l2.shape[2:], mode="bilinear", align_corners=True)(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        layers["layer2"] = x2
        x2 = nn.Upsample(size=l1.shape[2:], mode="bilinear", align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        layers["layer1"] = x1

        out = OrderedDict()
        for name in self.return_layers:
            out_name = self.return_layers[name]
            out[out_name] = layers[name]
        return out
