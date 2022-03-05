# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .deformable_detr import build as build_deformable_detr
from .motr import build as build_motr
from .motr_debug import build as build_debug_motr

def build_model(args):
    arch_catalog = {
        'deformable_detr': build_deformable_detr,
        'ddetr' : build_deformable_detr,
        'motr': build_motr,
    }
    assert args.meta_arch in arch_catalog, 'invalid arch: {}'.format(args.meta_arch)

    # if args.motr_debug is not None:
    #     return build_debug_motr(args)
    build_func = arch_catalog[args.meta_arch]
    return build_func(args)

