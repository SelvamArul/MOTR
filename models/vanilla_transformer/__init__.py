# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from . import smca, vanilla


def build_transformer(args):
    if args.transformer == 'smca':
        print ("Transformer type is smca! This should never happen")
        import ipdb; ipdb.set_trace()
        print ()
        return smca.Transformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
            smooth=args.smooth,
            dynamic_scale=args.dynamic_scale
        )
    else:
        return vanilla.Transformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True
        )
