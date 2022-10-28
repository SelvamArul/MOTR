import torch
from pathlib import Path as P
from collections import OrderedDict

in_model_path = P('/home/nfs/inf6/data/models/motr_models/yolopose.pth')

in_model = torch.load(in_model_path)['model']
out_model = OrderedDict()

for k, v in in_model.items():
    _k = k
    if 'translation_embed' in _k:
        _k = 'trans_embed.' + '.'.join( _k.split('.')[1:] )
    if 'rotation_embed' in _k:
        _k = 'rot_embed.'  + '.'.join( _k.split('.')[1:] )
    out_model[_k] = v

save_model = {'model':out_model}
torch.save(save_model, in_model_path.parent / 'yolopose_refactored.pth')

