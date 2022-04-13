from pathlib import Path
import trimesh
import torch
import numpy as np


def read_models_points(ds_path=None, num_points=1500): #FIXME
    models = [trimesh.load('/home/cache/datasets/ycbv/models/obj_{:06d}.ply'.format(i)) for i range(1, 22)]
    num_points = min(num_points, min([len(model.vertices) for model in models]))
    points = []
    for model in models:
        points.append(trimesh.sample.sample_surface(model, num_points)[0])
    points = np.stack(points, axis=0)
    return torch.from_numpy(points.astype(np.float32))
