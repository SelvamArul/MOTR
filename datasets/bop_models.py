import json
from pathlib import Path

import torch
import torch.utils.data
import numpy as np
import trimesh

class BOPModel(torch.utils.data.Dataset):
    def __init__(self, root, kpts_file, num_keypoints, trans_scale=1):
        self.num_keypoints = num_keypoints
        self.trans_scale = trans_scale

        with open(root / 'models_info.json') as f:
            info = json.load(f)
        keypoints_pickle = np.load(root / kpts_file, allow_pickle=True)  # 'box_points_new_mm.pkl', 'fps_points_mm.pkl'
        models, points, diameters, sym_classes, keypoints = [], [], [], [], []
        for obj_id, data in sorted(info.items(), key=lambda x: int(x[0])):
            models.append(trimesh.load(root / 'obj_{:06d}.ply'.format(int(obj_id))))
            points.append(np.array(models[-1].vertices, dtype=np.float64))
            diameters.append(data['diameter'])
            if 'symmetries_discrete' in data:
                sym_classes.append(int(obj_id))
            keypoints.append(np.concatenate([keypoints_pickle[obj_id][:num_keypoints], np.zeros((1, 3))], axis=0))  # keypoints_pickle[obj_id]['fps8_and_center'][:-1]
        self.models = models
        self.points = np.array(points, dtype=object)
        self.diameters = np.array(diameters, dtype=np.float64) * 0.001  # mm to meter
        self.keypoints = np.array(keypoints, dtype=np.float64)
        # back to posecnn points
        if False: #FIXME
            offsets = []
            with open(root / 'offsets.txt') as f:
                for offset in f.read().splitlines():
                    offsets.append([float(x) for x in offset[4:-1].split(', ')])
            self.offsets = np.array(offsets, dtype=np.float64)
            self.posecnn_points = np.array([pts - offset for pts, offset in zip(self.points, self.offsets)], dtype=object)
        # self.points = self.posecnn_points
        # self.keypoints -= self.offsets[:, None]
        # self.diameters /= trans_scale
        # self.keypoints /= trans_scale

        self.sym_classes = [13, 16, 19, 20, 21]
        self.bop_sym_classes = sym_classes

    def sample_points(self, num_points, seed=None):
        points = []
        for model in self.models:
            if isinstance(model, trimesh.points.PointCloud):
                indices = np.random.RandomState(seed=seed).choice(model.shape[0], num_points, replace=False)
                points.append(model[indices])
                # step = max(model.shape[0] // num_points, 1)
                # points.append(model[::step][:num_points])
            else:
                points.append(trimesh.sample.sample_surface(model, num_points)[0])  # trimesh.sample.sample_surface_even(model, num_points)[0]
        points = np.array(points, dtype=np.float32)
        return torch.from_numpy(points)

    def __getitem__(self, index):
        return self.models[index]

    def __len__(self):
        return len(self.models)



def build(args):
    root = Path(args.dataset_path).parent # FIXME 
    assert root.exists(), f'provided dataset path {root} does not exist'
    if args.dataset_file in ('synpick', 'ycbv'):
        path = root / "models_bop"
        if args.posecnn_val:
            path = root / "models_bop_compat_eval"
        kpts_file = "box_points_new_mm.pkl"
        dataset = BOPModel(path, kpts_file, num_keypoints=args.num_keypoints, trans_scale=args.trans_scale)

    return dataset



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='BOP models in dataset interface')
    parser.add_argument('--dataset_file', default='ycbv')
    parser.add_argument('--trans_scale', type=int, default=1, help='pose dataset unit')
    parser.add_argument('--num_keypoints', default=32, type=int)
    parser.add_argument('--dataset_path', default='/home/cache/datasets/YCB_VIDEO_DATASET/YCB_Video_Dataset/data')
    parser.add_argument('--posecnn_val', action='store_true')
    args = parser.parse_args()
    dataset = build(args)
    
    import ipdb; ipdb.set_trace()
    print()
