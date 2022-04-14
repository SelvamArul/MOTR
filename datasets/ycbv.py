# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Dataset interface for ycbv videos (in coco format)
Adapted from detmot.py 

Important details:
    ycbv frames have only one instance per class
    Instead of separate annotations for track_ids, we use object_ids as the track_ids with video specific prefix
    class id labels follow ycbv covention (22 classes in total for 21 objects with 0 for background)
"""
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
from PIL import Image, ImageDraw
import copy
# import datasets.transforms as T
import json

if __name__ == '__main__':
    from pathlib import Path
    import sys
    print ("Adding ",Path(__file__).resolve().parent.parent, " to path")
    sys.path.append(str(Path(__file__).resolve().parent.parent))


from models.structures import Instances

from pathlib import Path
import argparse
from scipy.io import loadmat

class YCBV:
    def __init__(self, args, transforms=None):
        self.args = args
        self._transforms = transforms
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.vis = args.vis
        self.video_names = []
        self.video_dict = {}
        self.dataset_desc_file = args.dataset_desc_file_train if args.image_set == "train" else args.dataset_desc_file_val
        self.img_files = []
        
        _dataset_path = Path(args.dataset_path)
        if not _dataset_path.exists():
            import sys
            sys.exit("dataset_path "+ str(_dataset_path)+ " does not exist")
        
        with open(self.dataset_desc_file, "r") as desc_file:
            lines = desc_file.readlines()
            self.video_names += [line.strip() for line in lines]
        
        for video_name in self.video_names:
            video_path = _dataset_path / video_name
            self.video_dict[str(video_path)] = len(self.video_dict)
            self.img_files += sorted([str(x) for x in video_path.iterdir() if "-color.png" in str(x)])

        # import ipdb; ipdb.set_trace()


        self.item_num = len(self.img_files) - (self.num_frames_per_batch - 1) * self.sample_interval
        # self._register_videos()

        # video sampler.
        self.sampler_steps: list = args.sampler_steps
        self.lengths: list = args.sampler_lengths
        print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.lengths))
        if self.sampler_steps is not None and len(self.sampler_steps) > 0:
            # Enable sampling length adjustment.
            assert len(self.lengths) > 0
            assert len(self.lengths) == len(self.sampler_steps) + 1
            for i in range(len(self.sampler_steps) - 1):
                assert self.sampler_steps[i] < self.sampler_steps[i + 1]
            self.item_num = len(self.img_files) - (self.lengths[-1] - 1) * self.sample_interval
            self.period_idx = 0
            self.num_frames_per_batch = self.lengths[0]
            self.current_epoch = 0

        self.obj_id_to_name  = {
                1:"002_master_chef_can",
                2:"003_cracker_box",
                3:"004_sugar_box",
                4:"005_tomato_soup_can",
                5:"006_mustard_bottle",
                6:"007_tuna_fish_can",
                7:"008_pudding_box",
                8:"009_gelatin_box",
                9:"010_potted_meat_can",
                10:"011_banana",
                11:"019_pitcher_base",
                12:"021_bleach_cleanser",
                13:"024_bowl",
                14:"025_mug",
                15:"035_power_drill",
                16:"036_wood_block",
                17:"037_scissors",
                18:"040_large_marker",
                19:"051_large_clamp",
                20:"052_extra_large_clamp",
                21:"061_foam_brick"
                }

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = targets['boxes']
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        gt_instances.area = targets['area']
        gt_instances.poses = targets['poses']
        gt_instances.translations = targets['translations']
        gt_instances.rotations = targets['rotations']
        return gt_instances

    @staticmethod
    def _read_boxes(file_path):
        boxes = {}
        with open(file_path, 'r') as boxes_file:
            for line in boxes_file:
                line = line.strip().split()
                boxes[line[0]] = [float(x) for x in line[1:]]

        return boxes

    def _pre_single_frame(self, idx: int):
        # TODO
        img_path = self.img_files[idx]
        img = Image.open(img_path)
        targets = {}
        w, h = img._size
        assert w > 0 and h > 0, "invalid image {} with shape {} {}".format(img_path, w, h)
        
        # import ipdb; ipdb.set_trace()
        label_path = img_path[:-10] + "-meta.mat"
        boxes_path = img_path[:-10] + "-box.txt"

        labels = loadmat(label_path) 
        
        # filter out all objects that are outside of field of view
        valid_cls_ids = torch.ones(labels['cls_indexes'].squeeze(axis=1).shape[0], dtype=torch.bool)
        for _idx in range(valid_cls_ids.shape[0]):
            center = labels['center'][_idx]
            if center[0] >= w or center[1] >= h:
                valid_cls_ids[_idx] = False
        labels['cls_indexes'] = labels['cls_indexes'].squeeze(axis=1)[valid_cls_ids.numpy()]

        _boxes = self._read_boxes(boxes_path) 
        video_name = str(Path(img_path).parent)
        obj_idx_offset = (self.video_dict[video_name] + 1) * 100000  # 100000 unique ids is enough for a video.
        targets['boxes'] = []
        targets['area'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['obj_ids'] = []
        targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])
        targets['poses'] = torch.tensor(labels['poses']).permute(2,0,1) # NX3X4 objects
       
        # filter out all objects that are outside of field of view
        targets['poses'] = targets['poses'][valid_cls_ids]
        targets['rotations'] = targets['poses'][:, :, :3] # Nx3X3
        targets['translations'] = targets['poses'][:, :, 3] # Nx3 #FIXME loss of a dim okay?

        cls_ids = labels['cls_indexes'].tolist()
        boxes = []
        try:
            for _idx in cls_ids:
                _bb = _boxes[self.obj_id_to_name[_idx]] # NOTE: _bb is in x1y1x2y2 format
                # _xywh = [_bb[0], _bb[1], _bb[2] - _bb[0], _bb[3] - _bb[1]]
                boxes.append(_bb)
        except Exception as e:
            print (e)
            import ipdb; ipdb.set_trace()
            print ()
        targets['boxes'] = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        targets['boxes'][:, 0::2].clamp_(min=0, max=w)
        targets['boxes'][:, 1::2].clamp_(min=0, max=h)
        # import ipdb; ipdb.set_trace()
        targets['obj_ids'] = torch.as_tensor(cls_ids, dtype=torch.int64) + obj_idx_offset
        targets['labels'] = torch.as_tensor(cls_ids, dtype=torch.int64)
        targets['area'] = (targets['boxes'][:, 2] * targets['boxes'][:,3])
        targets['iscrowd'] = torch.zeros_like(targets['labels'])
        # import ipdb; ipdb.set_trace()
        # print()

        return img, targets

    def _get_sample_range(self, start_idx):

        # take default sampling method for normal dataset.
        assert self.sample_mode in ['fixed_interval', 'random_interval'], 'invalid sample mode: {}'.format(self.sample_mode)
        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = start_idx, start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1, sample_interval
        return default_range

    def pre_continuous_frames(self, start, end, interval=1):
        targets = []
        images = []
        for i in range(start, end, interval):
            img_i, targets_i = self._pre_single_frame(i)
            images.append(img_i)
            targets.append(targets_i)
        return images, targets

    def __getitem__(self, idx):
        sample_start, sample_end, sample_interval = self._get_sample_range(idx)
        images, targets = self.pre_continuous_frames(sample_start, sample_end, sample_interval)
        data = {}
        if self._transforms is not None:
            images, targets = self._transforms(images, targets)
        # import ipdb; ipdb.set_trace()
        try:
            gt_instances = []
            for img_i, targets_i in zip(images, targets):
                gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
                gt_instances.append(gt_instances_i)
            data.update({
                'imgs': images,
                'gt_instances': gt_instances,
            })
        except Exception as e:
            print (e)
            import ipdb; ipdb.set_trace()
        if self.args.vis:
            data['ori_img'] = [target_i['ori_img'] for target_i in targets]
        return data

    def __len__(self):
        return self.item_num


# class DetMOTDetectionValidation(DetMOTDetection):
#     def __init__(self, args, seqs_folder, transforms):
#         args.data_txt_path = args.val_data_txt_path
#         super().__init__(args, seqs_folder, transforms)


def make_detmot_transforms(image_set, args=None):
    import datasets.transforms as T
    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        color_transforms = []
        scale_transforms = [
            T.MotRandomHorizontalFlip(),
            T.MotRandomResize(scales, max_size=1333),
            normalize,
        ]

        return T.MotCompose(color_transforms + scale_transforms)

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def get_ycbv_transforms():
    
    import datasets.transforms as T
    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms = T.MotCompose([normalize])
    return transforms


def build(image_set, args):
    # root = Path(args.mot_path)
    # assert root.exists(), f'provided YCBV path {root} does not exist'
    transforms = get_ycbv_transforms()
    args.image_set = image_set
    if image_set == 'train':
        dataset = YCBV(args, transforms=transforms)
    if image_set == 'val':
        dataset = YCBV(args, transforms=transforms)
    return dataset





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='YCBV dataset test')
    parser.add_argument("--sampler_lengths", type=int, default=[2, 3, 4, 5], nargs="*")
    parser.add_argument("--sample_interval", type=int, default=10)
    parser.add_argument("--sampler_steps", type=int, default=[50, 90, 150], nargs="*")
    parser.add_argument("--sample_mode", default="random_interval")
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--dataset_path", default="/home/data/datasets/YCB_Video_Dataset/data/")
    parser.add_argument("--dataset_desc_file_train", default="/home/user/periyasa/workspace/MOTR/datasets/ycbv_train_desc_mini.txt")
    parser.add_argument("--image_set", default="train")

    args = parser.parse_args()
    transforms = get_ycbv_transforms()
    dataset = YCBV(args, transforms)
    d = dataset[10]
    import ipdb; ipdb.set_trace()



