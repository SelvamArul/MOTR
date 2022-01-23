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
    Instead of separate annotations for track_ids, we use object_ids as the track_ids
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

class YCBV:
    def __init__(self, args, transforms=None):
        self.args = args
        self._transforms = transforms
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.vis = args.vis
        self.video_dict = {}
        
        self.img_files = []
        self.labels = {}
        
        _dataset_path = Path(args.dataset_path)
        if not _dataset_path.exists():
            import sys
            sys.exit("dataset_path ",dataset_path, " does not exist")
        
        for f in _dataset_path.iterdir():
            self.video_dict[str(f)] = len(self.video_dict)
            video_path = f / "rgb"
            self.img_files += sorted([str(x) for x in video_path.iterdir()])
            with open ( f / 'scene_gt.json') as gt_file:
                gt_labels = json.loads(gt_file.read())
                gt_labels_dict = {f"{video_path}/{int(k):06}.png":v for (k,v) in gt_labels.items()}
                self.labels.update(gt_labels_dict)
                

        self.item_num = len(self.img_files) - (self.num_frames_per_batch - 1) * self.sample_interval
        import ipdb; ipdb.set_trace()
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

    # def _register_videos(self):
    #     for label_name in self.label_files:
    #         video_name = '/'.join(label_name.split('/')[:-1])
    #         if video_name not in self.video_dict:
    #             print("register {}-th video: {} ".format(len(self.video_dict) + 1, video_name))
    #             self.video_dict[video_name] = len(self.video_dict)
    #             assert len(self.video_dict) <= 300

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
        return gt_instances

    def _pre_single_frame(self, idx: int):
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]
        img = Image.open(img_path)
        targets = {}
        w, h = img._size
        assert w > 0 and h > 0, "invalid image {} with shape {} {}".format(img_path, w, h)
        if osp.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

            # normalized cewh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = w * (labels0[:, 2] - labels0[:, 4] / 2)
            labels[:, 3] = h * (labels0[:, 3] - labels0[:, 5] / 2)
            labels[:, 4] = w * (labels0[:, 2] + labels0[:, 4] / 2)
            labels[:, 5] = h * (labels0[:, 3] + labels0[:, 5] / 2)
        else:
            raise ValueError('invalid label path: {}'.format(label_path))
        video_name = '/'.join(label_path.split('/')[:-1])
        obj_idx_offset = self.video_dict[video_name] * 100000  # 100000 unique ids is enough for a video.
        targets['boxes'] = []
        targets['area'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['obj_ids'] = []
        targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])
        for label in labels:
            targets['boxes'].append(label[2:6].tolist())
            targets['area'].append(label[4] * label[5])
            targets['iscrowd'].append(0)
            targets['labels'].append(0)
            obj_id = label[1] + obj_idx_offset if label[1] >= 0 else label[1]
            targets['obj_ids'].append(obj_id)  # relative id

        import ipdb; ipdb.set_trace()
        targets['area'] = torch.as_tensor(targets['area'])
        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'])
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'])
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        targets['boxes'][:, 0::2].clamp_(min=0, max=w)
        targets['boxes'][:, 1::2].clamp_(min=0, max=h)
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
        gt_instances = []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)
        data.update({
            'imgs': images,
            'gt_instances': gt_instances,
        })
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


def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'provided MOT path {root} does not exist'
    transforms = make_detmot_transforms(image_set, args)
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
    parser.add_argument("--dataset_path", default="/home/user/amini/datasets/ycbv_coco/train_mini/")
    args = parser.parse_args()

    dataset = YCBV(args)




