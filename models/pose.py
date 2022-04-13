import math

import torch
from torch import nn
import torch.nn.functional as F

from util.pose_ops import vec2mat, quat2mat, euler2mat, transform_poses, rot6d2mat


def separate_axis_from_angle(vectors):
    angles = torch.sqrt(torch.square(vectors).sum(1))
    mask = angles != 0
    angles = angles.unsqueeze(1)
    axes = torch.zeros_like(vectors)
    axes[mask] = torch.divide(vectors[mask], angles[mask])
    return axes, angles


def cross(input, other, dim=-1):
    x1, y1, z1 = input.unbind(dim)
    x2, y2, z2 = other.unbind(dim)
    x = y1 * z2 - z1 * y2
    y = z1 * x2 - x1 * z2
    z = x1 * y2 - y1 * x2
    return torch.stack([x, y, z], dim)


def dot(input, other, dim=-1):
    return torch.sum(input * other, dim).unsqueeze(dim)


def axis_angle_rotate(points, rotations):
    axes, angles = separate_axis_from_angle(rotations)
    axes = axes.unsqueeze(1)
    cos_angles = torch.cos(angles).unsqueeze(1)
    sin_angles = torch.sin(angles).unsqueeze(1)
    return points * cos_angles + cross(axes, points) * sin_angles + axes * dot(axes, points) * (1.0 - cos_angles)


def rotation_6d_rotate(points, rotations):
    return torch.squeeze(rotations.unsqueeze(1) @ points.unsqueeze(-1), dim=-1)


class PostProcessPose(nn.Module):
    def __init__(self, rotation_representation, translation_scale):
        super().__init__()
        self.rotation_representation = rotation_representation
        self.translation_scale = translation_scale

    @torch.no_grad()
    def forward(self, results, outputs, target_intrinsics, target_transforms, target_scales):
        out_rotations, out_translations = outputs['pred_rotations'], outputs['pred_translations']

        assert len(out_rotations) == len(out_translations) == len(target_intrinsics) == len(target_transforms) == len(target_scales)

        rotations = out_rotations.flatten(0, 1)
        translations = out_translations.flatten(0, 1) * 1000 / self.translation_scale
        if self.rotation_representation == "axis_angle":
            rotations = vec2mat(rotations * math.pi).cuda()
        elif self.rotation_representation == "quaternion":
            rotations = quat2mat(rotations)
        elif self.rotation_representation == "rotation_6d":
            rotations = rot6d2mat(rotations)
        else:
            rotations = rotations.view(-1, 3, 3)

        poses = torch.cat([rotations, translations[..., None]], dim=2).view(target_intrinsics.size(0), -1, 3, 4)
        for result, cur_poses, intrinsic, transform, scale in zip(results, poses, target_intrinsics, target_transforms, target_scales):
            # rotations = vec2mat(result["gt_rotations"]).cuda()
            # cur_gt_poses = torch.cat([rotations, result["gt_translations"][..., None]], dim=2)
            # orig_gt_poses = transform_poses(cur_gt_poses, intrinsic, transform, scale, inv=True)
            # result['gt_rotations'] = orig_gt_poses[..., :3]
            # result['gt_translations'] = orig_gt_poses[..., 3]
            orig_poses = transform_poses(cur_poses, intrinsic, transform, scale, inv=True)
            result['rotations'] = orig_poses[..., :3]
            result['translations'] = orig_poses[..., 3]

        return results
