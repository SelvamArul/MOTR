import os
import csv
import copy
import pickle
from collections import OrderedDict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.io import savemat
from tabulate import tabulate
from matplotlib import pyplot as plt

# import ..pose_util.misc as utils
from pose_util import misc as utils

print ('utils', utils)
from pose_util.match import bbox_overlaps, rot_errors
from pose_util.pysixd.pose_error import adi, add, te, re, reproj
from pose_util.pysixd.score import voc_ap
from pose_util.visualization import draw_poses
from pose_util.pose_ops import pnp, Transform, rot6d2mat


def mat_from_qt(qt):
    wxyz = qt[:4].copy().tolist()
    xyzw = [*wxyz[1:], wxyz[0]]
    t = qt[4:].copy()
    return Transform(xyzw, t)


def norm(x):
    min_x, max_x = x.min(), x.max()
    return np.zeros_like(x) if min_x == max_x else (x - min_x) / (x.max() - min_x)


class PoseEvaluator(object):

    def __init__(self, model_ds, obj_id_dict, output_dir="pose_eval", threshold=0.5):
        self.model_ds = model_ds
        self.output_dir = output_dir
        self.threshold = threshold
        _d = {0: 'all'}
        _d_full = {**_d, **obj_id_dict}
        self.classes = OrderedDict(sorted(_d_full.items()))
        self.errors = {}
        self.bins = list(range(0, 200, 20))
        self.stats = [.0] * 2
        self.predictions = []  # bop

    def matching(self, dts, gts, threshold):
        labels = gts['labels']
        scores = dts['scores'] #.squeeze().max(axis=-1)
        # dts['labels'] =
        try:
            indices = np.nonzero(scores >= threshold)[0]
            order = np.argsort(-scores[indices], kind='mergesort')
            keep = indices[order]
        except Exception as e:
            print (e)
            import ipdb; ipdb.set_trace()
            print ()
        if keep.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        dts_labels = dts['labels'][keep]
        dts_boxes = dts['boxes'][keep] if 'boxes' in dts else None
        dts_rot = dts['rotations'][keep]
        dts_trans = dts['translations'][keep]
        gts_ind, dts_ind = [], []
        try:
            for l in np.unique(dts_labels):
                mask_g, mask_d = labels == l, dts_labels == l
                if mask_g.sum() > 0:
                    # import ipdb; ipdb.set_trace()
                    costs = norm(rot_errors(gts['rotations'][mask_g], dts_rot[mask_d]))
                    costs += norm(np.linalg.norm(gts['translations'][mask_g][:, None] - dts_trans[mask_d][None], axis=2))
                    if dts_boxes is not None:
                        costs += norm(bbox_overlaps(gts['boxes'][mask_g], dts_boxes[mask_d]))
                    gts_tmp_ind, dts_tmp_ind = linear_sum_assignment(costs)
                    gts_ind += np.nonzero(mask_g)[0][gts_tmp_ind].tolist()
                    dts_ind += keep[mask_d][dts_tmp_ind].tolist()
        except Exception as e:
            print (e)
            import ipdb; ipdb.set_trace()
            print()
        gts_ind = np.asarray(gts_ind, dtype=np.int64)
        dts_ind = np.asarray(dts_ind, dtype=np.int64)
        return gts_ind, dts_ind
    
    def gt_instances_to_dict(self, gt_instances):
        gts = {}
        gts['labels'] = gt_instances.labels.detach().cpu().numpy()
        gts['boxes'] = gt_instances.boxes.detach().cpu().numpy().astype('float32')
        gts['translations'] = gt_instances.translations.detach().cpu().numpy().astype('float32')
        gts['rotations'] = gt_instances.rotations.detach().cpu().numpy().astype('float32')
        gts['image_id'] = gt_instances.image_id[0].detach().cpu().item()
        # import ipdb; ipdb.set_trace()
        return gts
    def outputs_to_dts(self, outputs, k):
        dts = {}
        dts['scores'] = outputs['pred_logits'][k].detach().squeeze().max(dim=-1).values.cpu().numpy()
        dts['boxes'] = outputs['pred_boxes'][k].detach().squeeze().cpu().numpy()
        dts['translations'] = outputs['pred_translations'][k].detach().squeeze().cpu().numpy()
        dts['rotations'] = rot6d2mat(outputs['pred_rotations'][k].detach()).squeeze().cpu().numpy()
        _labels = outputs['pred_logits'][k].detach().squeeze().max(dim=-1).indices.cpu().numpy()
        dts['labels'] = _labels
        # import ipdb; ipdb.set_trace()
        return dts
    def update(self, outputs, gt_instances):

        # results = self.prepare(predictions)
        num_frames = len(outputs['pred_logits'])
        for frame in range(num_frames):
            gt_frame = gt_instances[frame]
            gts = self.gt_instances_to_dict(gt_frame)
            dts = self.outputs_to_dts(outputs, frame)
            scores, dts_rot, dts_trans = dts['scores'], dts['rotations'], dts['translations']
            labels, gts_rot, gts_trans = gts['labels'], gts['rotations'], gts['translations']
            # import ipdb; ipdb.set_trace()
            if 'keypoints' in dts:
                dts_kpts, gts_kpts = dts['keypoints'], gts['keypoints']
            scene_id, im_id = int(gts['image_id']), int(gts['image_id'])
            # mat = {'poses': [], 'rois': []}
            errors = ["class", "score", "add", "adds", "te", "re"]  # "xyz", #'proj' FIXME ARUL proj metric
            image_errors = {}
            image_errors['class'] = labels
            image_errors['score'] = np.ones_like(labels) * -np.inf
            for error in errors:
                if error in ["add", "adds", "te", "re", "proj"]:
                    image_errors[error] = np.ones_like(labels) * np.inf
                elif error == "xyz":
                    image_errors[error] = np.ones((labels.size, 3), dtype=np.float64) * np.inf
                elif error in ["kpts", "kpts_z"]:
                    errors_kpts = np.ones((labels.size, self.model_ds.num_keypoints), dtype=np.float64) * np.inf
                    image_errors['kpts'] = errors_kpts                    
                    image_errors['kpts_z'] = np.ones_like(errors_kpts) * np.inf
            gts_ind, dts_ind = self.matching(dts, gts, self.threshold)
            dts_boxes = dts['boxes'] if 'boxes' in dts else gts['boxes'][gts_ind]
            for g_i, d_i in zip(gts_ind, dts_ind):
                # dts_trans[d_i] -= self.model_ds.offsets[labels[g_i] - 1]
                points = self.model_ds.points[labels[g_i] - 1]
                image_errors['score'][g_i] = scores[d_i]
                for error in image_errors:
                    if error == "add":
                        image_errors[error][g_i] = add(dts_rot[d_i], dts_trans[d_i], gts_rot[g_i], gts_trans[g_i], points)
                    elif error == "adds":
                        image_errors[error][g_i] = adi(dts_rot[d_i], dts_trans[d_i], gts_rot[g_i], gts_trans[g_i], points)
                    elif error == "te":
                        image_errors[error][g_i] = te(dts_trans[d_i], gts_trans[g_i])
                    elif error == "re":
                        image_errors[error][g_i] = re(dts_rot[d_i], gts_rot[g_i])
                    elif error == "proj":
                        image_errors[error][g_i] = reproj(gts['intrinsic'], dts_rot[d_i], dts_trans[d_i], gts_rot[g_i], gts_trans[g_i], points)
                    elif error == "xyz":
                        image_errors[error][g_i] = np.abs(dts_trans[d_i] - gts_trans[g_i])
                    elif error == "kpts":
                        image_errors[error][g_i] = np.linalg.norm(dts_kpts[d_i] - gts_kpts[g_i][..., :2], axis=1)
                    elif error == "kpts_z":
                        image_errors[error][g_i] = np.linalg.norm(dts_kpts[d_i] - gts_kpts[g_i][..., :2], axis=1)[np.argsort(gts_kpts[g_i][..., 2], kind='mergesort')]

            self.errors[im_id] = image_errors


    def synchronize_between_processes(self):
        pass

    def accumulate(self):
        metrics = ["AUC ADD@0.1d", "AUC ADDS@0.1d", "AUC TE@0.1d", "AUC ADD@0.1", "AUC ADDS@0.1", "AUC TE@0.1", "AR ADD@0.1d", "AR ADDS@0.1d", "AR TE@0.05d", "AR PROJ@5", "HIST RE", "AR RE@5"]  # "MEAN XYZ"
        errors_accumulated = {}
        for image_errors in self.errors.values():
            for name, errors in image_errors.items():
                errors_accumulated[name] = np.concatenate([errors_accumulated.get(name, [] if errors.ndim == 1 else np.zeros((0, errors.shape[1]))), errors], axis=0)

        # mm to meter
        for error in errors_accumulated:
            if error in ["add", "adds", "te", "xyz"]:
                errors_accumulated[error] /= 1000

        num_classes = len(self.classes)  # first class is all
        self.metrics_per_class = OrderedDict()
        # self.plots_per_class = OrderedDict()
        errors_accumulated['class'] = np.int64(errors_accumulated['class'])
        for metric in metrics:
            mtrc, err = metric.split(' ')
            err = err.lower()
            info = ''
            if '@' in metric:
                err, info = err.split('@')
                threshold = float(info.replace('d', ''))
            if mtrc == "HIST":
                self.metrics_per_class[metric] = np.zeros((num_classes, len(self.bins) - 1), dtype=np.float64)
            elif mtrc == "MEAN":
                self.metrics_per_class[metric] = np.zeros((num_classes, errors_accumulated[err].shape[1]), dtype=np.float64)
            else:
                self.metrics_per_class[metric] = np.zeros(num_classes, dtype=np.float64)
                # if mtrc == "AUC":
                #     self.plots_per_class[metric] = [[]] * num_classes

            for i, k in enumerate(self.classes.keys()):
                cls_mask = np.ones(errors_accumulated['class'].shape, dtype=bool) if i == 0 else errors_accumulated['class'] == k
                errors = errors_accumulated[err][cls_mask].copy()
                diameter = self.model_ds.diameters[errors_accumulated['class'][cls_mask] - 1] if 'd' in info else 1
                if '@' in metric:
                    errors[errors > threshold * diameter] = np.inf

                if mtrc == "AR":
                    self.metrics_per_class[metric][i] = np.isfinite(errors).sum() / errors.shape[0]
                elif mtrc == "AUC":
                    ap, rec, prec = self.compute_ap(errors)
                    self.metrics_per_class[metric][i] = ap
                    # self.plots_per_class[metric][i] = [rec, prec]
                elif mtrc == "MISS":
                    self.metrics_per_class[metric][i] = np.isinf(errors).sum()
                elif mtrc == "HIST":
                    self.metrics_per_class[metric][i] = np.histogram(errors[np.isfinite(errors)], bins=self.bins, range=(min(self.bins), max(self.bins)), density=False)[0] / np.isfinite(errors).sum()
                else:
                    self.metrics_per_class[metric][i] = np.mean(errors[np.isfinite(errors)], axis=0)

    def summarize(self):
        # tabulate it
        metrics = [metric for metric in self.metrics_per_class.keys() if 'MEAN' not in metric and 'HIST' not in metric]
        metrics_per_category = []
        for i, name in enumerate(self.classes.values()):
            row = [f"{name}"]
            for metric in metrics:
                row.append(self.metrics_per_class[metric][i])
            metrics_per_category.append(row)
        table = tabulate(
            metrics_per_category,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category"] + [f"{metric}" for metric in metrics],
            numalign="left",
        )
        print("Per-category:\n" + table)

        if "HIST RE" in self.metrics_per_class:
            rot_errs_per_category = [(f"{name}", *self.metrics_per_class["HIST RE"][i]) for i, name in enumerate(self.classes.values())]
            table = tabulate(
                rot_errs_per_category,
                tablefmt="pipe",
                floatfmt=".3f",
                headers=["category"] + [f"{self.bins[i]}-{self.bins[i + 1]}" for i in range(len(self.bins) - 1)],
                numalign="left",
            )
            print("Per-category pose rotation angle error percentage:\n" + table)

        for metric in self.metrics_per_class:
            if 'ADDS' in metric:
                sym = np.zeros(len(self.classes), dtype=np.bool)
                for i, k in enumerate(self.classes.keys()):
                    if i > 0 and k in self.model_ds.sym_classes:
                        sym[i] = True
                add_s = np.concatenate([self.metrics_per_class[metric][sym], self.metrics_per_class[metric.replace('ADDS', 'ADD')][~sym][1:]]).mean()
                print("{}: {:.3f}".format(metric.replace('ADDS', 'ADD(-S)'), add_s))

        if 'MEAN XYZ' in self.metrics_per_class:
            for i, name in enumerate(self.classes.values()):
                print(f"MEAN XYZ {name}:", self.metrics_per_class['MEAN XYZ'][i])

        self.stats = [self.metrics_per_class["AR ADD@0.1d"].item(0), self.metrics_per_class["AR ADDS@0.1d"].item(0)]

    def prepare(self, predictions):
        results = {}
        for img_id, pred in predictions.items():
            data = {
                'scores': pred['scores'].cpu().numpy(),
                'labels': pred['labels'].cpu().numpy()
            }
            if 'boxes' in pred:
                data['boxes'] = pred['boxes'].cpu().numpy()
            if 'keypoints' in pred:
                data['keypoints'] = pred['keypoints'].cpu().numpy()
                gts = self.groundtruths[img_id]
                K = gts['intrinsic']
                rotations, translations = [], []
                for score, label, keypoints in zip(data['scores'], data['labels'], data['keypoints']):
                    if score >= self.threshold:
                        # keypoints += np.random.uniform(size=keypoints.shape) * 25
                        R, t = pnp(self.model_ds.keypoints[label - 1, :-1], keypoints, K, method=cv2.SOLVEPNP_EPNP, ransac=True, ransac_reprojErr=3, ransac_iter=100) 
                        t = t[:, 0]
                    else:
                        R, t = np.zeros((3, 3), dtype=np.float32), np.zeros(3, dtype=np.float32)
                    rotations.append(R.astype(np.float32))
                    translations.append(t.astype(np.float32))
                rotations = np.stack(rotations)
                translations = np.stack(translations)
            if 'rotations' in pred:
                rotations = pred['rotations'].cpu().numpy()
            if 'translations' in pred:
                translations = pred['translations'].cpu().numpy()

            data['rotations'] = rotations
            data['translations'] = translations
            results[img_id] = data
        return results

    @staticmethod
    def compute_ap(errors):
        rec = np.sort(errors, kind='mergesort')
        prec = np.cumsum(np.ones(errors.size, dtype=np.float64)) / errors.size
        inds = np.isfinite(rec)
        if inds.sum() == 0:
            return 0.0, rec, prec
        ap = voc_ap(rec[inds], prec[inds])
        return ap, rec, prec
