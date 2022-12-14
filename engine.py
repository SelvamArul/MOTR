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
Train and eval functions used in main.py
"""
import cv2
import math
import numpy as np
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from util import box_ops

from torch import Tensor
from util.plot_utils import draw_boxes, draw_ref_pts, image_hwc2chw
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher, data_dict_to_cuda
from object_info import obj_id_to_name

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    # import ipdb; ipdb.set_trace()
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes() 
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_ddetr(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    # import ipdb; ipdb.set_trace()
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        data_dict = data_dict_to_cuda(data_dict, device)
        samples = data_dict['imgs']
        targets = data_dict['gt_instances']
        outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        # print("iter {} after model".format(cnt-1))
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        # gather the stats from all processes
        
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

import wandb

def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, exp_name: str,
                    max_norm: float = 0, 
                    do_visualize=False, profile: bool = False):
    if do_visualize:
        if epoch == 0:
            wandb.login()
            wandb.init(project='setup_ddp',  #"cuda12_temporal_pose",
                    name=exp_name,
                    config={"lr":optimizer.param_groups[0]["lr"]})
    
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    # import ipdb; ipdb.set_trace()
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    
    x_count = 0
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        data_dict = data_dict_to_cuda(data_dict, device)
        outputs = model(data_dict)
        import ipdb; ipdb.set_trace()
        loss_dict = criterion(outputs, data_dict)
        # print("iter {} after model".format(cnt-1))
        weight_dict = criterion.weight_dict
        _loss_dict = {k:v.item() for (k, v) in loss_dict.items() }
        _losses = [_loss_dict[k] * weight_dict[k] for k in _loss_dict.keys() if k in weight_dict]
        _lll = {k:_loss_dict[k] * weight_dict[k] for k in _loss_dict.keys() if k in weight_dict}
        # import ipdb; ipdb.set_trace()
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
        log_loss = {k:v.item() for k,v in loss_dict_reduced_scaled.items()}
        if do_visualize:
            frame_0_sum =0
            frame_1_sum =0
            frame_2_sum =0
            frame_3_sum =0
            frame_4_sum =0
            frame_5_sum =0
            for k, v in log_loss.items():
                if 'frame_0' in k:
                    frame_0_sum += v
                elif 'frame_1' in k:
                    frame_1_sum += v
                elif 'frame_2' in k:
                    frame_2_sum += v
                elif 'frame_3' in k:
                    frame_3_sum += v
                elif 'frame_4' in k:
                    frame_4_sum += v
                elif 'frame_5' in k:
                    frame_5_sum += v
            log_loss['frame_0_sum'] = frame_0_sum
            log_loss['frame_1_sum'] = frame_1_sum
            log_loss['frame_2_sum'] = frame_2_sum
            log_loss['frame_3_sum'] = frame_3_sum
            log_loss['frame_4_sum'] = frame_4_sum
            log_loss['frame_5_sum'] = frame_5_sum

            log_loss_trim  = {}
            for k, v in log_loss.items():
                if 'aux' not in k:
                    log_loss_trim[k] = v
            # import ipdb; ipdb.set_trace()
            wandb.log(log_loss_trim)
        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        # gather the stats from all processes
        if profile:
            if x_count == 10:
                break
            else:
                x_count += 1
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

from rich import print as rprint
from collections import Counter
from datasets.pose_eval import PoseEvaluator
@torch.no_grad()
def eval_pose(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    models_ds : Iterable,
                    device: torch.device,
                    exp_name: str,
                    do_visualize = False):
    if do_visualize:
        import wandb
        wandb.login()
        wandb.init(project="int_debug",
                name="bbox_viz")
    obj_id_dict = data_loader.dataset.obj_id_to_name
    pose_evaluator = PoseEvaluator(models_ds, obj_id_dict, exp_name)
    x_count = 1
    def compute_cardinality_error(outputs, data_dict, x_count):
        bs = len(outputs['pred_logits'])
        _cardinality_error = 0
        for i in range(bs):
            _pred_idx = outputs['pred_logits'][i][0].max(dim=-1).indices
            _pl = outputs['pred_logits'][i][0]
            pred_idx = _pred_idx[_pred_idx != 22]
            _pp = torch.nn.functional.softmax(_pl, dim=-1)
            # print ("Probs ", _pp[pred_idx].max(dim=-1))
            _ppi = outputs['pred_logits'][i][0].max(dim=-1, keepdim=True).indices
            _pm = _ppi != 22
            import ipdb; ipdb.set_trace()
            # print (torch.gather(_pp, -1, _ppi)[_pm])
            gt_idx = data_dict['gt_instances'][i].get_fields()['labels']
            pred_idx = pred_idx.sort().values
            gt_idx = gt_idx.sort().values
            gt_list = gt_idx.tolist()
            pred_list = pred_idx.tolist()
            gt_cnt = Counter(gt_list)
            pred_cnt = Counter(pred_list)
            missing_ids = [k for k, counts in pred_cnt.items() if gt_cnt[k] != counts]
            cardinality_error = len(missing_ids) 
            # import ipdb ; ipdb.set_trace()
            _cardinality_error += cardinality_error

            # if cardinality_error == 0:
            #     rprint(x_count, gt_idx.tolist(), pred_idx.tolist(), f"[bold red] {cardinality_error} [/bold red]   :thumbsup:")
            #     print ("Ok")
            #     # pass
            # else:
            #     rprint(x_count, gt_idx.tolist(), pred_idx.tolist(), f"[bold red] {cardinality_error} [/bold red]   :warning:")
            #     rprint(missing_ids)
            # import ipdb; ipdb.set_trace()
            # print()
        return _cardinality_error
    model.train() # TODO
    criterion.train()
    epoch = 0
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    # for data_dict in metric_logger.log_every(data_loader, print_freq, header):
    _card_error = 0
    @torch.no_grad()
    def predictions_to_wandb_dict(output, data_dict):
        bs = len(outputs['pred_logits'])

        for i in range(bs):
            # import ipdb; ipdb.set_trace()
            _pred_idx = outputs['pred_logits'][i][0].max(dim=-1).indices
            _pl = outputs['pred_logits'][i][0]
            pred_idx = _pred_idx[_pred_idx != 22]
            _pp = torch.nn.functional.softmax(_pl, dim=-1)
            # print ("Probs ", _pp[pred_idx].max(dim=-1))
            _ppi = outputs['pred_logits'][i][0].max(dim=-1, keepdim=True).indices
            _pm = _ppi != 22
            # import ipdb; ipdb.set_trace()
            pred_probs = torch.gather(_pp, -1, _ppi)[_pm]
            pred_bboxes = outputs['pred_boxes'][i][0][_pm.squeeze()]
            img = (data_dict['orig_img'][i].permute(1, 2, 0) * 255).cpu().int()
            H, W = img.shape[0], img.shape[1] 
            pred_dict = []
            for idx in range(len(pred_idx)):
                bbox = pred_bboxes[idx].cpu().tolist()
                bbox[0::2] = map(lambda x: x * W, bbox[0::2])
                bbox[1::2] = map(lambda x: x * H, bbox[1::2])
                bbox = [int(x) for x in bbox]
                pred_dict.append(
                        {
                            'position': {
                                         'middle' : bbox[:2],
                                         'width': bbox[2],
                                         'height': bbox[3],
                                        },
                            'domain': 'pixel',
                            'class_id' : pred_idx[idx].item(),
                            'box_caption': obj_id_dict[pred_idx[idx].item()],
                            'scores': {'acc': pred_probs[idx].item()}
                        })
            gt_cls_idxs = data_dict['gt_instances'][i].get_fields()['labels'].tolist()
            gt_boxes = data_dict['gt_instances'][i].get_fields()['boxes']
            gt_dict = []
            for idx in range(len(data_dict['gt_instances'][i])):
                cls_idx = gt_cls_idxs[idx]
                bbox = gt_boxes[idx].cpu().tolist()
                bbox[0::2] = map(lambda x: x * W, bbox[0::2])
                bbox[1::2] = map(lambda x: x * H, bbox[1::2])
                bbox = [int(x) for x in bbox]
                gt_dict.append(
                        {
                            'position': {
                                'middle' : bbox[:2],
                                'width': bbox[2],
                                'height': bbox[3],
                                        },
                            'domain': 'pixel',
                            'class_id' : gt_cls_idxs[idx],
                            'box_caption': obj_id_dict[cls_idx],
                            })
            # break
            pred_vis = wandb.Image(img.numpy(), boxes ={'predictions' : {"box_data": pred_dict, 'class_labels': obj_id_dict}})
            gt_vis = wandb.Image(img.numpy(), boxes ={'predictions' : {"box_data": gt_dict, 'class_labels': obj_id_dict}})
            wandb.log({"pred_viz": pred_vis})
            wandb.log({"gt_viz": gt_vis})


            # img_vis = wandb.Image(img,
            #         boxes = 

    for data_dict in data_loader:   
        data_dict = data_dict_to_cuda(data_dict, device)
        outputs = model(data_dict)
        # if do_visualize:
        #     predictions_to_wandb_dict(outputs, data_dict)
        # _card_error += compute_cardinality_error(outputs, data_dict, x_count)
        loss_dict = criterion(outputs, data_dict)
        # import ipdb; ipdb.set_trace()
        pose_evaluator.update(outputs, data_dict['gt_instances'])


        # print("iter {} after model".format(cnt-1))
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # import ipdb; ipdb.set_trace()
        # x_count += 1
    # print ("Total error ", _card_error)
    # import ipdb; ipdb.set_trace()

    if pose_evaluator is not None:
        pose_evaluator.accumulate()
        pose_evaluator.summarize()
    import ipdb; ipdb.set_trace()

    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # return #{k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return pose_evaluator.metrics_per_class 



def debug_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    # model.train()
    # criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    # import ipdb; ipdb.set_trace()
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    

    model.eval()
    criterion.eval()
    x_count = 0
    for _data_dict in metric_logger.log_every(data_loader, print_freq, header):
        if x_count == 0:
            data_dict = _data_dict
            data_dict = data_dict_to_cuda(data_dict, device)
            x_count += 1

        model.eval() 
        _outputs = []
        _eval_outputs = []
        _train_outputs = []

        for ii in range(1):
            outputs = model(data_dict)
            loss_dict = criterion(outputs, data_dict)
            _outputs.append(outputs)

            # debug eval mode
            model.eval()
            criterion.eval()
            eval_outputs = model(data_dict)
            eval_loss_dict = criterion(eval_outputs, data_dict)
            _eval_outputs.append(eval_outputs)

            # model.train()
            # criterion.train()
            train_outputs = model(data_dict)
            _train_outputs.append(train_outputs)
            print ("pred_logits ", outputs['pred_logits'][0][0][0])
            print ("loss_dict", loss_dict)
        # loss_dict = criterion(outputs, data_dict)
        # import ipdb; ipdb.set_trace()
        # print ("engine.py:170")


        # print("iter {} after model".format(cnt-1))
        weight_dict = criterion.weight_dict
        # print ("weight_dict ", weight_dict)
        import ipdb; ipdb.set_trace()
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # optimizer.zero_grad()
        # losses.backward()
        # if max_norm > 0:
        #     grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # else:
        #     grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        # optimizer.step()

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # metric_logger.update(grad_norm=grad_total_norm)
        # gather the stats from all processes
        print ("Loss value ",loss_value)
        model.train()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def eval_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable,
                    device: torch.device, epoch: int):
    # model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        data_dict = data_dict_to_cuda(data_dict, device)
        outputs = model(data_dict)

        # import ipdb; ipdb.set_trace()
        loss_dict = criterion(outputs, data_dict)
        # print("iter {} after model".format(cnt-1))
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        # gather the stats from all processes
        # print ('Loss value ', loss_value)
        # print ('loss_dict_reduced_scaled ', loss_dict_reduced_scaled)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def visualize_predictions_mot(model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()
    
    for data_dict in data_loader:
        data_dict = data_dict_to_cuda(data_dict, device)
        outputs = model(data_dict)
        import ipdb; ipdb.set_trace()
        print ("vis  ")

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, model_ds, device, cuda_filter, output_dir):
    # model.eval()
    # criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = OrderedDict({k: k for k in ('bbox', 'segm', 'keypoints') if k in postprocessors.keys()})
    coco_evaluator = CocoEvaluator(base_ds, tuple(iou_types.keys()))
    # coco_evaluator.coco_eval[iou_types['segm']].params.iouThrs = [0, 0.1, 0.5, 0.75]
    if 'keypoints' in iou_types:
        coco_evaluator.coco_eval[iou_types['keypoints']].params.kpt_oks_sigmas = np.ones(model_ds.num_keypoints) * .1

    pose_evaluator = None
    if 'pose' in postprocessors.keys():
        pose_evaluator = PoseEvaluator(base_ds, model_ds, output_dir=os.path.join(output_dir, "pose_eval"))

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    iter = 0
    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) if k in cuda_filter else v for k, v in t.items()} for t in targets]

        outputs = model(samples, targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'pose' in postprocessors.keys():
            target_intrinsics = torch.stack([t["intrinsic"] for t in targets], dim=0)
            results = postprocessors['pose'](results, outputs, orig_target_sizes, target_intrinsics)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if pose_evaluator is not None:
            pose_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        # if iter == 2:
        #     break
        # iter += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    # if pose_evaluator is not None:
    #     pose_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        coco_evaluator.summarize_per_category()
    if pose_evaluator is not None:
        pose_evaluator.accumulate()
        pose_evaluator.summarize()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
        if 'keypoints' in postprocessors.keys():
            stats['coco_eval_keypoints'] = coco_evaluator.coco_eval['keypoints'].stats.tolist()
    if pose_evaluator is not None:
        stats['pose_eval'] = pose_evaluator.stats
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator 
