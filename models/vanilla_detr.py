# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import math

import kornia
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.pose_ops import rot6d2mat, affine_transform
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, accuracy,
                       get_world_size, interpolate, is_dist_avail_and_initialized)

from .vanilla_backbone import build_backbone
from .vanilla_matcher import build_matcher
from .pose import (cross_ratio, trans_from_centroid, axangle_rotate, rotate,
                   calc_dist, PostProcessPose, BPnP, BPnP_fast, project)
# from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
#                            dice_loss, sigmoid_focal_loss)
from .vanilla_transformer import build_transformer


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, box=True, keypoint=False, num_keypoints=32, pose=False, num_rot_params=6, rot_type="reg", trans_type="xyz", bwl=False, aux_loss=False, weight_dict=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            bwl: bayesian weight learning
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_queries = num_queries
        self.box = box
        self.keypoint = keypoint
        self.rot_type = rot_type
        self.trans_type = trans_type
        self.pose = pose
        self.bwl = bwl
        self.aux_loss = aux_loss

        hidden_dim = transformer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # nn.init.constant_(self.class_embed.bias, bias_init_with_prob(0.01))  # focal loss
        if box:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if keypoint:
            # self.keypoint_embed = MLP(hidden_dim, 4096, 2 * num_keypoints, 3)
            self.keypoint_embed_x = MLP(hidden_dim, 1024, num_keypoints, 3)
            self.keypoint_embed_y = MLP(hidden_dim, 1024, num_keypoints, 3)
        if pose:
            if "reg" == rot_type:
                self.rotation_embed = MLP(hidden_dim, hidden_dim, num_rot_params, 3)
                elif "pnp" == rot_type:
                    assert keypoint
                    self.rotation_embed = MLP(2 * num_keypoints, 1024, num_rot_params, 6, 0.5)  # lifter
                if trans_type != "kpts":
                    self.translation_embed = MLP(hidden_dim, hidden_dim, 3, 3)
            self.bayesian_weights = None
            if bwl:
                self.bayesian_weights = nn.ParameterDict({k: nn.Parameter(torch.tensor(v, dtype=torch.float32)) for k, v in weight_dict.items()})

        def forward(self, samples: NestedTensor, query_embed, targets=None):
            """ The forward expects a NestedTensor, which consists of:
                   - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
                   - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

                It returns a dict with the following elements:
                   - "pred_logits": the classification logits (including no-object) for all queries.
                                    Shape= [batch_size x num_queries x (num_classes + 1)]
                   - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                                   (center_x, center_y, height, width). These values are normalized in [0, 1],
                                   relative to the size of each individual image (disregarding possible padding).
                                   See PostProcess for information on how to retrieve the unnormalized bounding box.
                   - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                    dictionnaries containing the two above keys for each decoder layer.
            """
            if isinstance(samples, (list, torch.Tensor)):
                samples = nested_tensor_from_tensor_list(samples)
            features, pos = self.backbone(samples)

            src, mask = features[-1].decompose()
            assert mask is not None
            if self.transformer.name == "smca":
                h_w = torch.stack([
                    torch.stack([t["size"] for t in targets])[:, 1],
                    torch.stack([t["size"] for t in targets])[:, 0]
                ], dim=-1)
                h_w = h_w.unsqueeze(0)
                hs, points = self.transformer(self.input_proj(src), mask, query_embed.weight, pos[-1], h_w)
            else:
                hs = self.transformer(self.input_proj(src), mask, query_embed.weight, pos[-1])[0]
            '''
            outputs_class = self.class_embed(hs)
            out = {'pred_logits': outputs_class[-1]}
            outputs_coord = None
            if self.box:
                outputs_coord = self.bbox_embed(hs)
                if self.transformer.name == "smca":
                    num_decoder = hs.shape[0]
                    points = points.unsqueeze(0).repeat(num_decoder, 1, 1, 1)
                    outputs_coord[..., :2] = outputs_coord[..., :2] + points
                outputs_coord = outputs_coord.sigmoid()
                out['pred_boxes'] = outputs_coord[-1]
            outputs_keypoint = None
            if self.keypoint:
                # outputs_keypoint = self.keypoint_embed(hs)
                outputs_keypoint = torch.stack([self.keypoint_embed_x(hs), self.keypoint_embed_y(hs)], dim=-1)
                out['pred_keypoints'] = outputs_keypoint[-1]
            outputs_rotation, outputs_translation = None, None
            if self.pose:
                if self.rot_type == "reg":
                    outputs_rotation = self.rotation_embed(hs)
                    out['pred_rotations'] = outputs_rotation[-1]
                elif self.rot_type == "pnp":
                    kpts_flattened = outputs_keypoint.flatten(-2, -1)
                    outputs_rotation = self.rotation_embed(kpts_flattened)
                    out['pred_rotations'] = outputs_rotation[-1]
                if self.trans_type != "kpts":
                    outputs_translation = self.translation_embed(hs)
                    out['pred_translations'] = outputs_translation[-1]
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_keypoint, outputs_rotation, outputs_translation)
            '''
            return hs

        @torch.jit.unused
        def _set_aux_loss(self, outputs_class, outputs_coord=None, outputs_keypoint=None, outputs_rotation=None, outputs_translation=None):
            # this is a workaround to make torchscript happy, as torchscript
            # doesn't support dictionary with non-homogeneous values, such
            # as a dict having both a Tensor and a list.
            outputs = [{'pred_logits': a} for a in outputs_class[:-1]]
            if outputs_coord is not None:
                for b, out in zip(outputs_coord[:-1], outputs):
                    out['pred_boxes'] = b
            if outputs_keypoint is not None:
                for c, out in zip(outputs_keypoint[:-1], outputs):
                    out['pred_keypoints'] = c
            if outputs_rotation is not None:
                for d, out in zip(outputs_rotation[:-1], outputs):
                    out['pred_rotations'] = d
            if outputs_translation is not None:
                for e, out in zip(outputs_translation[:-1], outputs):
                    out['pred_translations'] = e
            return outputs


    class SetCriterion(nn.Module):
        """ This class computes the loss for DETR.
        The process happens in two steps:
            1) we compute hungarian assignment between ground truth boxes and the outputs of the model
            2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
        """
        def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, bayesian_weights=None, focal_loss=False, models_kpts=None, rot_rep="ego_rot6d", trans_type="xyz", models_pts=None, sym_classes=None):
            """ Create the criterion.
            Parameters:
                num_classes: number of object categories, omitting the special no-object category
                matcher: module able to compute a matching between targets and proposals
                weight_dict: dict containing as key the names of the losses and as values their relative weight.
                eos_coef: relative classification weight applied to the no-object category
                losses: list of all the losses to be applied. See get_loss for list of available losses.
            """
            super().__init__()
            self.num_classes = num_classes
            self.matcher = matcher
            self.weight_dict = weight_dict
            self.eos_coef = eos_coef
            self.losses = losses
            self.bayesian_weights = bayesian_weights
            self.focal_loss = focal_loss
            self.rot_rep = rot_rep
            self.trans_type = trans_type

            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer('empty_weight', empty_weight)
            if models_pts is not None:
                assert num_classes - 1 == models_pts.size(0)
                self.register_buffer('models_points', models_pts)
            if models_kpts is not None:
                assert num_classes - 1 == models_kpts.size(0)
                self.register_buffer('models_keypoints', models_kpts)
            if sym_classes is not None:
                self.register_buffer('symmetric_classes', torch.as_tensor(sym_classes))
            # self.bpnp = BPnP_fast.apply

        def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
            """Classification loss (NLL)
            targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
            """
            assert 'pred_logits' in outputs
            src_logits = outputs['pred_logits']

            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)])
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o

            if self.focal_loss:
                src_logits_flattened = src_logits.flatten(0, 1)
                target_classes = target_classes.flatten(0, 1)
                pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
                labels = torch.zeros_like(src_logits_flattened)
                labels[pos_inds, target_classes[pos_inds]] = 1
                loss_ce = sigmoid_focal_loss(src_logits_flattened, labels).sum() / num_boxes
            else:
                loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

            losses = {'loss_ce': loss_ce}
            if log:
                # TODO this should probably be a separate loss, not hacked in this one here
                losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
                # losses['class_error_wo_no'] = 100 - accuracy(src_logits[idx][..., :-1], target_classes_o)[0]
            return losses

        @torch.no_grad()
        def loss_cardinality(self, outputs, targets, indices, num_boxes):
            """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
            This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
            """
            pred_logits = outputs['pred_logits']
            device = pred_logits.device
            tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets], device=device)
            # Count the number of predictions that are NOT "no-object" (which is the last class)
            card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
            card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
            losses = {'cardinality_error': card_err}
            return losses

        def loss_boxes(self, outputs, targets, indices, num_boxes):
            """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
               targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
               The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
            """
            assert 'pred_boxes' in outputs
            idx = self._get_src_permutation_idx(indices)
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)])

            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
            losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes)))
            losses['loss_giou'] = loss_giou.sum() / num_boxes
            return losses

        def loss_keypoints(self, outputs, targets, indices, num_boxes):
            assert 'pred_keypoints' in outputs
            idx = self._get_src_permutation_idx(indices)
            # target_classes = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)])
            # intrinsics = torch.cat([t['intrinsic'].expand(len(t['labels']), 3, 3) for t in targets])
            # scales = torch.cat([t['size'].expand(len(t['labels']), 2) for t in targets])
            # rotations = torch.cat([t['rotations'][i] for t, (_, i) in zip(targets, indices)])
            # rotations = kornia.rotation_matrix_to_angle_axis(rotations)
            # translations = torch.cat([t['translations'][i] for t, (_, i) in zip(targets, indices)])
            # poses = torch.cat([rotations, translations], dim=-1)

            src_keypoints = outputs['pred_keypoints'][idx]
            target_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)])

            # bpnp_losses = []
            # for i in range(poses.size(0)):
            #     pts2d_gt = target_keypoints[i] * scales[i]
            #     pts2d = src_keypoints[i] * scales[i]
            #     pts3d_gt = self.models_keypoints[target_classes[i] - 1]
            #     K = intrinsics[i]
            #     P_out = self.bpnp(pts2d[None], pts3d_gt, K)
            #     pts2d_pro = project(P_out, pts3d_gt, K)
            #     pts2d_pro = pts2d_pro[0]  # / scales[i]
            #     # pts2d = src_keypoints[i]
            #     bpnp_losses.append(F.mse_loss(pts2d_pro, pts2d_gt, reduction='none') + F.mse_loss(pts2d, pts2d_gt, reduction='none'))

            # loss_keypoint = torch.stack(bpnp_losses).mean(dim=(1, 2))

            mask = torch.any((target_keypoints < 0) | (target_keypoints > 1), dim=2).unsqueeze(2)
            loss_keypoint = F.l1_loss(src_keypoints, target_keypoints[:, :-1], reduction='none') * mask[:, :-1]

            cr_sqr, cr_mask = cross_ratio(src_keypoints, 0.1)
            cr_sqr = cr_sqr / (4 / 3) ** 2
            cr_loss = F.smooth_l1_loss(cr_sqr, torch.ones_like(cr_sqr.detach()), reduction='none') * cr_mask
            losses = {
                'loss_keypoint': loss_keypoint.sum() / num_boxes,  # loss_keypoint.mean()
                'loss_cr': cr_loss.sum() / num_boxes  # cr_loss.sum() / cr_mask.sum()
            }
            return losses

        def loss_poses(self, outputs, targets, indices, num_boxes):
            assert 'pred_rotations' in outputs or 'pred_translations' in outputs
            idx = self._get_src_permutation_idx(indices)

            loss_pose = []
            if 'pred_translations' in outputs:
                src_translations = outputs['pred_translations'][idx] * 1000  # meter to mm
                if "cxcy" in self.trans_type:
                    intrinsics = torch.cat([t['intrinsic'].expand(len(t['labels']), 3, 3) for t in targets])
                    src_translations = trans_from_centroid(src_translations[:, :2], src_translations[:, 2], intrinsics)
                target_translations = torch.cat([t['translations'][i] for t, (_, i) in zip(targets, indices)])

                trans_dists = torch.norm(src_translations - target_translations, p=1, dim=1)  # F.l1_loss(src_translations, target_translations, reduction='none')
                loss_pose.append(trans_dists.sum())

            if 'pred_rotations' in outputs:
                target_classes = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)])

                src_rotations = outputs['pred_rotations'][idx]
                target_rotations = torch.cat([t['rotations'][i] for t, (_, i) in zip(targets, indices)])

                points = self.models_points[target_classes - 1]
                if "axangle" in self.rot_rep:
                    src_transformations = axangle_rotate(points, src_rotations)
                    target_transformations = axangle_rotate(points, target_rotations)
                elif "rot6d" in self.rot_rep:
                    src_transformations = rotate(points, rot6d2mat(src_rotations))
                    target_transformations = rotate(points, target_rotations)
                elif "mat" in self.rot_rep:
                    src_transformations = rotate(points, src_rotations.view(-1, 3, 3))
                    target_transformations = rotate(points, target_rotations)
                # src_transformations = src_transformations + src_translations.unsqueeze(1)
                # target_transformations = target_transformations + target_translations.unsqueeze(1)

                diffs = target_classes[:, None] - self.symmetric_classes[None]
                asyms = target_classes > 0
                asyms[torch.nonzero(diffs == 0)[:, 0]] = False
                rot_dists = calc_dist(src_transformations, target_transformations, asyms, p=1)
                loss_pose.append(rot_dists.mean(1).sum())

            losses = {
                'loss_pose': sum(loss_pose) / num_boxes
            }
            return losses

        def loss_masks(self, outputs, targets, indices, num_boxes):
            """Compute the losses related to the masks: the focal loss and the dice loss.
               targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
            """
            assert 'pred_masks' in outputs
            src_idx = self._get_src_permutation_idx(indices)
            tgt_idx = self._get_tgt_permutation_idx(indices)
            src_masks = outputs['pred_masks']
            src_masks = src_masks[src_idx]
            masks = [t['masks'] for t in targets]
            # TODO use valid to mask invalid areas due to padding in loss
            target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
            target_masks = target_masks.to(src_masks)
            target_masks = target_masks[tgt_idx]

            # upsample predictions to the target size
            src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                    mode='bilinear', align_corners=False)
            src_masks = src_masks[:, 0].flatten(1)

            target_masks = target_masks.flatten(1)
            target_masks = target_masks.view(src_masks.shape)
            losses = {
                'loss_mask': sigmoid_focal_loss(src_masks, target_masks).mean(1).sum() / num_boxes,
                'loss_dice': dice_loss(src_masks, target_masks, num_boxes),
            }
            return losses

        def _get_src_permutation_idx(self, indices):
            # permute predictions following indices
            batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
            src_idx = torch.cat([src for (src, _) in indices])
            return batch_idx, src_idx

        def _get_tgt_permutation_idx(self, indices):
            # permute targets following indices
            batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
            tgt_idx = torch.cat([tgt for (_, tgt) in indices])
            return batch_idx, tgt_idx

        def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
            loss_map = {
                'labels': self.loss_labels,
                'cardinality': self.loss_cardinality,
                'boxes': self.loss_boxes,
                'keypoints': self.loss_keypoints,
                'poses': self.loss_poses,
                'masks': self.loss_masks
            }
            assert loss in loss_map, f'do you really want to compute {loss} loss?'
            return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

        def forward(self, outputs, targets):
            """ This performs the loss computation.
            Parameters:
                 outputs: dict of tensors, see the output specification of the model for the format
                 targets: list of dicts, such that len(targets) == batch_size.
                          The expected keys in each dict depends on the losses applied, see each loss' doc
            """
            outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

            # Retrieve the matching between the outputs of the last layer and the targets
            indices = self.matcher(outputs_without_aux, targets)

            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

            # Compute all the requested losses
            losses = {}
            for loss in self.losses:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    indices = self.matcher(aux_outputs, targets)
                    for loss in self.losses:
                        if loss == 'masks':  # or loss == 'keypoints':  # bpnp
                            # Intermediate masks losses are too costly to compute, we ignore them.
                            continue
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs = {'log': False}
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

            return losses


    class PostProcess(nn.Module):
        """ This module converts the model's output into the format expected by the coco api"""
        def __init__(self, focal_loss=False):
            super().__init__()
            self.focal_loss = focal_loss

        @torch.no_grad()
        def forward(self, outputs, target_sizes, target_transforms=None):
            """ Perform the computation
            Parameters:
                outputs: raw outputs of the model
                target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                              For evaluation, this must be the original image size (before any data augmentation)
                              For visualization, this should be the image size after data augment, but before padding
            """
            out_logits = outputs['pred_logits'].cpu()

            assert len(out_logits) == len(target_sizes)
            assert target_sizes.shape[1] == 2

            batch_size, num_queries, num_classes = out_logits.shape
            if self.focal_loss:
                prob = out_logits.sigmoid()
                scores, topk_indices = torch.topk(prob.view(batch_size, -1), num_queries, dim=1, sorted=False)
                topk_boxes = topk_indices // num_classes
                labels = topk_indices % num_classes
            else:
                prob = F.softmax(out_logits, -1)
                scores, labels = prob[..., :-1].max(-1)

            results = [{'scores': s, 'labels': l} for s, l in zip(scores, labels)]

            if 'pred_boxes' in outputs:
                out_boxes = outputs['pred_boxes'].cpu()
                assert len(out_boxes) == len(target_sizes)

                # convert to [x0, y0, x1, y1] format
                boxes = box_ops.box_cxcywh_to_xyxy(out_boxes)
                # and from relative [0, 1] to absolute [0, height] coordinates
                img_h, img_w = target_sizes.unbind(1)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
                boxes = boxes * scale_fct[:, None, :]
                if self.focal_loss:
                    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

                for result, b in zip(results, boxes):  # target_transforms
                    # boxes = affine_transform(b.reshape(-1, 2, 2), t)
                    # tl, br = boxes.min(1)[0], boxes.max(1)[0]
                    result['boxes'] = b  # torch.cat([tl, br], dim=1)

            if 'pred_keypoints' in outputs:
                out_keypoints = outputs['pred_keypoints'].cpu()
                assert len(out_keypoints) == len(target_sizes)

                # and from relative [0, 1] to absolute [0, height] coordinates
                img_h, img_w = target_sizes.unbind(1)
                scale_fct = torch.stack([img_w, img_h], dim=1)  # scale.repeat(2)
                keypoints = out_keypoints * scale_fct[:, None, None, :]

                for result, k in zip(results, keypoints):  # target_transforms
                    result['keypoints'] = k  # affine_transform(k, t)

            return results


    class MLP(nn.Module):
        """ Very simple multi-layer perceptron (also called FFN)"""

        def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0):
            super().__init__()
            self.num_layers = num_layers
            self.dropout = dropout
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
                x = F.dropout(x, self.dropout, self.training) if i < self.num_layers - 1 and self.dropout > 0 else x
            return x


    def bias_init_with_prob(prior_prob):
        """initialize conv/fc bias value according to giving probablity."""
        return -math.log((1 - prior_prob) / prior_prob)


    def build(args, model_ds):
        representations = {"axangle": 3, "quat": 4, "rot6d": 6, "mat": 9}
        num_rot_params = representations[args.rot_rep.split('_')[-1]]

        weight_dict = {'loss_ce': args.class_loss_coef}
        if args.boxes:
            weight_dict['loss_bbox'] = args.bbox_loss_coef
            weight_dict['loss_giou'] = args.giou_loss_coef
        if args.keypoints:
            weight_dict['loss_keypoint'] = args.keypoint_loss_coef
            weight_dict['loss_cr'] = args.cr_loss_coef
        if args.poses:
            weight_dict['loss_pose'] = args.pose_loss_coef
        if args.masks:
            weight_dict['loss_mask'] = args.mask_loss_coef
            weight_dict['loss_dice'] = args.dice_loss_coef
        # TODO this is a hack
        if args.aux_loss:
            aux_weight_dict = {}
            for i in range(args.dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        
        device = torch.device(args.device)

        backbone = build_backbone(args)

        transformer = build_transformer(args)

        model = DETR(
            backbone,
            transformer,
            num_classes=args.num_classes,
            num_queries=args.num_queries,
            box=args.boxes,
            keypoint=args.keypoints,
            num_keypoints=args.num_keypoints,
            pose=args.poses,
            num_rot_params=num_rot_params,
            rot_type=args.rot_type,
            trans_type=args.trans_type,
            bwl=args.bwl,
            aux_loss=args.aux_loss,
            weight_dict=weight_dict
        )
        if args.masks:
            model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
        matcher = build_matcher(args)

        losses = ['labels', 'cardinality']
        if args.boxes:
            losses += ['boxes']
        models_pts, models_kpts, sym_classes = None, None, None
        if args.keypoints:
            losses += ['keypoints']
            # models_kpts = torch.from_numpy(model_ds.keypoints.astype(np.float32))  # bpnp
        if args.poses:
            losses += ['poses']
            models_pts = model_ds.sample_points(args.num_points, args.seed)
            sym_classes = args.sym_classes
        if args.masks:
            losses += ['masks']
        criterion = SetCriterion(args.num_classes, matcher, weight_dict, args.eos_coef, losses, model.bayesian_weights, args.focal_loss, models_kpts, args.rot_rep, args.trans_type, models_pts, sym_classes)
        criterion.to(device)
        postprocessors = {}
        if args.boxes:
            postprocessors['bbox'] = PostProcess(args.focal_loss)
        if args.keypoints:
            postprocessors['keypoints'] = postprocessors['bbox'] if 'bbox' in postprocessors else PostProcess(args.focal_loss)
        if args.poses:
            postprocessors['pose'] = PostProcessPose(args.rot_rep, args.trans_type)
        if args.masks:
            postprocessors['segm'] = PostProcessSegm()
            if args.dataset == "coco_panoptic":
                is_thing_map = {i: i <= 90 for i in range(201)}
                postprocessors['panoptic'] = PostProcessPanoptic(is_thing_map, threshold=0.85)

        return model, criterion, postprocessors

