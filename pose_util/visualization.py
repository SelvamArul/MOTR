import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from .pose_ops import mat2axangle, mat2quat, axangle2mat, quat2mat
from .box_ops import box_cxcywh_to_xyxy

cmap = plt.get_cmap('rainbow')
edeg_cmap = plt.get_cmap('binary')
colors = np.flip(np.round(cmap(np.tile(np.linspace(0, 1, 22), 2))[:, :3] * 255), axis=1)
edge_colors = np.flip(np.round(edeg_cmap(np.tile(np.linspace(0, 1, 12), 2))[:, :3] * 255), axis=1)
classes = [
    '__background__', 'chef_can', 'cracker', 'sugar', 'soup_can', 'mustard',
    'tuna_can', 'pudding', 'gelatin', 'meat_can', 'banana', 'pitcher',
    'bleach', 'bowl', 'mug', 'drill', 'wood_block', 'scissors', 'marker',
    'clamp', 'xl_clamp', 'foam_brick'
]


def normalize_image(image):
    min = image.min().item()
    max = image.max().item()
    image.add_(-min).div_(max - min + 1e-5)  # add noise to avoid divide by zero


def prepare_images(images, mean=None, std=None, normalize=True):
    images = images.clone()
    if None not in (mean, std):
        images.mul_(std[None, :, None, None]).add_(mean[None, :, None, None])

    if normalize:
        for image in images:
            normalize_image(image)

    images = images.mul(255).clamp(0, 255).byte().permute(0, 2, 3, 1)
    images = np.flip(images.cpu().numpy(), axis=3)  # convert to bgr

    return images


def draw_bbox_preds(samples, targets, outputs, image_folder, threshold=0.5, output_dir="bbox_eval"):
    images = prepare_images(samples.tensors)
    for image, gts, dts in zip(images, targets, outputs):
        filename = gts["filename"]
        labels = dts["labels"]
        boxes = dts["boxes"]
        scores = dts["scores"]
        keep = scores > threshold
        labels = labels[keep]
        boxes = boxes[keep]
        image = image.copy()
        labels = labels.cpu().numpy()
        boxes = boxes.cpu().numpy().astype(np.int64)
        for j, (l, b) in enumerate(zip(labels, boxes)):
            color = colors[l - 1]
            image = cv2.line(image, tuple(b[:2]), (b[0], b[3]), color, 2, cv2.LINE_AA)
            image = cv2.line(image, tuple(b[:2]), (b[2], b[1]), color, 2, cv2.LINE_AA)
            image = cv2.line(image, (b[2], b[1]), tuple(b[2:]), color, 2, cv2.LINE_AA)
            image = cv2.line(image, (b[0], b[3]), tuple(b[2:]), color, 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(output_dir, filename), image)


def draw_poses_preds(samples, targets, outputs, models_3d, threshold=0.5, output_dir="pose_eval"):
    images = prepare_images(samples.tensors)
    color = (255, 255, 255)
    for image, gts, dts in zip(images, targets, outputs):
        filename = gts["filename"]
        intrinsic = gts["intrinsic"].numpy().astype(np.float64)
        labels = dts["labels"]
        rotations = dts["rotations"]
        translations = dts["translations"]
        keypoints = dts["keypoints"]
        scores = dts["scores"]
        keep = scores > threshold
        labels = labels[keep]
        image = image.copy()
        rotations = rotations[keep].numpy().astype(np.float64)
        translations = translations[keep].numpy().astype(np.float64)
        keypoints = keypoints[keep].numpy().astype(np.int64)
        for l, r, t, kpts in zip(labels, rotations, translations, keypoints):
            axis_points, _ = cv2.projectPoints(np.float64([[30., 0, 0], [0, 30., 0], [0, 0, 30.], [0, 0, 0]]), r, t, intrinsic, np.zeros(5))  # 30
            axis_points = np.int64(axis_points[:, 0])
            image = cv2.line(image, tuple(axis_points[3]), tuple(axis_points[0]), (255, 0, 0), 2, cv2.LINE_AA)
            image = cv2.line(image, tuple(axis_points[3]), tuple(axis_points[1]), (0, 255, 0), 2, cv2.LINE_AA)
            image = cv2.line(image, tuple(axis_points[3]), tuple(axis_points[2]), (0, 0, 255), 2, cv2.LINE_AA)
            bbox_points, _ = cv2.projectPoints(get_3d_bbox(models_3d[l - 1]), r, t, intrinsic, None)
            bbox_points = np.int64(bbox_points[:, 0])
            image = cv2.line(image, tuple(bbox_points[0]), tuple(bbox_points[1]), edge_colors[0], 2, cv2.LINE_AA)
            image = cv2.line(image, tuple(bbox_points[0]), tuple(bbox_points[2]), edge_colors[1], 2, cv2.LINE_AA)
            image = cv2.line(image, tuple(bbox_points[0]), tuple(bbox_points[4]), edge_colors[2], 2, cv2.LINE_AA)
            image = cv2.line(image, tuple(bbox_points[1]), tuple(bbox_points[3]), edge_colors[3], 2, cv2.LINE_AA)
            image = cv2.line(image, tuple(bbox_points[1]), tuple(bbox_points[5]), edge_colors[4], 2, cv2.LINE_AA)
            image = cv2.line(image, tuple(bbox_points[2]), tuple(bbox_points[3]), edge_colors[5], 2, cv2.LINE_AA)
            image = cv2.line(image, tuple(bbox_points[2]), tuple(bbox_points[6]), edge_colors[6], 2, cv2.LINE_AA)
            image = cv2.line(image, tuple(bbox_points[3]), tuple(bbox_points[7]), edge_colors[7], 2, cv2.LINE_AA)
            image = cv2.line(image, tuple(bbox_points[4]), tuple(bbox_points[5]), edge_colors[8], 2, cv2.LINE_AA)
            image = cv2.line(image, tuple(bbox_points[4]), tuple(bbox_points[6]), edge_colors[9], 2, cv2.LINE_AA)
            image = cv2.line(image, tuple(bbox_points[5]), tuple(bbox_points[7]), edge_colors[10], 2, cv2.LINE_AA)
            image = cv2.line(image, tuple(bbox_points[6]), tuple(bbox_points[7]), edge_colors[11], 2, cv2.LINE_AA)
            points, _ = cv2.projectPoints(models_3d[l - 1], r, t, intrinsic, None)
            for j, p in enumerate(np.int64(points[:, 0])):
                image = cv2.circle(image, tuple(p), 2, colors[l - 1], -1)  # 2 * j
            # for j, p in enumerate(kpts[:8]):
            #     image = cv2.circle(image, tuple(p[:2]), 5, colors[j], -1)
            # image = cv2.polylines(image, np.int64(points[:, 0])[None], True, colors[l - 1])
        cv2.imwrite(os.path.join(output_dir, filename), image)


def draw_bbox_preds2(targets, outputs, image_folder, threshold=0.5, output_dir="bbox_eval"):
    for gts, dts in zip(targets, outputs):
        filename = gts['filename']
        images = []
        for i, data in enumerate([gts, dts]):
            labels = data["labels"]
            boxes = data["boxes"]
            if i == 1:
                scores = data["scores"]
                keep = scores > threshold
                labels = labels[keep]
                boxes = boxes[keep]
                scores = scores[keep]
            else:
                boxes = box_cxcywh_to_xyxy(boxes)
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy().astype(np.int64)
            if i == 0:
                boxes *= np.array([[640, 480, 640, 480]])
            image = cv2.imread(os.path.join(image_folder, filename[:6], 'rgb', filename[7:]), cv2.IMREAD_COLOR)
            for j, (l, b) in enumerate(zip(labels, boxes)):
                color = colors[l - 1]
                image = cv2.line(image, tuple(b[:2]), (b[0], b[3]), color, 2, cv2.LINE_AA)
                image = cv2.line(image, tuple(b[:2]), (b[2], b[1]), color, 2, cv2.LINE_AA)
                image = cv2.line(image, (b[2], b[1]), tuple(b[2:]), color, 2, cv2.LINE_AA)
                image = cv2.line(image, (b[0], b[3]), tuple(b[2:]), color, 2, cv2.LINE_AA)
                text = classes[l] if i == 0 else '{}:{:.2f}'.format(classes[l], scores[j])
                cv2.putText(image, text, tuple(b[:2] - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
            images += [image, np.ones((480, 5, 3), dtype=np.uint8) * 255]
        cv2.imwrite(os.path.join(output_dir, filename), np.concatenate(images[:-1], axis=1))


def draw_poses(preds, filename, models_3d, intrinsic, dists_sys, dists_asys, output_dir="pose_eval"):
    images = []
    sets = ['gts', 'preds']
    for i, pred in enumerate(preds):
        image = cv2.imread(os.path.join('../datasets/bop_ycbv_coco/test', filename[:6], 'rgb', filename[7:]), cv2.IMREAD_COLOR)
        labels = pred["labels"]
        rotations = pred["rotations"].astype(np.float64)
        translations = pred["translations"].astype(np.float64)
        keypoints = pred["keypoints"].astype(np.int64)
        color = (255, 255, 255)
        for l, r, t, kpts, s, ns in zip(labels, rotations, translations, keypoints, dists_sys, dists_asys):
            # axis_points, _ = cv2.projectPoints(np.float64([[30., 0, 0], [0, 30., 0], [0, 0, 30.], [0, 0, 0]]), r, t, intrinsic, np.zeros(5))  # 30
            # axis_points = np.int64(axis_points[:, 0])
            # image = cv2.line(image, tuple(axis_points[3]), tuple(axis_points[0]), (255, 0, 0), 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(axis_points[3]), tuple(axis_points[1]), (0, 255, 0), 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(axis_points[3]), tuple(axis_points[2]), (0, 0, 255), 2, cv2.LINE_AA)
            # bbox_points, _ = cv2.projectPoints(get_3d_bbox(models_3d[i][l - 1]), r, t, intrinsic, None)
            # bbox_points = np.int64(bbox_points[:, 0])
            # color = (0, 0, 255) if i == 0 and ns > 0.1 else (255, 255, 255)
            # image = cv2.line(image, tuple(bbox_points[0]), tuple(bbox_points[1]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[0]), tuple(bbox_points[2]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[0]), tuple(bbox_points[4]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[1]), tuple(bbox_points[3]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[1]), tuple(bbox_points[5]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[2]), tuple(bbox_points[3]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[2]), tuple(bbox_points[6]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[3]), tuple(bbox_points[7]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[4]), tuple(bbox_points[5]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[4]), tuple(bbox_points[6]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[5]), tuple(bbox_points[7]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[6]), tuple(bbox_points[7]), color, 2, cv2.LINE_AA)
            # points, _ = cv2.projectPoints(models_3d[i][l - 1], r, t, intrinsic, None)
            # for j, p in enumerate(np.int64(points[:, 0])):
            #     image = cv2.circle(image, tuple(p), 2, colors[l - 1], -1)  # 2 * j
            for j, p in enumerate(kpts):
                image = cv2.circle(image, tuple(p[:2]), 5, colors[l - 1], -1)
            # image = cv2.polylines(image, np.int64(points[:, 0])[None], True, colors[l - 1])
            # cv2.putText(image, '{}'.format(('GT', 'PoseCNN', 'DETR')[i]), (0, 475), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 1, cv2.LINE_AA)
            # if i == 0:
            #     cv2.putText(image, '{:.2f}/{:.2f}'.format(s, ns), tuple(bbox_points.min(0)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
        # images += [image, np.ones((10, 640, 3), dtype=np.uint8) * 255]
        cv2.imwrite(os.path.join(output_dir, sets[i], filename), image)
    # cv2.imwrite(os.path.join(output_dir, filename), np.concatenate(images[2:-1], axis=0))


def draw_poses_countor(preds, filename, models_3d, intrinsic, dists_sys, dists_asys, output_dir="pose_eval"):
    images = []
    image = cv2.imread(os.path.join('../datasets/bop_ycbv_coco/test', filename[:6], 'rgb', filename[7:]), cv2.IMREAD_COLOR)
    for i, pred in enumerate(preds):
        # image = cv2.imread(os.path.join('../datasets/bop_ycbv_coco/test', filename[:6], 'rgb', filename[7:]), cv2.IMREAD_COLOR)
        labels = pred["labels"]
        rotations = pred["rotations"].astype(np.float64)
        translations = pred["translations"].astype(np.float64)
        keypoints = pred["keypoints"].astype(np.int64)
        color = (255, 255, 255)
        for l, r, t, kpts, s, ns in zip(labels, rotations, translations, keypoints, dists_sys, dists_asys):
            # axis_points, _ = cv2.projectPoints(np.float64([[30., 0, 0], [0, 30., 0], [0, 0, 30.], [0, 0, 0]]), r, t, intrinsic, np.zeros(5))  # 30
            # axis_points = np.int64(axis_points[:, 0])
            # image = cv2.line(image, tuple(axis_points[3]), tuple(axis_points[0]), (255, 0, 0), 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(axis_points[3]), tuple(axis_points[1]), (0, 255, 0), 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(axis_points[3]), tuple(axis_points[2]), (0, 0, 255), 2, cv2.LINE_AA)
            # bbox_points, _ = cv2.projectPoints(get_3d_bbox(models_3d[i][l - 1]), r, t, intrinsic, None)
            # bbox_points = np.int64(bbox_points[:, 0])
            # color = (0, 0, 255) if i == 0 and ns > 0.1 else (255, 255, 255)
            # image = cv2.line(image, tuple(bbox_points[0]), tuple(bbox_points[1]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[0]), tuple(bbox_points[2]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[0]), tuple(bbox_points[4]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[1]), tuple(bbox_points[3]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[1]), tuple(bbox_points[5]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[2]), tuple(bbox_points[3]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[2]), tuple(bbox_points[6]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[3]), tuple(bbox_points[7]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[4]), tuple(bbox_points[5]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[4]), tuple(bbox_points[6]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[5]), tuple(bbox_points[7]), color, 2, cv2.LINE_AA)
            # image = cv2.line(image, tuple(bbox_points[6]), tuple(bbox_points[7]), color, 2, cv2.LINE_AA)
            points, _ = cv2.projectPoints(models_3d[i][l - 1], r, t, intrinsic, None)
            cont = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            for j, p in enumerate(np.int64(points[:, 0])):
                cont = cv2.circle(cont, tuple(p), 2, (255, 255, 255), -1)
                # image = cv2.circle(image, tuple(p), 5, colors[2 * j], -1)  # 2 * j, colors[l - 1]
            # for j, p in enumerate(kpts):
            #     image = cv2.circle(image, tuple(p[:2]), 5, colors[l - 1], -1)
            contours, hier = cv2.findContours(cont, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            stencil = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            for cnt in contours:
                cv2.drawContours(stencil,[cnt], 0, 255, -1)
                # cv2.fillPoly(stencil, cnt, (0, 0, 255))
            edges = cv2.Canny(stencil, 100, 200)
            edges = cv2.dilate(edges,np.ones((3, 3),np.uint8), iterations=1)
            image[edges != 0] = (0, 255, 0) if i == 0 else (255, 0, 0)
            # # contours, hier = cv2.findContours(stencil, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # # for cnt in contours:
            # #     image = cv2.polylines(image, cnt, True, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            #     # cv2.fillPoly(image, cnt, (0, 255, 0), cv2.MARKER_CROSS)
            # # image = cv2.polylines(image, np.int64(points[:, 0])[None], True, colors[l - 1])
            # cv2.putText(image, '{}'.format(('GT', 'PoseCNN', 'DETR')[i]), (0, 475), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 1, cv2.LINE_AA)
            # if i == 0:
            #     cv2.putText(image, '{:.2f}/{:.2f}'.format(s, ns), tuple(bbox_points.min(0)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
        # images += [image, np.ones((10, 640, 3), dtype=np.uint8) * 255]
    cv2.imwrite(os.path.join(output_dir, filename), image)  # np.concatenate(images[-2:-1], axis=0)


def draw_target(image, target, models_3d):
    filename, labels = target["image_id"].item(), target["labels"].numpy()
    rotations = target["rotations"].numpy().astype(np.float64).reshape(-1, 3, 3)
    translations = target["translations"].numpy().astype(np.float64)
    intrinsic = target["intrinsic"].numpy().astype(np.float64)
    image = image.numpy().transpose(1, 2, 0)
    # image = np.asarray(target["orig_image"])
    image = np.flip(image, axis=2).copy()
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    for l, r, t in zip(labels, rotations, translations):
        points, _ = cv2.projectPoints(models_3d[l - 1], r, t, intrinsic, np.zeros(5))
        for p in np.int64(points[:, 0]):
            image = cv2.circle(image, tuple(p), 2, colors[l - 1], -1)
        # image = cv2.polylines(image, np.int64(points[:, 0])[None], True, colors[l - 1])
        axis_points, _ = cv2.projectPoints(np.float64([[.03, 0, 0], [0, .03, 0], [0, 0, .03], [0, 0, 0]]), r, t, intrinsic, np.zeros(5))
        axis_points = np.int64(axis_points[:, 0])
        image = cv2.line(image, tuple(axis_points[3]), tuple(axis_points[0]), (255, 0, 0), 2, cv2.LINE_AA)
        image = cv2.line(image, tuple(axis_points[3]), tuple(axis_points[1]), (0, 255, 0), 2, cv2.LINE_AA)
        image = cv2.line(image, tuple(axis_points[3]), tuple(axis_points[2]), (0, 0, 255), 2, cv2.LINE_AA)
        bbox_points, _ = cv2.projectPoints(get_3d_bbox(models_3d[l - 1]), r, t, intrinsic, np.zeros(5))
        bbox_points = np.int64(bbox_points[:, 0])
        image = cv2.line(image, tuple(bbox_points[0]), tuple(bbox_points[1]), (255, 255, 255), 2, cv2.LINE_AA)
        image = cv2.line(image, tuple(bbox_points[0]), tuple(bbox_points[2]), (255, 255, 255), 2, cv2.LINE_AA)
        image = cv2.line(image, tuple(bbox_points[0]), tuple(bbox_points[4]), (255, 255, 255), 2, cv2.LINE_AA)
        image = cv2.line(image, tuple(bbox_points[1]), tuple(bbox_points[3]), (255, 255, 255), 2, cv2.LINE_AA)
        image = cv2.line(image, tuple(bbox_points[1]), tuple(bbox_points[5]), (255, 255, 255), 2, cv2.LINE_AA)
        image = cv2.line(image, tuple(bbox_points[2]), tuple(bbox_points[3]), (255, 255, 255), 2, cv2.LINE_AA)
        image = cv2.line(image, tuple(bbox_points[2]), tuple(bbox_points[6]), (255, 255, 255), 2, cv2.LINE_AA)
        image = cv2.line(image, tuple(bbox_points[3]), tuple(bbox_points[7]), (255, 255, 255), 2, cv2.LINE_AA)
        image = cv2.line(image, tuple(bbox_points[4]), tuple(bbox_points[5]), (255, 255, 255), 2, cv2.LINE_AA)
        image = cv2.line(image, tuple(bbox_points[4]), tuple(bbox_points[6]), (255, 255, 255), 2, cv2.LINE_AA)
        image = cv2.line(image, tuple(bbox_points[5]), tuple(bbox_points[7]), (255, 255, 255), 2, cv2.LINE_AA)
        image = cv2.line(image, tuple(bbox_points[6]), tuple(bbox_points[7]), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(f'{filename}.jpg', image)


def get_3d_bbox(points):
    extent = 2 * np.max(np.absolute(points), axis=0)
    bbox = np.zeros((8, 3), dtype=np.float32)

    min_x, min_y, min_z = np.min(points, axis=0)
    max_x, max_y, max_z = np.max(points, axis=0)

    bbox[0] = [max_x, max_y, max_z]
    bbox[1] = [min_x, max_y, max_z]
    bbox[2] = [max_x, min_y, max_z]
    bbox[3] = [min_x, min_y, max_z]
    bbox[4] = [max_x, max_y, min_z]
    bbox[5] = [min_x, max_y, min_z]
    bbox[6] = [max_x, min_y, min_z]
    bbox[7] = [min_x, min_y, min_z]

    # x_half, y_half, z_half = extent * 0.5

    # bbox[0] = [x_half, y_half, z_half]
    # bbox[1] = [-x_half, y_half, z_half]
    # bbox[2] = [x_half, -y_half, z_half]
    # bbox[3] = [-x_half, -y_half, z_half]
    # bbox[4] = [x_half, y_half, -z_half]
    # bbox[5] = [-x_half, y_half, -z_half]
    # bbox[6] = [x_half, -y_half, -z_half]
    # bbox[7] = [-x_half, -y_half, -z_half]
    return bbox
