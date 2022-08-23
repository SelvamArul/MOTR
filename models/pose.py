import cv2
import numpy as np
import kornia
import torch
from torch import nn
import torch.nn.functional as F

from util import box_ops
from util.pose_ops import axangle2mat, quat2mat, rot6d2mat, allo2ego, transform_poses, affine_transform


def cross_ratio(preds_coor, threshold):
    edges = [[0, 8, 20, 1], [2, 9, 21, 3], [4, 10, 22, 5], [6, 11, 23, 7], [0, 12, 24, 4], [1, 13, 25, 5], [2, 14, 26, 6], [3, 15, 27, 7], [0, 16, 28, 2], [1, 17, 29, 3], [4, 18, 30, 6], [5, 19, 31, 7]]
    lines = preds_coor[:, edges]
    lines_detached = lines.detach()
    dists = torch.pow(lines_detached[:, :, :, None] - lines_detached[:, :, None], 2).sum(-1).sqrt()
    dists[dists == 0] = 2  # greater than max possible value which is sqrt(2)
    minval, _ = torch.min(dists.flatten(-2, -1), dim=2)
    mask = torch.gt(minval, threshold).float()
    end, start = [2, 3, 2, 3], [0, 1, 1, 0]
    diff = lines[:, :, end] - lines[:, :, start]
    dots = torch.sum(diff * diff, dim=3)
    cr_sqr = torch.mul(dots[:, :, 0], dots[:, :, 1]) / torch.mul(dots[:, :, 2], dots[:, :, 3])
    return cr_sqr, mask


def trans_from_centroid(centroid, z, intrinsic):
    return torch.stack(
        [z * (centroid[:, 0] - intrinsic[:, 0, 2]) / intrinsic[:, 0, 0], z * (centroid[:, 1] - intrinsic[:, 1, 2]) / intrinsic[:, 1, 1], z],
        dim=1,
    )


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


def axangle_rotate(points, rotations):
    axes, angles = separate_axis_from_angle(rotations)
    axes = axes.unsqueeze(1)
    cos_angles = torch.cos(angles).unsqueeze(1)
    sin_angles = torch.sin(angles).unsqueeze(1)
    return points * cos_angles + cross(axes, points) * sin_angles + axes * dot(axes, points) * (1.0 - cos_angles)


def rotate(points, rotations):
    return torch.squeeze(rotations.unsqueeze(1) @ points.unsqueeze(-1), dim=-1)


def calc_dist(src_transformations, target_transformations, asyms, p=2):
    dists = []
    if torch.bitwise_not(asyms).sum():
        # cosypose
        # dists_squared = (target_transformations[~asyms].unsqueeze(1) - src_transformations[~asyms].unsqueeze(2)) ** 2
        # dists_norm_squared = dists_squared.sum(dim=-1)
        # assign = dists_norm_squared.argmin(dim=1)
        # ids_row = torch.arange(dists_squared.shape[0]).unsqueeze(1).repeat(1, dists_squared.shape[1])
        # ids_col = torch.arange(dists_squared.shape[1]).unsqueeze(0).repeat(dists_squared.shape[0], 1)
        # sym_dists = dists_squared[ids_row, assign, ids_col].mean(2)

        # sym_dists, _ = torch.norm(src_transformations[~asyms].unsqueeze(2) - target_transformations[~asyms].unsqueeze(1), p=p, dim=3).min(2)
        batch_sym_dists = []
        a, b = src_transformations[~asyms], target_transformations[~asyms]
        for i in range(a.size(0)):  # loop through batch for efficient memory usage
            batch_sym_dists.append(torch.norm(a[i].unsqueeze(1) - b[i].unsqueeze(0), p=p, dim=2).min(1)[0])
            # batch_sym_dists.append(torch.pow(a[i].unsqueeze(1) - b[i].unsqueeze(0), 2).sum(-1).min(0)[0])
        sym_dists = torch.stack(batch_sym_dists, dim=0)
        # sym_dists, _ = torch.pow(src_transformations[~asyms].unsqueeze(2) - target_transformations[~asyms].unsqueeze(1), 2).sum(-1).min(1)
        # sym_dists = sym_dists / 3
        dists.append(sym_dists)  # 2 *
    if torch.sum(asyms):
        # asym_dists = torch.abs(src_transformations[asyms] - target_transformations[asyms]).mean(2)
        asym_dists = torch.norm(src_transformations[asyms] - target_transformations[asyms], p=p, dim=2)
        dists.append(asym_dists)
    return torch.cat(dists, dim=0)


def angular_dist(src_rotations, target_rotations):
    mats = torch.bmm(src_rotations, target_rotations.transpose(-2, -1))  # torch.linalg.inv(target_rotations)
    traces = torch.diagonal(mats, dim1=-2, dim2=-1).sum(-1)  # torch.einsum('bii->b', mats)
    cosines = torch.clamp(0.5 * (traces - 1.0), -1.0, 1.0)  # avoid invalid values due to numerical errors
    dists = 0.5 * (1.0 - cosines)  # [0, 1]
    # thetas = torch.rad2deg(torch.acos(cosines))
    return dists


class BPnP(torch.autograd.Function):
    """
    Back-propagatable PnP
    INPUTS:
    pts2d - the 2D keypoints coordinates of size [batch_size, num_keypoints, 2]
    pts3d - the 3D keypoints coordinates of size [num_keypoints, 3]
    K     - the camera intrinsic matrix of size [3, 3]
    OUTPUT:
    P_6d  - the 6 DOF poses of size [batch_size, 6], where the first 3 elements of each row are the angle-axis rotation 
    vector (Euler vector) and the last 3 elements are the translation vector. 
    NOTE:
    This BPnP function assumes that all sets of 2D points in the mini-batch correspond to one common set of 3D points. 
    For situations where pts3d is also a mini-batch, use the BPnP_m3d class.
    """
    @staticmethod
    def forward(ctx, pts2d, pts3d, K, ini_pose=None):
        bs = pts2d.size(0)
        n = pts2d.size(1)
        device = pts2d.device
        pts3d_np = np.array(pts3d.detach().cpu())
        K_np = np.array(K.detach().cpu())
        P_6d = torch.zeros(bs, 6, device=device)

        for i in range(bs):
            pts2d_i_np = np.ascontiguousarray(pts2d[i].detach().cpu()).reshape((n, 1, 2))
            if ini_pose is None:
                _, rvec0, T0, _ = cv2.solvePnPRansac(objectPoints=pts3d_np, imagePoints=pts2d_i_np, cameraMatrix=K_np, distCoeffs=None, flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.9999 ,reprojectionError=3)
            else:
                rvec0 = np.array(ini_pose[i, :3].cpu().view(3, 1))
                T0 = np.array(ini_pose[i, 3:6].cpu().view(3, 1))
            _, rvec, T = cv2.solvePnP(objectPoints=pts3d_np, imagePoints=pts2d_i_np, cameraMatrix=K_np, distCoeffs=None, flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=rvec0, tvec=T0)
            angle_axis = torch.tensor(rvec, device=device, dtype=torch.float32).view(1, 3)
            T = torch.tensor(T, device=device, dtype=torch.float32).view(1, 3)
            P_6d[i] = torch.cat((angle_axis, T), dim=-1)

        ctx.save_for_backward(pts2d, P_6d, pts3d,K)
        return P_6d

    @staticmethod
    def backward(ctx, grad_output):
        pts2d, P_6d, pts3d, K = ctx.saved_tensors
        # pts2d = pts2d.contiguous()
        # P_6d = P_6d.contiguous()
        # pts3d = pts3d.contiguous()
        # K = K.contiguous()
        device = pts2d.device
        bs = pts2d.size(0)
        n = pts2d.size(1)
        m = 6

        grad_x = torch.zeros_like(pts2d)
        grad_z = torch.zeros_like(pts3d)
        grad_K = torch.zeros_like(K)

        for i in range(bs):
            J_fy = torch.zeros(m, m, device=device)
            J_fx = torch.zeros(m, 2 * n, device=device)
            J_fz = torch.zeros(m, 3 * n, device=device)
            J_fK = torch.zeros(m, 9, device=device)

            torch.set_grad_enabled(True)
            pts2d_flat = pts2d[i].clone().view(-1).detach().requires_grad_()
            P_6d_flat = P_6d[i].clone().view(-1).detach().requires_grad_()
            pts3d_flat = pts3d.clone().view(-1).detach().requires_grad_()
            K_flat = K.clone().view(-1).detach().requires_grad_()

            for j in range(m):
                torch.set_grad_enabled(True)
                if j > 0:
                    pts2d_flat.grad.zero_()
                    P_6d_flat.grad.zero_()
                    pts3d_flat.grad.zero_()
                    K_flat.grad.zero_()

                R = kornia.angle_axis_to_rotation_matrix(P_6d_flat[:m - 3].view(1, 3))

                P = torch.cat((R[0, :3, :3].view(3, 3), P_6d_flat[m - 3:m].view(3, 1)), dim=-1)
                KP = torch.mm(K_flat.view(3, 3), P)
                pts2d_i = pts2d_flat.view(n, 2).transpose(0, 1)
                pts3d_i = torch.cat((pts3d_flat.view(n, 3),torch.ones(n, 1, device=device)), dim=-1).t()
                proj_i = KP.mm(pts3d_i)
                Si = proj_i[2].view(1, n)

                r = pts2d_i * Si-proj_i[:2]
                coefs = get_coefs(P_6d_flat.view(1, 6), pts3d_flat.view(n, 3), K_flat.view(3, 3))
                coef = coefs[:, :, j].transpose(0, 1)  # 2, n
                fj = torch.sum(coef * r)
                fj.backward()
                J_fy[j] = P_6d_flat.grad.clone()
                J_fx[j] = pts2d_flat.grad.clone()
                J_fz[j] = pts3d_flat.grad.clone()
                J_fK[j] = K_flat.grad.clone()

            inv_J_fy = torch.inverse(J_fy)

            J_yx = -1 * torch.mm(inv_J_fy, J_fx)
            J_yz = -1 * torch.mm(inv_J_fy, J_fz)
            J_yK = -1 * torch.mm(inv_J_fy, J_fK)

            grad_x[i] = grad_output[i].view(1, m).mm(J_yx).view(n, 2)
            grad_z += grad_output[i].view(1, m).mm(J_yz).view(n, 3)
            grad_K += grad_output[i].view(1, m).mm(J_yK).view(3, 3)

        return grad_x, grad_z, grad_K, None


class BPnP_fast(torch.autograd.Function):
    """
    BPnP_fast is the efficient version of the BPnP class which ignores the higher order dirivatives through the coefs' graph. This sacrifices
    negligible gradient accuracy yet saves significant runtime. 
    INPUTS:
    pts2d - the 2D keypoints coordinates of size [batch_size, num_keypoints, 2]
    pts3d - the 3D keypoints coordinates of size [num_keypoints, 3]
    K     - the camera intrinsic matrix of size [3, 3]
    OUTPUT:
    P_6d  - the 6 DOF poses of size [batch_size, 6], where the first 3 elements of each row are the angle-axis rotation 
    vector (Euler vector) and the last 3 elements are the translation vector. 
    NOTE:
    This BPnP function assumes that all sets of 2D points in the mini-batch correspond to one common set of 3D points. 
    For situations where pts3d is also a mini-batch, use the BPnP_m3d class.
    """
    @staticmethod
    def forward(ctx, pts2d, pts3d, K, ini_pose=None):
        bs = pts2d.size(0)
        n = pts2d.size(1)
        device = pts2d.device
        pts3d_np = np.array(pts3d.detach().cpu())
        K_np = np.array(K.detach().cpu())
        P_6d = torch.zeros(bs, 6, device=device)

        for i in range(bs):
            pts2d_i_np = np.ascontiguousarray(pts2d[i].detach().cpu()).reshape((n, 1, 2))
            if ini_pose is None:
                _, rvec0, T0, _ = cv2.solvePnPRansac(objectPoints=pts3d_np, imagePoints=pts2d_i_np, cameraMatrix=K_np, distCoeffs=None, flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.9999 ,reprojectionError=3)
            else:
                rvec0 = np.array(ini_pose[i, :3].cpu().view(3, 1))
                T0 = np.array(ini_pose[i, 3:6].cpu().view(3, 1))
            _, rvec, T = cv2.solvePnP(objectPoints=pts3d_np, imagePoints=pts2d_i_np, cameraMatrix=K_np, distCoeffs=None, flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=rvec0, tvec=T0)
            angle_axis = torch.tensor(rvec, device=device,dtype=torch.float32).view(1, 3)
            T = torch.tensor(T, device=device, dtype=torch.float32).view(1, 3)
            P_6d[i] = torch.cat((angle_axis, T), dim=-1)

        ctx.save_for_backward(pts2d, P_6d, pts3d, K)
        return P_6d

    @staticmethod
    def backward(ctx, grad_output):
        pts2d, P_6d, pts3d, K = ctx.saved_tensors
        device = pts2d.device
        bs = pts2d.size(0)
        n = pts2d.size(1)
        m = 6

        grad_x = torch.zeros_like(pts2d)
        grad_z = torch.zeros_like(pts3d)
        grad_K = torch.zeros_like(K)

        for i in range(bs):
            J_fy = torch.zeros(m, m, device=device)
            J_fx = torch.zeros(m, 2 * n, device=device)
            J_fz = torch.zeros(m, 3 * n, device=device)
            J_fK = torch.zeros(m, 9, device=device)

            coefs = get_coefs(P_6d[i].view(1, 6), pts3d, K, create_graph=False).detach()

            pts2d_flat = pts2d[i].clone().view(-1).detach().requires_grad_()
            P_6d_flat = P_6d[i].clone().view(-1).detach().requires_grad_()
            pts3d_flat = pts3d.clone().view(-1).detach().requires_grad_()
            K_flat = K.clone().view(-1).detach().requires_grad_()

            for j in range(m):
                torch.set_grad_enabled(True)
                if j > 0:
                    pts2d_flat.grad.zero_()
                    P_6d_flat.grad.zero_()
                    pts3d_flat.grad.zero_()
                    K_flat.grad.zero_()

                R = kornia.angle_axis_to_rotation_matrix(P_6d_flat[:m - 3].view(1, 3))

                P = torch.cat((R[0, :3, :3].view(3, 3), P_6d_flat[m - 3:m].view(3, 1)), dim=-1)
                KP = torch.mm(K_flat.view(3, 3), P)
                pts2d_i = pts2d_flat.view(n, 2).transpose(0, 1)
                pts3d_i = torch.cat((pts3d_flat.view(n, 3),torch.ones(n, 1, device=device)), dim=-1).t()
                proj_i = KP.mm(pts3d_i)
                Si = proj_i[2].view(1, n)

                r = pts2d_i * Si-proj_i[:2]
                coef = coefs[:, :, j].transpose(0, 1)  # 2, n
                fj = torch.sum(coef * r)
                fj.backward()
                J_fy[j] = P_6d_flat.grad.clone()
                J_fx[j] = pts2d_flat.grad.clone()
                J_fz[j] = pts3d_flat.grad.clone()
                J_fK[j] = K_flat.grad.clone()

            inv_J_fy = torch.inverse(J_fy)

            J_yx = -1 * torch.mm(inv_J_fy, J_fx)
            J_yz = -1 * torch.mm(inv_J_fy, J_fz)
            J_yK = -1 * torch.mm(inv_J_fy, J_fK)

            grad_x[i] = grad_output[i].view(1, m).mm(J_yx).view(n, 2)
            grad_z += grad_output[i].view(1, m).mm(J_yz).view(n, 3)
            grad_K += grad_output[i].view(1, m).mm(J_yK).view(3, 3)

        return grad_x, grad_z, grad_K, None


def get_coefs(P_6d, pts3d, K, create_graph=True):
    device = P_6d.device
    n = pts3d.size(0)
    m = P_6d.size(-1)
    coefs = torch.zeros(n, 2, m, device=device)
    torch.set_grad_enabled(True)
    y = P_6d.repeat(n, 1)
    proj = project(y, pts3d, K).squeeze()
    vec = torch.ones(n, device=device, dtype=torch.float32).diag()
    for k in range(2):
        torch.set_grad_enabled(True)
        y_grad = torch.autograd.grad(proj[:, :, k], y, vec, retain_graph=True, create_graph=create_graph)
        coefs[:, k] = -2 * y_grad[0].clone()
    return coefs


def project(P, pts3d, K, angle_axis=True):
    n = pts3d.size(0)
    bs = P.size(0)
    device = P.device
    pts3d_h = torch.cat((pts3d, torch.ones(n, 1, device=device, dtype=torch.float32)), dim=-1)
    if angle_axis:
        R_out = kornia.angle_axis_to_rotation_matrix(P[:, :3].view(bs, 3))
        PM = torch.cat((R_out[:, :3, :3], P[:, 3:6].view(bs, 3, 1)), dim=-1)
    else:
        PM = P
    pts3d_cam = pts3d_h.matmul(PM.transpose(-2, -1))
    pts2d_proj = pts3d_cam.matmul(K.t())
    S = pts2d_proj[:, :, 2].view(bs, n, 1)
    pts2d_pro = pts2d_proj[:, :, :2].div(S)  # avoid nan
    return pts2d_pro


class PostProcessPose(nn.Module):
    def __init__(self, rotation_representation, translation_type):
        super().__init__()
        self.rotation_representation = rotation_representation
        self.translation_type = translation_type

    @torch.no_grad()
    def forward(self, results, outputs, target_sizes, target_intrinsics, target_transforms=None):
        if 'pred_translations' in outputs:
            out_translations = outputs['pred_translations'].cpu()
            assert len(out_translations) == len(target_sizes)

            translations = out_translations.flatten(0, 1) * 1000  # mm
            if "cxcy" in self.translation_type:
                intrinsics = target_intrinsics.cpu()
                translations = trans_from_centroid(translations[:, :2], translations[:, 2], intrinsics.repeat(out_translations.size(1), 1, 1))

            for result, t in zip(results, translations.view(out_translations.size(0), -1, 3)):
                result['translations'] = t

        if 'pred_rotations' in outputs:
            out_rotations = outputs['pred_rotations'].cpu()
            assert len(out_rotations) == len(target_sizes)

            rotations = out_rotations.flatten(0, 1)
            if "axangle" in self.rotation_representation:
                rotations = axangle2mat(rotations)
            elif "quat" in self.rotation_representation:
                rotations = quat2mat(rotations)
            elif "rot6d" in self.rotation_representation:
                rotations = rot6d2mat(rotations)
            else:
                rotations = rotations.view(-1, 3, 3)

            if "allo" in self.rotation_representation and 'pred_translations' in outputs:
                poses = torch.cat([rotations, translations[..., None]], dim=2)
                poses = allo2ego(poses, src_type="mat", dst_type="mat")
                rotations = poses[..., :3]

            for result, r in zip(results, rotations.view(out_rotations.size(0), -1, 3, 3)):
                result['rotations'] = r

        return results
