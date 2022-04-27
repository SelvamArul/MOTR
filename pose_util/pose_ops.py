import cv2
import torch
import torch.nn.functional as F
import kornia
import numpy as np
import pinocchio as pin
import eigenpy
eigenpy.switchToNumpyArray()


def transform_poses(poses, intrinsic, transform, scale, inv=False):
    poses = poses.clone()
    new_poses = poses
    if not inv:
        poses[:, 2, 3] /= scale  # translations[:, 2]
        intrinsic_inv = torch.linalg.inv(intrinsic)
        for i in range(poses.size(0)):
            new_poses[i] = torch.matmul(intrinsic_inv, torch.matmul(transform, poses[i]))
        # new_poses[:, 0, 3] = (centers[:, 0] - intrinsic[0, 2]) * new_poses[:, 2, 3] / intrinsic[0, 0]
        # new_poses[:, 1, 3] = (centers[:, 1] - intrinsic[1, 2]) * new_poses[:, 2, 3] / intrinsic[1, 1]
    else:
        transform_inv = torch.linalg.inv(transform)
        for i in range(poses.size(0)):
            new_poses[i] = torch.matmul(transform_inv, torch.matmul(intrinsic, poses[i]))
        new_poses[:, 2, 3] *= scale  # translations[:, 2]
    return new_poses


def affine_transform(points, transform):
    points_reshaped = points.view(-1, 2)
    points_transformed = torch.matmul(torch.cat(
        [points_reshaped, torch.ones((points_reshaped.size(0), 1), dtype=points_reshaped.dtype, device=points_reshaped.device)], dim=1), transform.t())
    return points_transformed.view(points.shape)


def mat2axangle(matrices):
    return kornia.rotation_matrix_to_angle_axis(matrices)


def axangle2mat(vectors):
    return kornia.angle_axis_to_rotation_matrix(vectors)


def mat2quat(matrices):
    return kornia.rotation_matrix_to_quaternion(matrices)


def quat2mat(quaternions):
    return kornia.quaternion_to_rotation_matrix(quaternions)


def mat2rot6d(matrices):
    return torch.cat([matrices[..., 0], matrices[..., 1]], dim=1)


def rot6d2mat(ortho6ds):
    x_raw = ortho6ds[..., :3]
    y_raw = ortho6ds[..., 3:]
    x = F.normalize(x_raw, p=2, dim=-1, eps=1e-8)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, p=2, dim=-1, eps=1e-8)
    y = torch.cross(z, x, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def axangle2mat(axes, angles, is_normalized=False):
    # http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    if not is_normalized:
        axes = F.normalize(axes, p=2, dim=1, eps=1e-8)
    x, y, z = axes.unbind(1)
    c = torch.cos(angles)
    s = torch.sin(angles)
    xs, ys, zs = x * s, y * s, z * s
    xc, yc, zc = x * (1 - c), y * (1 - c), z * (1 - c)
    xyc, yzc, zxc = x * yc, y * zc, z * xc
    rot_mat = torch.stack(
        [x * xc + c, xyc - zs, zxc + ys, xyc + zs, y * yc + c, yzc - xs, zxc - ys, yzc + xs, z * zc + c], dim=1)
    return rot_mat.view(-1, 3, 3)


def ego2allo(ego_poses, src_type="mat", dst_type="mat", cam_ray=[0, 0, 1.0]):
    """Given an allocentric (object-centric) pose, compute new camera-centric
    pose Since we do detection on the image plane and our kernels are
    2D-translationally invariant, we need to ensure that rendered objects
    always look identical, independent of where we render them.

    Since objects further away from the optical center undergo skewing,
    we try to visually correct by rotating back the amount between
    optical center ray and object centroid ray. Another way to solve
    that might be translational variance
    (https://arxiv.org/abs/1807.03247)
    """
    # compute rotation between ray to object centroid and optical center ray
    if src_type == "mat":
        trans = ego_poses[:, :3, 3]
    else:
        raise ValueError("src_type should be mat or quat, got: {}".format(src_type))
    obj_ray = F.normalize(trans, p=2, dim=1, eps=1e-8)
    cam_ray = torch.as_tensor(cam_ray, dtype=obj_ray.dtype, device=obj_ray.device)
    angle = torch.acos(torch.sum(cam_ray * obj_ray, dim=1))  # dot product

    # # rotate back by that amount
    mask = angle > 0
    allo_poses = ego_poses.clone()
    if dst_type == "mat":
        rot_mat = axangle2mat(torch.cross(cam_ray.expand(mask.sum(), -1), obj_ray[mask], dim=1), -angle[mask])
        if src_type == "mat":
            allo_poses[mask, :3, :3] = torch.matmul(rot_mat, ego_poses[mask, :3, :3])
    else:
        raise ValueError("dst_type should be mat or quat, got: {}".format(dst_type))
    return allo_poses


def allo2ego(allo_poses, src_type="mat", dst_type="mat", cam_ray=[0, 0, 1.0]):
    """Given an allocentric (object-centric) pose, compute new camera-centric
    pose Since we do detection on the image plane and our kernels are
    2D-translationally invariant, we need to ensure that rendered objects
    always look identical, independent of where we render them.

    Since objects further away from the optical center undergo skewing,
    we try to visually correct by rotating back the amount between
    optical center ray and object centroid ray. Another way to solve
    that might be translational variance
    (https://arxiv.org/abs/1807.03247)
    """
    # compute rotation between ray to object centroid and optical center ray
    if src_type == "mat":
        trans = allo_poses[:, :3, 3]
    else:
        raise ValueError("src_type should be mat or quat, got: {}".format(src_type))
    obj_ray = F.normalize(trans, p=2, dim=1, eps=1e-8)
    cam_ray = torch.as_tensor(cam_ray, dtype=obj_ray.dtype, device=obj_ray.device)
    angle = torch.acos(torch.sum(cam_ray * obj_ray, dim=1))  # dot product

    # # rotate back by that amount
    mask = angle > 0
    ego_poses = allo_poses.clone()
    if dst_type == "mat":
        rot_mat = axangle2mat(torch.cross(cam_ray.expand(mask.sum(), -1), obj_ray[mask], dim=1), angle[mask])
        if src_type == "mat":
            ego_poses[mask, :3, :3] = torch.matmul(rot_mat, allo_poses[mask, :3, :3])
    else:
        raise ValueError("dst_type should be mat or quat, got: {}".format(dst_type))
    return ego_poses


def points_to_2d(points, R, t, K):
    # using opencv
    points_2d, _ = cv2.projectPoints(points, R, t, K, None)
    points_2d = points_2d[:, 0]
    points_in_world = np.matmul(R, points.T) + t[:, None]  # (3, N)
    # points_in_camera = np.matmul(K, points_in_world)  # (3, N), z is not changed in this step
    # points_2d = np.zeros((2, points_in_world.shape[1]))  # (2, N)
    # points_2d[0, :] = points_in_camera[0, :] / (points_in_camera[2, :] + 1e-15)
    # points_2d[1, :] = points_in_camera[1, :] / (points_in_camera[2, :] + 1e-15)
    # points_2d = points_2d.T
    z = points_in_world[2]
    return points_2d, z


def pnp(points_3d, points_2d, camera_matrix, method=cv2.SOLVEPNP_EPNP, ransac=False, ransac_reprojErr=3.0, ransac_iter=100):
    """
    method: cv2.SOLVEPNP_P3P, cv2.SOLVEPNP_DLS, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE
        DLS seems to be similar to EPNP
        SOLVEPNP_EPNP does not work with no ransac
    RANSAC:
        CDPN: 3.0, 100
        default ransac params:   float reprojectionError=8.0, int iterationsCount=100, double confidence=0.99
        in DPOD paper: reproj error=1.0, ransac_iter=150
    """
    dist_coeffs = np.zeros(shape=[8, 1], dtype=np.float64)

    assert points_3d.shape[0] == points_2d.shape[0], "points 3D and points 2D must have same number of vertices"
    if method == cv2.SOLVEPNP_EPNP:
        points_3d = points_3d[None]
        points_2d = points_2d[None]

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d)
    # camera_matrix = camera_matrix.astype(np.float64)
    if not ransac:
        _, R_exp, t = cv2.solvePnP(points_3d, points_2d, camera_matrix, dist_coeffs, flags=method)
    else:
        _, R_exp, t, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            camera_matrix,
            dist_coeffs,
            flags=method,  # cv2.SOLVEPNP_EPNP,
            reprojectionError=ransac_reprojErr,
            iterationsCount=ransac_iter
        )
    # , None, None, False, cv2.SOLVEPNP_UPNP)

    # R_exp, t, _ = cv2.solvePnPRansac(points_3D,
    #                            points_2D,
    #                            cameraMatrix,
    #                            distCoeffs,
    #                            reprojectionError=12.0)

    R, _ = cv2.Rodrigues(R_exp)  # R = cv2.Rodrigues(R_exp, jacobian=0)[0]
    # trans_3d=np.matmul(points_3d, R.transpose()) + t.transpose()
    # if np.max(trans_3d[:,2]<0):
    #     R=-R
    #     t=-t

    return R, t


class Transform:
    def __init__(self, *args):
        if len(args) == 0:
            raise NotImplementedError
        elif len(args) == 7:
            raise NotImplementedError
        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, Transform):
                self.T = arg.T
            elif isinstance(arg, pin.SE3):
                self.T = arg
            else:
                arg_array = np.asarray(arg)
                if arg_array.shape == (4, 4):
                    R = arg_array[:3, :3]
                    t = arg_array[:3, -1]
                    self.T = pin.SE3(R, t.reshape(3, 1))
                else:
                    raise NotImplementedError
        elif len(args) == 2:
            args_0_array = np.asarray(args[0])
            n_elem_rot = len(args_0_array.flatten())
            if n_elem_rot == 4:
                xyzw = np.asarray(args[0]).flatten().tolist()
                wxyz = [xyzw[-1], *xyzw[:-1]]
                assert len(wxyz) == 4
                q = pin.Quaternion(*wxyz)
                q.normalize()
                R = q.matrix()
            elif n_elem_rot == 9:
                assert args_0_array.shape == (3, 3)
                R = args_0_array
            t = np.asarray(args[1])
            assert len(t) == 3
            self.T = pin.SE3(R, t.reshape(3, 1))
        elif len(args) == 1:
            if isinstance(args[0], Transform):
                raise NotImplementedError
            elif isinstance(args[0], list):
                array = np.array(args[0])
                assert array.shape == (4, 4)
                self.T = pin.SE3(array)
            elif isinstance(args[0], np.ndarray):
                assert args[0].shape == (4, 4)
                self.T = pin.SE3(args[0])

    def __mul__(self, other):
        T = self.T * other.T
        return Transform(T)

    def inverse(self):
        return Transform(self.T.inverse())

    def __str__(self):
        return str(self.T)

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return 7

    def toHomogeneousMatrix(self):
        return self.T.homogeneous

    @property
    def translation(self):
        return self.T.translation.reshape(3)

    @property
    def quaternion(self):
        return pin.Quaternion(self.T.rotation)
