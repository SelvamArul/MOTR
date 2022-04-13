import cv2
import torch
import numpy as np
import pinocchio as pin
import eigenpy
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle, axis_angle_to_quaternion, quaternion_to_matrix
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


def mat2vec(matrices):
    matrices = matrices.numpy()
    # r = Rotation.from_matrix(matrices)
    # return torch.from_numpy(r.as_rotvec().astype(np.float32))
    vectors = []
    for i in range(matrices.shape[0]):
        vec, _ = cv2.Rodrigues(matrices[i])
        vectors.append(np.squeeze(vec))
    vectors = np.stack(vectors).astype(np.float32) if len(vectors) else np.asarray([], dtype=np.float32)
    return torch.from_numpy(vectors)


def vec2mat(vectors):
    vectors = vectors.cpu().numpy()
    # r = Rotation.from_rotvec(vectors)
    # return torch.from_numpy(r.as_matrix().astype(np.float32))
    matrices = []
    for i in range(vectors.shape[0]):
        mat, _ = cv2.Rodrigues(vectors[i])
        matrices.append(mat)
    matrices = np.stack(matrices).astype(np.float32) if len(matrices) else np.asarray([], dtype=np.float32)
    return torch.from_numpy(matrices)


def mat2quat(matrices):
    r = Rotation.from_matrix(matrices.numpy())
    return torch.from_numpy(r.as_quat().astype(np.float32))


def quat2mat(quaternions):
    r = Rotation.from_quat(quaternions.numpy())
    return torch.from_numpy(r.as_matrix().astype(np.float32))


def mat2euler(matrices):
    r = Rotation.from_matrix(matrices.numpy())
    return torch.from_numpy(r.as_euler('xyz', degrees=False).astype(np.float32))


def euler2mat(angles):
    r = Rotation.from_euler('xyz', angles.numpy(), degrees=False)
    return torch.from_numpy(r.as_matrix().astype(np.float32))


def mat2rot6d(matrices):
    return matrices[..., :6]


def rot6d2mat(ortho6d):
    x_raw = ortho6d[..., :3]
    y_raw = ortho6d[..., 3:]
    x = x_raw / torch.norm(x_raw, p=2, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, p=2, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)
    return torch.stack((x, y, z), -1)


def re(outs, tgts):
    diff = torch.bmm(outs, tgts.transpose(-2, -1))  # torch.linalg.inv(b)
    trace = torch.diagonal(diff, dim1=-2, dim2=-1).sum(-1)  # torch.einsum('bii->b', diff)
    # numerical stability
    trace[trace > 3] = 3
    error_cos = torch.clamp(0.5 * (trace - 1.0), -1.0, 1.0)

    error = torch.rad2deg(torch.acos(error_cos))
    return error


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
