import torch
import torch.nn.functional as F


def matrix_to_rot6d(matrix):
    first_two = matrix[..., :, :2].transpose(-2, -1)
    return first_two.reshape(*matrix.shape[:-2], 6)


def rot6d_to_matrix(rot6d):
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)
