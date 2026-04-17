import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperDecHead(nn.Module):
    """Superquadric prediction head with explicit existence logits."""

    def __init__(self, emb_dims, ctx):
        super().__init__()
        self.emb_dims = emb_dims
        self.scale_head = nn.Linear(emb_dims, 3)
        self.shape_head = nn.Linear(emb_dims, 2)
        self.rotation6d = getattr(ctx, "rotation6d", False)
        self.rot_head = nn.Linear(emb_dims, 6 if self.rotation6d else 4)
        self.t_head = nn.Linear(emb_dims, 3)
        self.exist_head = nn.Linear(emb_dims, 1)

        if getattr(ctx, "clear_orientation_heads", False):
            nn.init.constant_(self.t_head.bias, 0.5)
            nn.init.normal_(self.t_head.weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.shape_head.bias, 1.0)
            nn.init.normal_(self.shape_head.weight, mean=0.0, std=1e-3)
            nn.init.orthogonal_(self.rot_head.weight)
            nn.init.zeros_(self.rot_head.bias)

        self.extended = getattr(ctx, "extended", False)
        if self.extended:
            extended_non_zero_init = getattr(ctx, "extended_non_zero_init", False)
            self.tapering_head = nn.Linear(emb_dims, 2)
            self.bending_k_head = nn.Linear(emb_dims, 3)
            self.bending_a_head = nn.Linear(emb_dims, 3)

            if extended_non_zero_init:
                nn.init.normal_(self.tapering_head.weight, mean=0.0, std=1e-4)
                nn.init.normal_(self.tapering_head.bias, mean=0.0, std=1e-4)
                nn.init.normal_(self.bending_a_head.weight, mean=0.0, std=1e-3)
                nn.init.normal_(self.bending_a_head.bias, mean=0.0, std=1e-3)
                nn.init.normal_(self.bending_k_head.weight, mean=0.0, std=1e-3)
                nn.init.constant_(self.bending_k_head.bias, -5.0)
            else:
                nn.init.zeros_(self.tapering_head.weight)
                nn.init.zeros_(self.tapering_head.bias)
                nn.init.zeros_(self.bending_a_head.weight)
                nn.init.zeros_(self.bending_a_head.bias)
                nn.init.zeros_(self.bending_k_head.weight)
                nn.init.constant_(self.bending_k_head.bias, -6.0)

    def forward(self, x):
        scale = self.scale_activation(self.scale_head(x))
        shape = self.shape_activation(self.shape_head(x))

        rot_raw = self.rot_head(x)
        if self.rotation6d:
            rotation = self.rot6d2mat(rot_raw)
        else:
            rotation_quat = F.normalize(rot_raw, dim=-1, p=2)
            rotation = self.quat2mat(rotation_quat)

        translation = self.t_head(x)
        exist_logit = self.exist_head(x)
        exist = self.exist_activation(exist_logit)

        out_dict = {
            "scale": scale,
            "shape": shape,
            "rotate": rotation,
            "trans": translation,
            "exist_logit": exist_logit,
            "exist": exist,
        }
        if self.rotation6d:
            out_dict["rotation_6d"] = rot_raw
        else:
            out_dict["rotation_quat"] = rotation_quat

        if self.extended:
            tapering = self.tapering_activation(self.tapering_head(x))
            bending_k = self.bending_k_activation(self.bending_k_head(x), scale)
            bending_a = self.bending_a_activation(self.bending_a_head(x))
            out_dict.update(
                {"tapering": tapering, "bending_k": bending_k, "bending_a": bending_a}
            )
        return out_dict

    @staticmethod
    def quat2mat(quat):
        batch = quat.shape[0]
        n_queries = quat.shape[1]
        quat = F.normalize(quat, dim=2)
        quat = quat.contiguous().view(-1, 4)
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z
        rot_mat = torch.stack(
            [
                w2 + x2 - y2 - z2,
                2 * xy - 2 * wz,
                2 * wy + 2 * xz,
                2 * wz + 2 * xy,
                w2 - x2 + y2 - z2,
                2 * yz - 2 * wx,
                2 * xz - 2 * wy,
                2 * wx + 2 * yz,
                w2 - x2 - y2 + z2,
            ],
            dim=1,
        ).view(batch, n_queries, 3, 3)
        return rot_mat

    @staticmethod
    def rot6d2mat(rot6d):
        batch, n_queries, _ = rot6d.shape
        rot6d = rot6d.view(-1, 6)
        a1 = rot6d[:, :3]
        a2 = rot6d[:, 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = F.normalize(a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        rot_mat = torch.stack((b1, b2, b3), dim=-1)
        return rot_mat.view(batch, n_queries, 3, 3)

    @staticmethod
    def scale_activation(x):
        return torch.sigmoid(x)

    @staticmethod
    def shape_activation(x):
        return 0.1 + 1.8 * torch.sigmoid(x)

    @staticmethod
    def exist_activation(x):
        return torch.sigmoid(x)

    @staticmethod
    def tapering_activation(x):
        return torch.tanh(x)

    @staticmethod
    def bending_k_activation(x, scale):
        tau = 0.01
        max_scale = tau * torch.logsumexp(scale / tau, keepdim=True, dim=-1)
        return (torch.sigmoid(x) * 0.95) / max_scale

    @staticmethod
    def bending_a_activation(x):
        return torch.sigmoid(x) * 2 * torch.pi
