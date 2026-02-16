import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
import math
import os

# Enable TensorFloat32 for better performance on Ampere+ GPUs
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

from typing import Optional, assert_never
from dataclasses import dataclass
import gc

from ..utils import quat2mat, mat2quat
from superdec.utils.safe_operations import safe_pow, safe_mul
from superdec.utils.predictions_handler_extended import PredictionHandler

class BatchSuperQMulti(nn.Module):
    def __init__(
        self, 
        pred_handler: PredictionHandler,
        indices: list[int],
        ply_paths: list[str] = None,
        truncation: float = 0.05,
        device: str = "cuda",
    ):
        super().__init__()
        self.indices = indices
        self.device = device
        self.truncation = truncation
        self.pred_handler = pred_handler

        B = len(indices)
        self.N_max = pred_handler.scale.shape[1]

        scale_list = []
        exp_list = []
        rot_list = []
        trans_list = []
        self.masks = [] 

        for i, idx in enumerate(indices):
            # compute bbox on surface points
            pts = pred_handler.pc[idx]
            pcd = trimesh.points.PointCloud(pts)
            obb = pcd.bounding_box_oriented
            
            obb_transform = obb.primitive.transform
            obb_extents = obb.primitive.extents

            center = torch.tensor(obb_transform[:3, 3], dtype=torch.float, device=device)
            scale = torch.tensor(obb_extents, dtype=torch.float, device=device) / 2.0
            R = torch.tensor(obb_transform[:3, :3], dtype=torch.float, device=device)

            # create parameter arrays for N_max primitives, enable only first
            s_full = torch.ones((self.N_max, 3), dtype=torch.float, device=device)
            s_full[0] = scale

            e_full = torch.full((self.N_max, 2), 0.01, dtype=torch.float, device=device)

            r_full = torch.tile(torch.eye(3, dtype=torch.float, device=device), (self.N_max, 1, 1)).reshape(self.N_max, 3, 3)
            r_full[0] = R

            t_full = torch.zeros((self.N_max, 3), dtype=torch.float, device=device)
            t_full[0] = center

            mask = torch.zeros(self.N_max, dtype=torch.bool, device=device)
            mask[0] = True

            self.masks.append(mask)
            scale_list.append(s_full)
            exp_list.append(e_full)
            rot_list.append(mat2quat(r_full))
            trans_list.append(t_full)
            
        self.raw_scale = torch.stack(scale_list) # (B, N, 3)
        self.raw_exponents = torch.stack(exp_list) # (B, N, 2)
        self.raw_rotation = torch.stack(rot_list) # (B, N, 4)
        self.translation = torch.stack(trans_list) # (B, N, 3)
        self.raw_tapering = torch.full((B, self.N_max, 2), 0, dtype=torch.float, device=device)
        
        self.exist_mask = torch.stack(self.masks) # (B, N)
        self._ = nn.Parameter(torch.zeros((B), dtype=torch.float, device=device))

    @torch.compile
    def scale(self):
        return self.raw_scale

    @torch.compile   
    def exponents(self):
        return self.raw_exponents

    @torch.compile  
    def rotation(self):
        return quat2mat(self.raw_rotation)

    @torch.compile 
    def tapering(self):
        return self.raw_tapering

    def get_param_groups(self):
        return [{"params": [self._], "lr": 0}]
    
    @torch.compile
    def sdf_batch(self, points):
        B, _, M = points.shape
        N = self.N_max
        
        points_expanded = points.unsqueeze(1) # (B, 1, 3, M)
        t = self.translation.unsqueeze(-1) # (B, N, 3, 1)
        points_centered = points_expanded - t # (B, N, 3, M)
        
        X = torch.matmul(self.rotation().transpose(-2, -1), points_centered) # (B, N, 3, M)
        
        e1 = self.exponents()[..., 0].unsqueeze(-1)
        e2 = self.exponents()[..., 1].unsqueeze(-1)
        sx = self.scale()[..., 0].unsqueeze(-1)
        sy = self.scale()[..., 1].unsqueeze(-1)
        sz = self.scale()[..., 2].unsqueeze(-1)
        
        x = X[:, :, 0, :]
        y = X[:, :, 1, :]
        z = X[:, :, 2, :]
        
        eps = 1e-6
        # Use torch.where and clamp for speed and avoiding tensor creation
        # Original: ((x > 0).float() * 2 - 1) maps >0 to 1, <=0 to -1
        x = torch.where(x > 0, 1.0, -1.0) * torch.clamp(torch.abs(x), min=eps)
        y = torch.where(y > 0, 1.0, -1.0) * torch.clamp(torch.abs(y), min=eps)
        z = torch.where(z > 0, 1.0, -1.0) * torch.clamp(torch.abs(z), min=eps)
        
        kx = self.tapering()[..., 0].unsqueeze(-1)
        ky = self.tapering()[..., 1].unsqueeze(-1)
        
        fx = safe_mul(kx/sz, z) + 1
        fy = safe_mul(ky/sz, z) + 1
        
        fx = torch.where(fx > 0, 1.0, -1.0) * torch.clamp(torch.abs(fx), min=eps)
        fy = torch.where(fy > 0, 1.0, -1.0) * torch.clamp(torch.abs(fy), min=eps)
        
        x = x / fx
        y = y / fy
        
        term1 = safe_pow(safe_pow(x / sx, 2), 1 / e2)
        term2 = safe_pow(safe_pow(y / sy, 2), 1 / e2)
        term3 = safe_pow(safe_pow(z / sz, 2), 1 / e1)
        
        f_func = safe_pow(safe_pow(term1 + term2, e2 / e1) + term3, -e1 / 2)
        sdf = (1 - f_func) #no need to approximated distance, used only for eval
        return sdf
    
    def compute_losses(self, forward_out):
        # Return zero loss tensor (naive solver)
        return self._, {}
    
    def forward(self):
        # Naive solver: forward does no computation.
        return {}

    def update_handler(self, compute_meshes=True):
        for i, idx in enumerate(self.indices):
            mask = self.exist_mask[i].cpu().numpy()
            self.pred_handler.exist[idx] = np.expand_dims(mask, axis=1)
            self.pred_handler.scale[idx][mask] = self.scale()[i][mask].detach().cpu().numpy()
            self.pred_handler.exponents[idx][mask] = self.exponents()[i][mask].detach().cpu().numpy()
            self.pred_handler.tapering[idx][mask] = self.tapering()[i][mask].detach().cpu().numpy()
            self.pred_handler.rotation[idx][mask] = self.rotation()[i][mask].detach().cpu().numpy()
            self.pred_handler.translation[idx][mask] = self.translation[i][mask].detach().cpu().numpy()

        if compute_meshes:
            return self.pred_handler, self.pred_handler.get_meshes(resolution=30)
        else:
            return self.pred_handler

