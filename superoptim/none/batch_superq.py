import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
        external_data: dict = None,
        cfg: Optional[object] = None
    ):
        super().__init__()
        self.indices = indices
        self.device = device
        self.truncation = truncation
        self.pred_handler = pred_handler
        self.minS = 0.01
        self.minE, self.maxE = 0.1, 1.9
        
        self.cfg = cfg
        self.enable_tapering = self._parse_flag('tapering', True)
        self.enable_bending = self._parse_flag('bending', True)

        B = len(indices)
        self.N_max = pred_handler.scale.shape[1]

        scale_list = []
        exp_list = []
        rot_list = []
        trans_list = []
        tp_list = []
        bd_list = []
        self.masks = [] 

        for i, idx in enumerate(indices):
            # --- Params ---
            mask = (pred_handler.exist[idx] > 0.5)
            self.masks.append(torch.tensor(mask, dtype=torch.bool, device=device).reshape(-1))

            s = torch.tensor(pred_handler.scale[idx], dtype=torch.float, device=device).reshape(-1, 3)
            e = torch.tensor(pred_handler.exponents[idx], dtype=torch.float, device=device).reshape(-1, 2)
            r = torch.tensor(pred_handler.rotation[idx], dtype=torch.float, device=device).reshape(-1, 3, 3)
            t = torch.tensor(pred_handler.translation[idx], dtype=torch.float, device=device).reshape(-1, 3)
            
            tp = torch.tensor(pred_handler.tapering[idx], dtype=torch.float, device=device).reshape(-1, 2)
            bd = torch.tensor(pred_handler.bending[idx], dtype=torch.float, device=device).reshape(-1, 6)
            
            scale_list.append(s)
            exp_list.append(e)
            rot_list.append(mat2quat(r))
            trans_list.append(t)
            tp_list.append(tp)
            bd_list.append(bd)
            
        self.raw_scale = nn.Parameter(torch.stack(scale_list)) # (B, N, 3)
        self.raw_exponents = nn.Parameter(torch.stack(exp_list)) # (B, N, 2)
        self.raw_rotation = nn.Parameter(torch.stack(rot_list)) # (B, N, 4)
        self.translation = nn.Parameter(torch.stack(trans_list)) # (B, N, 3)
        
        self.raw_tapering = nn.Parameter(torch.stack(tp_list)) # (B, N, 2)
        self.raw_bending = nn.Parameter(torch.stack(bd_list)) # (B, N, 6)
        
        self.exist_mask = torch.stack(self.masks) # (B, N)

    def _parse_flag(self, key, default=True):
        if self.cfg is None:
            return default
        return bool(getattr(self.cfg, key))
    
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
    
    @torch.compile
    def bending(self):
        return self.raw_bending[..., 0::2], self.raw_bending[..., 1::2]
    
    def get_param_groups(self):
        groups = []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            groups.append({"params": [param], "lr": 2e-3})
        return groups

    @torch.compile
    def inverse_bending_axis(self, x, y, z, kb, alpha, axis):
        if axis == 'z':
            u, v, w = x, y, z
        elif axis == 'x':
            u, v, w = y, z, x
        elif axis == 'y':
            u, v, w = z, x, y
        else:
             return x, y, z

        # Precompute reciprocal efficiently
        inv_kb = 1.0 / (kb + 1e-6)
        
        angle_offset = torch.atan2(v, u)
        R = torch.sqrt(u**2 + v**2) * torch.cos(alpha - angle_offset)
        gamma = torch.atan2(w, inv_kb - R)
        r = inv_kb - torch.sqrt(w**2 + (inv_kb - R)**2)
        
        u = u - (R - r) * torch.cos(alpha)
        v = v - (R - r) * torch.sin(alpha)
        w = inv_kb * gamma

        if axis == 'z':
            return u, v, w
        elif axis == 'x':
            return w, u, v
        elif axis == 'y':
            return v, w, u
    
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
        
        # Apply tapering if enabled
        if self.enable_tapering:
            kx = self.tapering()[..., 0].unsqueeze(-1)
            ky = self.tapering()[..., 1].unsqueeze(-1)
            
            fx = safe_mul(kx/sz, z) + 1
            fy = safe_mul(ky/sz, z) + 1
            
            fx = torch.where(fx > 0, 1.0, -1.0) * torch.clamp(torch.abs(fx), min=eps)
            fy = torch.where(fy > 0, 1.0, -1.0) * torch.clamp(torch.abs(fy), min=eps)
            
            x = x / fx
            y = y / fy

        # Apply bending if enabled
        if self.enable_bending:
            kb, alpha = self.bending()
            x, y, z = self.inverse_bending_axis(x, y, z, kb[..., 0].unsqueeze(-1), alpha[..., 0].unsqueeze(-1), 'z')
            x, y, z = self.inverse_bending_axis(x, y, z, kb[..., 1].unsqueeze(-1), alpha[..., 1].unsqueeze(-1), 'x')
            x, y, z = self.inverse_bending_axis(x, y, z, kb[..., 2].unsqueeze(-1), alpha[..., 2].unsqueeze(-1), 'y')

        r0 = torch.sqrt(x**2 + y**2 + z**2)
        
        term1 = safe_pow(safe_pow(x / sx, 2), 1 / e2)
        term2 = safe_pow(safe_pow(y / sy, 2), 1 / e2)
        term3 = safe_pow(safe_pow(z / sz, 2), 1 / e1)
        
        f_func = safe_pow(safe_pow(term1 + term2, e2 / e1) + term3, -e1 / 2)
        
        sdf = safe_mul(r0, (1 - f_func))
        # sdf = (1 - f_func)
        return sdf

    def update_handler(self, compute_meshes=True):
        if compute_meshes:
            return self.pred_handler, self.pred_handler.get_meshes(resolution=30)
        else:
            return self.pred_handler

