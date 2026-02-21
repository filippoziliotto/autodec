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

from ..utils import quat2mat, mat2quat, timing
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
        cfg: Optional[object] = None,
    ):
        super().__init__()
        self.indices = indices
        self.device = device
        self.truncation = truncation
        self.pred_handler = pred_handler
        self.minS = 0.001
        self.minE, self.maxE = 0.1, 1.9

        B = len(indices)
        self.N_max = pred_handler.scale.shape[1]
        
        scale_list = []
        exp_list = []
        rot_list = []
        trans_list = []
        self.masks = [] 
        self.points = torch.zeros(B, 4096, 3, device=device)
        for i, idx in enumerate(indices):
            # --- Params ---
            mask = (pred_handler.exist[idx] > 0.5)
            self.masks.append(torch.tensor(mask, dtype=torch.bool, device=device).reshape(-1))

            s = torch.tensor(pred_handler.scale[idx], dtype=torch.float, device=device).reshape(-1, 3)
            e = torch.tensor(pred_handler.exponents[idx], dtype=torch.float, device=device).reshape(-1, 2)
            r = torch.tensor(pred_handler.rotation[idx], dtype=torch.float, device=device).reshape(-1, 3, 3)
            t = torch.tensor(pred_handler.translation[idx], dtype=torch.float, device=device).reshape(-1, 3)
            
            s[~mask.reshape(-1)] = 1.0 
            scale_list.append(torch.log(torch.clamp(s - self.minS, min=1e-6)))
            e = torch.clamp(e, self.minE + 1e-4, self.maxE - 1e-4)
            exp_list.append(torch.logit((e - self.minE) / (self.maxE - self.minE)))
            rot_list.append(mat2quat(r))
            trans_list.append(t)
            
            try:
                if external_data is not None:
                    pts = external_data['points'][i]
                else:
                    pts = torch.tensor(pred_handler.pc[idx], dtype=torch.float, device=device)
                self.points[i] = pts
            except Exception as e:
                print(f"Error loading {points_file}: {e}")
                exit()
            
        self.raw_scale = nn.Parameter(torch.stack(scale_list)) # (B, N, 3)
        self.raw_exponents = nn.Parameter(torch.stack(exp_list)) # (B, N, 2)
        self.raw_rotation = nn.Parameter(torch.stack(rot_list)) # (B, N, 4)
        self.translation = nn.Parameter(torch.stack(trans_list)) # (B, N, 3)
        self.normalize()
        
        self.exist_mask = torch.stack(self.masks) # (B, N)

    @torch.compile
    def normalize(self):
        # normalize based on surface points
        translation = torch.mean(self.points, dim=1)  # (B,3)
        self.points = self.points - translation.unsqueeze(1)

        scale = 2 * torch.amax(torch.abs(self.points), dim=(1, 2))  # (B,)
        self.points = self.points / scale.view(-1, 1, 1)

        self.normalization_translation = translation
        self.normalization_scale = scale

        with torch.no_grad():
            self.translation.data = (self.translation.data - translation.unsqueeze(1)) / scale.view(-1, 1, 1)
            # Update raw_scale so that scale() reflects the normalization
            new_s = self.scale() / scale.view(-1, 1, 1)
            self.raw_scale.data = torch.log(torch.clamp(new_s - self.minS, min=1e-6))
        
        return translation, scale
    
    @torch.compile
    def scale(self):
        return torch.exp(self.raw_scale) + self.minS

    @torch.compile   
    def exponents(self):
        return (torch.sigmoid(self.raw_exponents) * (self.maxE - self.minE)) + self.minE

    @torch.compile  
    def rotation(self):
        return quat2mat(self.raw_rotation)
    
    def get_param_groups(self):
        lrs = {
            "raw_scale": 2e-2,
            "raw_exponents": 1e-2,
        }
        groups = []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            lr = lrs.get(name, 2e-3)
            groups.append({"params": [param], "lr": lr})
        return groups

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
        
        r0 = torch.sqrt(x**2 + y**2 + z**2)
        
        term1 = safe_pow(safe_pow(x / sx, 2), 1 / e2)
        term2 = safe_pow(safe_pow(y / sy, 2), 1 / e2)
        term3 = safe_pow(safe_pow(z / sz, 2), 1 / e1)
        
        f_func = safe_pow(safe_pow(term1 + term2, e2 / e1) + term3, -e1 / 2)
        
        sdf = safe_mul(r0, (1 - f_func))
        return sdf
    
    @torch.compile
    def _compute_losses(self, sdfs):
        weight_pos = 2.0
        weight_neg = 1.0
        weight_scale = 3e-3
        
        pos_part = torch.clamp(sdfs, min=0)
        neg_part = torch.clamp(sdfs, max=0)
        Lsdf = weight_pos * torch.mean(pos_part, dim=-1) + weight_neg * torch.mean(torch.abs(neg_part), dim=-1)
        Lsdf /= weight_pos + weight_neg
        # Lsdf = (sdfs**2).mean(dim=-1)
        
        Lreg = weight_scale * torch.norm(self.scale(), p=1, dim=1).mean(dim=-1)
        
        loss = Lsdf + Lreg
        return loss, Lsdf, Lreg
    
    def compute_losses(self, forward_out):
        loss, Lsdf, Lreg = self._compute_losses(forward_out.get('sdfs'))
        return loss, {"sdf": Lsdf, "reg": Lreg}

    def forward(self):
        all_sdfs = self.sdf_batch(self.points.transpose(1, 2).contiguous()) # (B, N, M_total)
    
        # mask out non-existing primitives
        mask = self.exist_mask.unsqueeze(-1).expand_as(all_sdfs)
        all_sdfs[~mask] = float('inf')

        # Union is min(sdf)
        min_sdf, _ = torch.min(all_sdfs, dim=1) # (B, M)
        return { 'sdfs': min_sdf }

    def update_handler(self, denormalize=True, compute_meshes=True):
        for i, idx in enumerate(self.indices):
            mask = self.exist_mask[i].cpu().numpy()
            if not np.any(mask): continue
            
            # Denormalize
            s = self.scale()[i][mask].detach().cpu().numpy()
            t = self.translation[i][mask].detach().cpu().numpy()
            if denormalize:
                norm_trans = self.normalization_translation[i].detach().cpu().numpy()
                norm_scale = self.normalization_scale[i].detach().cpu().numpy()
                s = s * norm_scale
                t = (t * norm_scale) + norm_trans

            self.pred_handler.scale[idx][mask] = s
            self.pred_handler.translation[idx][mask] = t
            self.pred_handler.exponents[idx][mask] = self.exponents()[i][mask].detach().cpu().numpy()
            self.pred_handler.rotation[idx][mask] = self.rotation()[i][mask].detach().cpu().numpy()
            
        if compute_meshes:
            meshes = [self.pred_handler.get_mesh(idx, resolution=30) for idx in self.indices]
            return self.pred_handler, meshes
        else:
            return self.pred_handler
