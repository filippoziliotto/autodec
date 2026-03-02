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
from superdec.utils.transforms import transform_to_primitive_frame
from superdec.loss.utils import sampling_from_parametric_space_to_equivalent_points
from superdec.loss.sampler import EqualDistanceSamplerSQ

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
        normals_list = []
        assign_list = []
        self.points = torch.zeros(B, 4096, 3, device=device)
        normals_list = []
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
            
            if external_data is not None:
                pts  = external_data['points'][i]
                nrms = external_data['normals'][i]
            else:
                ply  = ply_paths[i] if ply_paths else None
                pts  = torch.tensor(pred_handler.pc[idx], dtype=torch.float, device=device)
                nrms = None
                try:
                    data = np.load(ply)
                    pts_ply     = torch.tensor(np.array(data['points']),  dtype=torch.float, device=device)
                    ply_normals = torch.tensor(np.array(data['normals']), dtype=torch.float, device=device)
                    distances   = torch.cdist(pts, pts_ply)
                    closest     = torch.argmin(distances, dim=1)
                    nrms        = ply_normals[closest]
                except Exception as e:
                    print(f"Error loading {ply}: {e}")

            if nrms is None:
                print("Normals are needed for cuboid loss")
                exit()

            self.points[i] = pts
            normals_list.append(nrms)

            # --- Assign matrix ---
            am = torch.tensor(pred_handler.assign_matrix[idx], dtype=torch.float, device=device)  # (N, P)
            assign_list.append(am.permute(1, 0))  # (P, N)
            
        self.raw_scale = nn.Parameter(torch.stack(scale_list)) # (B, N, 3)
        self.raw_exponents = nn.Parameter(torch.stack(exp_list)) # (B, N, 2)
        self.raw_rotation = nn.Parameter(torch.stack(rot_list)) # (B, N, 4)
        self.translation = nn.Parameter(torch.stack(trans_list)) # (B, N, 3)
        s, t = self.normalize()
        self.n_scale = s
        self.n_trans = t
        
        self.exist_mask = torch.stack(self.masks)  # (B, N)
        self.normals = torch.stack(normals_list)    # (B, N_pts, 3)
        self.assign_matrix = torch.stack(assign_list)  # (B, P, N_pts)

        self._init_loss_components(cfg)

    def _init_loss_components(self, cfg=None):
        """Initialise buffers and modules used by cuboid and CD losses."""
        n_samples = getattr(cfg, 'n_samples', 500) if cfg is not None else 500
        self.sampler = EqualDistanceSamplerSQ(n_samples=n_samples, D_eta=0.05, D_omega=0.05)

        # Weights (can be overridden via cfg)
        self.w_cub = getattr(cfg, 'w_cub', 1.0) if cfg is not None else 1.0
        self.w_cd  = getattr(cfg, 'w_cd',  1.0) if cfg is not None else 1.0

        # Static tensors for cuboid loss
        dev = self.device
        self.register_buffer(
            'mask_project',
            torch.FloatTensor([[0,1,1],[0,1,1],[1,0,1],[1,0,1],[1,1,0],[1,1,0]])
                .unsqueeze(0).unsqueeze(0).to(dev))
        self.register_buffer(
            'mask_plane',
            torch.FloatTensor([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]])
                .unsqueeze(0).unsqueeze(0).to(dev))
        self.register_buffer(
            'cube_normal',
            torch.FloatTensor([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
                .unsqueeze(0).unsqueeze(0).to(dev))
        self.register_buffer(
            'cube_planes',
            torch.FloatTensor([[-1,-1,-1],[1,1,1]])
                .unsqueeze(0).unsqueeze(0).to(dev))
        self.cos_sim_cubes = nn.CosineSimilarity(dim=4, eps=1e-4)

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
    
    def _compute_cuboid_loss(self, pc_inver, normals_inver, scale, assign_matrix):
        """Cuboid loss: project each assigned point onto the nearest cube face.

        Mirrors SuperDecLoss.compute_cuboid_loss.
        """
        B, P, N, _ = pc_inver.shape

        planes_scaled = (self.cube_planes.repeat(B, P, 1, 1) *
                         scale.unsqueeze(2).repeat(1, 1, 2, 1))            # (B,P,2,3)
        planes_scaled = planes_scaled.unsqueeze(1).repeat(1, N, 1, 3, 1).reshape(B, N, P * 6, 3)

        mask_project   = self.mask_project.repeat(B, N, P, 1)              # (B,N,P*6,3)
        mask_plane     = self.mask_plane.repeat(B, N, P, 1)
        cube_normal    = self.cube_normal.unsqueeze(2).repeat(B, N, P, 1, 1)  # (B,N,P,6,3)
        scale_reshaped = scale.unsqueeze(1).repeat(1, N, 1, 6).reshape(B, N, P * 6, 3)

        normals_inver_rs = normals_inver.permute(0, 2, 1, 3).unsqueeze(3).repeat(1, 1, 1, 6, 1)
        _, idx_normals = torch.max(self.cos_sim_cubes(normals_inver_rs, cube_normal), dim=-1, keepdim=True)

        pc_project = (pc_inver.permute(0, 2, 1, 3).repeat(1, 1, 1, 6).reshape(B, N, P * 6, 3) *
                      mask_project + planes_scaled * mask_plane)
        pc_project = torch.max(torch.min(pc_project, scale_reshaped), -scale_reshaped)
        pc_project = pc_project.view(B, N, P, 6, 3)
        pc_project = torch.gather(pc_project, dim=3,
                                  index=idx_normals.unsqueeze(-1).repeat(1, 1, 1, 1, 3)).squeeze(3)
        pc_project = pc_project.permute(0, 2, 1, 3)  # (B, P, N, 3)

        diff = ((pc_project - pc_inver) ** 2).sum(-1).permute(0, 2, 1)  # (B, N, P)
        loss = torch.sum(diff * assign_matrix.permute(0, 2, 1), -1).mean(dim=1)  # (B,)
        return loss

    def _compute_cd_loss(self, pc_inver, scale, exps, assign_matrix):
        """Chamfer distance loss between sampled SQ surface points and input cloud.

        Mirrors SuperDecLoss.compute_cd_loss.
        """
        exist = self.exist_mask.float().unsqueeze(-1)  # (B, P, 1)

        # Sample points on superquadric surfaces
        X_SQ, _ = sampling_from_parametric_space_to_equivalent_points(
            scale, exps, self.sampler)                 # (B, P, S, 3)

        # Squared distances between SQ samples and input points
        diff = X_SQ.unsqueeze(3) - pc_inver.unsqueeze(2)  # (B, P, S, N, 3)
        D = (diff ** 2).sum(-1)                            # (B, P, S, N)

        # Point-to-primitive: for each input point, nearest SQ sample
        pcl_to_prim = D.min(dim=2)[0]                      # (B, P, N)
        pcl_to_prim_loss = (pcl_to_prim.transpose(-1, -2) *
                            assign_matrix.transpose(-1, -2)).sum(-1).mean(dim=1)  # (B,)

        # Primitive-to-point: for each SQ sample, nearest input point
        prim_to_pcl = D.min(dim=3)[0].mean(dim=-1)        # (B, P)
        prim_to_pcl_loss = (prim_to_pcl * exist.squeeze(-1)).sum(dim=-1)  # (B,)
        prim_to_pcl_loss = prim_to_pcl_loss / (exist.squeeze(-1).sum(dim=-1) + 1e-6)  # (B,)

        return pcl_to_prim_loss + prim_to_pcl_loss  # (B,)

    def compute_losses(self, forward_out):
        """Compute cuboid + CD losses from the output of forward().

        Returns:
            loss (scalar Tensor), loss_dict (dict of floats)
        """
        scale        = forward_out['scale']         # (B, P, 3)
        exps         = forward_out['shape']         # (B, P, 2)
        assign_matrix = forward_out['assign_matrix'] # (B, P, N)
        pc_inver     = forward_out['pc_inver']      # (B, P, N, 3)

        loss = torch.zeros(len(self.indices), device=self.device)  # (B,)
        loss_dict = {}

        # --- Cuboid loss (requires point normals) ---
        if self.w_cub > 0 and self.normals is not None:
            normals_inver = transform_to_primitive_frame(
                self.normals, self.translation, self.rotation())  # (B, P, N, 3)
            cub_loss = self._compute_cuboid_loss(pc_inver, normals_inver, scale, assign_matrix)  # (B,)
            loss = loss + self.w_cub * cub_loss
            loss_dict['cuboid'] = cub_loss

        # --- Chamfer distance loss ---
        if self.w_cd > 0:
            cd_loss = self._compute_cd_loss(pc_inver, scale, exps, assign_matrix)  # (B,)
            loss = loss + self.w_cd * cd_loss
            loss_dict['cd'] = cd_loss

        loss_dict['total'] = loss
        return loss, loss_dict

    def forward(self):
        """Build the out_dict used by compute_losses.

        Returns:
            dict with keys: scale, shape, trans, rotate, exist,
                            assign_matrix, pc_inver
        """
        scale  = self.scale()       # (B, P, 3)
        exps   = self.exponents()   # (B, P, 2)
        rot    = self.rotation()    # (B, P, 3, 3)
        trans  = self.translation   # (B, P, 3)
        exist  = self.exist_mask.float().unsqueeze(-1)  # (B, P, 1)

        # Transform input point cloud into each primitive's local frame
        pc_inver = transform_to_primitive_frame(
            self.points, trans, rot)  # (B, P, N_pts, 3)

        assign_matrix = self.assign_matrix  # (B, P, N_pts) – fixed from pred_handler

        return {
            'scale':          scale,
            'shape':          exps,
            'trans':          trans,
            'rotate':         rot,
            'exist':          exist,
            'assign_matrix':  assign_matrix,
            'pc_inver':       pc_inver,
        }

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
