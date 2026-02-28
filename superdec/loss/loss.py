import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

from superdec.loss.utils import sampling_from_parametric_space_to_equivalent_points, parametric_to_points_extended, sdf_batch
from superdec.loss.sampler import EqualDistanceSamplerSQ
from superdec.utils.transforms import transform_to_primitive_frame, quat2mat, mat2quat
from superdec.utils.safe_operations import safe_pow, safe_mul

class BaseLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_sps = getattr(cfg, 'w_sps', 0.0)
        self.w_ext = getattr(cfg, 'w_ext', 0.0)

    def compute_existence_loss(self, assign_matrix, exist):
        thred = 24
        loss = nn.BCELoss().cuda()
        gt = (assign_matrix.sum(1) > thred).to(torch.float32).detach()
        entropy = loss(exist.squeeze(-1), gt)
        return entropy

    def get_sparsity_loss(self, assign_matrix):
        num_points = assign_matrix.shape[1]
        norm_05 = (assign_matrix.sum(1)/num_points + 0.01).sqrt().mean(1).pow(2)
        norm_05 = torch.mean(norm_05)
        return norm_05

    def compute_common_losses(self, out_dict, loss_dict):
        loss = 0
        loss_dict['expected_prim_num'] = out_dict['exist'].squeeze(-1).sum(-1).mean().data.detach().item()

        if self.w_ext > 0:
            exist_loss = self.compute_existence_loss(out_dict['assign_matrix'], out_dict['exist'])
            loss += self.w_ext * exist_loss
            loss_dict['exist_loss'] = exist_loss.item()

        if self.w_sps > 0:
            sparsity_loss = self.get_sparsity_loss(out_dict['assign_matrix'])
            loss += self.w_sps * sparsity_loss
            loss_dict['sparsity_loss'] = sparsity_loss.item()
            
        return loss

class SuperDecLoss(BaseLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._init_buffers()
        self.sampler = EqualDistanceSamplerSQ(n_samples=cfg.n_samples, D_eta=0.05, D_omega=0.05)
        
        self.w_cub = cfg.w_cub
        self.w_cd = cfg.w_cd

        
        self.cos_sim_cubes = nn.CosineSimilarity(dim=4, eps=1e-4) 

    def _init_buffers(self):
        self.register_buffer('mask_project', torch.FloatTensor([[0,1,1],[0,1,1],[1,0,1],[1,0,1],[1,1,0],[1,1,0]]).unsqueeze(0).unsqueeze(0))
        self.register_buffer('mask_plane', torch.FloatTensor([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]]).unsqueeze(0).unsqueeze(0))
        self.register_buffer('cube_normal', torch.FloatTensor([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]).unsqueeze(0).unsqueeze(0))
        self.register_buffer('cube_planes', torch.FloatTensor([[-1,-1,-1],[1,1,1]]).unsqueeze(0).unsqueeze(0))

    def compute_cuboid_loss(self, pc_inver, normals_inver, out_dict):
        B, P = out_dict['scale'].shape[:2]
        N = pc_inver.shape[2]

        planes_scaled = self.cube_planes.repeat(B, P, 1, 1) * out_dict['scale'].unsqueeze(2).repeat(1,1,2,1)
        planes_scaled = planes_scaled.unsqueeze(1).repeat(1,N,1,3,1).reshape(B,N,P*6,3)
        mask_project = self.mask_project.repeat(B,N,P,1)
        mask_plane = self.mask_plane.repeat(B,N,P,1)
        cube_normal = self.cube_normal.unsqueeze(2).repeat(B,N,P,1,1)
        scale_reshaped = out_dict['scale'].unsqueeze(1).repeat(1,N,1,6).reshape(B,N,P*6,3)
        
        normals_inver_reshaped = normals_inver.permute(0,2,1,3).unsqueeze(3).repeat(1,1,1,6,1)
        _, idx_normals_sim_max = torch.max(self.cos_sim_cubes(normals_inver_reshaped, cube_normal),dim=-1,keepdim=True)

        pc_project = pc_inver.permute(0,2,1,3).repeat(1,1,1,6).reshape(B,N,P*6,3) * mask_project + planes_scaled * mask_plane
        pc_project = torch.max(torch.min(pc_project, scale_reshaped), -scale_reshaped).view(B, N, P, 6, 3)  
        pc_project = torch.gather(pc_project, dim=3, index = idx_normals_sim_max.unsqueeze(-1).repeat(1,1,1,1,3)).squeeze(3).permute(0,2,1,3)

        diff = ((pc_project - pc_inver) ** 2).sum(-1).permute(0,2,1)
        diff = torch.mean(torch.mean(torch.sum(diff * out_dict['assign_matrix'], -1), 1))

        return diff

    def compute_cd_loss(self, pc_inver, out_dict):
        weights = out_dict['assign_matrix']  # [B, P, N]
        scale = out_dict['scale']             # [B, P, 3]
        shape = out_dict['shape']             # [B, P, 2]
        exist = out_dict['exist']             # [B, P, 1]

        # Sample points and normals on superquadrics
        X_SQ, normals = sampling_from_parametric_space_to_equivalent_points(scale, shape, self.sampler)
        normals = normals.detach()        # [B, P, S, 3]
        
        # Compute squared distances
        diff = X_SQ.unsqueeze(3) - pc_inver.unsqueeze(2)  # [B, P, S, N, 3]
        D = (diff ** 2).sum(-1)                            # [B, P, S, N]

        # Point-to-Primitive Chamfer
        pcl_to_prim_loss = D.min(dim=2)[0]      
        pcl_to_prim_loss = (pcl_to_prim_loss.transpose(-1,-2) * weights).sum(-1).mean()

        # Primitive-to-Point Chamfer
        distances_bis = D.min(dim=3)[0]                     # [B, P, S]
        prim_to_pcl = distances_bis.mean(dim=-1)            # [B, P]
        prim_to_pcl_loss = (prim_to_pcl * exist.squeeze(-1)).sum(dim=-1)
        prim_to_pcl_loss = (prim_to_pcl_loss / (exist.squeeze(-1).sum(dim=-1) + 1e-6)).mean()

        return pcl_to_prim_loss, prim_to_pcl_loss

    def forward(self, batch, out_dict):
        pc, normals = batch['points'].cuda().float(), batch['normals'].cuda().float()
        pc_inver = transform_to_primitive_frame(pc, out_dict['trans'], out_dict['rotate'])
        normals_inver = transform_to_primitive_frame(normals, out_dict['trans'], out_dict['rotate'])

        loss = 0
        loss_dict = {}

        if self.w_cub > 0:
            cub_loss = self.compute_cuboid_loss(pc_inver, normals_inver, out_dict)
            loss += self.w_cub * cub_loss
            loss_dict['cub_loss'] = cub_loss.item()

        if self.w_cd > 0:
            pcl_to_prim_loss, prim_to_pcl_loss = self.compute_cd_loss(pc_inver, out_dict)
            cd_loss = pcl_to_prim_loss + prim_to_pcl_loss
            loss += self.w_cd * cd_loss
            loss_dict['cd_loss'] = cd_loss.item()

        loss += self.compute_common_losses(out_dict, loss_dict)
        
        loss_dict['all'] = loss.item()
        return loss, loss_dict

class ParamLoss(BaseLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.w_exist = getattr(cfg, 'w_exist', 1.0)
        self.w_scale = getattr(cfg, 'w_scale', 1.0)
        self.w_shape = getattr(cfg, 'w_shape', 1.0)
        self.w_trans = getattr(cfg, 'w_trans', 1.0)
        self.w_rot = getattr(cfg, 'w_rot', 1.0)
        self.w_tapering = getattr(cfg, 'w_tapering', 1.0)
        self.w_bending = getattr(cfg, 'w_bending', 1.0)
        
        self.c_exist = getattr(cfg, 'c_exist', 0.0)
        self.c_scale = getattr(cfg, 'c_scale', 0.0)
        self.c_shape = getattr(cfg, 'c_shape', 0.0)
        self.c_trans = getattr(cfg, 'c_trans', 0.0)
        self.c_rot = getattr(cfg, 'c_rot', 0.0)
        self.c_tapering = getattr(cfg, 'c_tapering', 0.0)
        self.c_bending = getattr(cfg, 'c_bending', 0.0)
        
    def scale_mse(self, pred, gt, mask=None):
        if mask is None:
            return torch.cdist(pred, gt, p=2)**2 / 3.0
        loss = nn.MSELoss(reduction='none')(pred, gt)
        return (loss * mask).sum() / (mask.sum() * 3 + 1e-6)

    def shape_mse(self, pred, gt, mask=None):
        if mask is None:
            return torch.cdist(pred, gt, p=2)**2 / 2.0
        loss = nn.MSELoss(reduction='none')(pred, gt)
        return (loss * mask).sum() / (mask.sum() * 2 + 1e-6)

    def trans_mse(self, pred, gt, mask=None):
        if mask is None:
            return torch.cdist(pred, gt, p=2)**2 / 3.0
        loss = torch.norm(pred - gt, dim=-1)  # (B, P)
        return (loss * mask.squeeze(-1)).sum() / (mask.sum() * 3 + 1e-6)

    def rot_mse(self, pred, gt, mask=None):
        if mask is None:
            return torch.cdist(pred, gt, p=2)**2 / 4.0
        
        # Geodesic distance (angle in radians)
        dot = torch.abs((pred * gt).sum(dim=-1))  # (B, P)
        dot = torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6)
        theta = 2.0 * torch.acos(dot).unsqueeze(-1)  # (B, P)
        loss = theta ** 2
        return (loss * mask).sum() / (mask.sum() + 1e-6)

    def tapering_mse(self, pred, gt, mask=None):
        if mask is None:
            return torch.cdist(pred, gt, p=2)**2 / 2.0
        loss = nn.MSELoss(reduction='none')(pred, gt)
        return (loss * mask).sum() / (mask.sum() * 2 + 1e-6)

    def bending_mse(self, pred_k, pred_a, gt, mask=None):
        B, P = pred_k.shape[:2]
        pred = torch.stack([pred_k, pred_a], dim=-1).reshape(B, P, 6)
        if mask is None:
            return torch.cdist(pred, gt, p=2)**2 / 6.0
        loss = nn.MSELoss(reduction='none')(pred, gt)
        return (loss * mask).sum() / (mask.sum() * 6 + 1e-6)
    
    def get_sparsity_loss(self, assign_matrix):
        num_points = assign_matrix.shape[1]
        norm_05 = (assign_matrix.sum(1)/num_points + 0.01).sqrt().mean(1).pow(2)
        norm_05 = torch.mean(norm_05)
        return norm_05
    
    def forward(self, batch, out_dict):
        gt_scale = batch['gt_scale'].cuda().float()
        gt_shape = batch['gt_shape'].cuda().float()
        gt_trans = batch['gt_trans'].cuda().float()
        gt_rotate = batch['gt_rotate_q'].cuda().float()
        gt_exist = batch['gt_exist'].cuda().float()
        if 'tapering' in out_dict:
            gt_tapering = batch['gt_tapering'].cuda().float()
        if 'bending_k' in out_dict and 'bending_a' in out_dict:
            gt_bending = batch['gt_bending'].cuda().float()

        rotation = mat2quat(out_dict['rotate'])
        B, P = gt_scale.shape[:2]
        with torch.no_grad():
            device = gt_scale.device
            C = torch.zeros((B, P, P), device=device)

            if self.c_scale > 0:
                cost_scale = self.c_scale * self.scale_mse(out_dict['scale'], gt_scale)
                C += cost_scale

            if self.c_shape > 0:
                cost_shape = self.c_shape * self.shape_mse(out_dict['shape'], gt_shape)
                C += cost_shape

            if self.c_trans > 0:
                cost_trans = self.c_trans * self.trans_mse(out_dict['trans'], gt_trans)
                C += cost_trans

            if self.c_rot > 0:
                cost_rot = self.c_rot * self.rot_mse(rotation, gt_rotate)
                C += cost_rot

            if 'tapering' in out_dict and self.c_tapering > 0:
                cost_tapering = self.c_tapering * self.tapering_mse(out_dict['tapering'], gt_tapering)
                C += cost_tapering

            if 'bending_k' in out_dict and 'bending_a' in out_dict and self.c_bending > 0:
                cost_bending = self.c_bending * self.bending_mse(out_dict['bending_k'], out_dict['bending_a'], gt_bending)
                C += cost_bending

            if self.c_exist > 0:
                cost_exist = self.c_exist * torch.abs(out_dict['exist'].view(B, P, 1) - gt_exist.view(B, 1, P))
                C += cost_exist

            C_np = C.cpu().numpy()
            
            indices = []
            for b in range(B):
                row_ind, col_ind = linear_sum_assignment(C_np[b])
                indices.append(col_ind)
                
            indices = torch.tensor(np.array(indices), device=gt_scale.device)
            batch_idx = torch.arange(B, device=gt_scale.device).unsqueeze(1).expand(B, P)
            gt_scale = gt_scale[batch_idx, indices]
            gt_shape = gt_shape[batch_idx, indices]
            gt_trans = gt_trans[batch_idx, indices]
            gt_rotate = gt_rotate[batch_idx, indices]
            gt_exist = gt_exist[batch_idx, indices]
            if 'tapering' in out_dict:
                gt_tapering = gt_tapering[batch_idx, indices]
            if 'bending_k' in out_dict and 'bending_a' in out_dict:
                gt_bending = gt_bending[batch_idx, indices]
        
        loss = 0
        loss_dict = {}            
        mask = (gt_exist > 0.5).float()

        # Existance (binary cross-entropy)
        exist_loss = nn.BCELoss()(out_dict['exist'], mask)
        loss += self.w_exist * exist_loss
        loss_dict['param_exist'] = exist_loss.item()
        
        # Scale
        scale_loss = self.scale_mse(out_dict['scale'], gt_scale, mask)
        loss += self.w_scale * scale_loss
        loss_dict['param_scale'] = scale_loss.item()
        
        # Shape
        shape_loss = self.shape_mse(out_dict['shape'], gt_shape, mask)
        loss += self.w_shape * shape_loss
        loss_dict['param_shape'] = shape_loss.item()
        
        # Translation
        trans_loss = self.trans_mse(out_dict['trans'], gt_trans, mask)
        loss += self.w_trans * trans_loss
        loss_dict['param_trans'] = trans_loss.item()
        
        # Rotation
        rot_loss = self.rot_mse(rotation, gt_rotate, mask)
        loss += self.w_rot * rot_loss
        loss_dict['param_rot'] = rot_loss.item()

        # Tapering
        if 'tapering' in out_dict:
            tapering_loss = self.tapering_mse(out_dict['tapering'], gt_tapering, mask)
            loss += self.w_tapering * tapering_loss
            loss_dict['param_tapering'] = tapering_loss.item()

        # Bending
        if 'bending_k' in out_dict and 'bending_a' in out_dict:
            bend_loss = self.bending_mse(out_dict['bending_k'], out_dict['bending_a'], gt_bending, mask)
            loss += self.w_bending * bend_loss
            loss_dict['param_bending'] = bend_loss.item()
        
        # stop trainer from complaining about unused parameters (TODO should we use it?)
        loss += self.compute_common_losses(out_dict, loss_dict)
        
        loss_dict['all'] = loss.item()
        return loss, loss_dict

class ParamLossGeometric(ParamLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.geometric_cd = getattr(cfg, 'geometric_cd', False) 
        self.free_axis = self.geometric_cd
        
        self.w_geometric = getattr(cfg, 'w_geometric', 1.0)
        self.c_geometric = getattr(cfg, 'c_geometric', 1.0)
        # self.w_sdf = getattr(cfg, 'w_sdf', 0.0)
    
    def geometric_error_cd(self, pred, gt, mask=None):
        if mask is None:
            B, P, _, _ = gt.shape
            gt_ext = gt.unsqueeze(1).expand(B, P, P, -1, 3)
            dists = torch.norm(pred.unsqueeze(4) - gt_ext.unsqueeze(3), dim=-1)
            mean_pred_to_gt = dists.min(dim=4)[0].mean(dim=3)  # (B, P_pred, P_gt)
            mean_gt_to_pred = dists.min(dim=3)[0].mean(dim=3)  # (B, P_pred, P_gt)
            chamfer = mean_pred_to_gt + mean_gt_to_pred
            return chamfer

        # Masked mode: compute per-primitive chamfer and aggregate using mask
        dists = torch.norm(pred.unsqueeze(3) - gt.unsqueeze(2), dim=-1)  # (B, P, Np, Ng)
        mean_pred_to_gt = dists.min(dim=3)[0].mean(dim=2)  # (B, P)
        mean_gt_to_pred = dists.min(dim=2)[0].mean(dim=2)  # (B, P)
        chamfer_per_prim = mean_pred_to_gt + mean_gt_to_pred  # (B, P)
        return (chamfer_per_prim * mask.squeeze(-1)).sum() / (mask.sum() + 1e-6)
    
    def geometric_error_11(self, pred, gt, mask=None):
        if mask is None:
            B, P, _, _ = gt.shape
            gt_ext = gt.unsqueeze(1).expand(B, P, P, -1, 3)
            return torch.norm(pred - gt_ext, dim=-1).mean(dim=-1) # (B, P, P)
        
        dist = torch.norm(pred - gt, dim=-1)  # (B, P, N)
        loss = dist.mean(dim=-1)  # (B, P)
        return (loss * mask.squeeze(-1)).sum() / (mask.sum() + 1e-6)
    
    def geometric_error(self, pred, gt, mask=None):
        if self.geometric_cd:
            return self.geometric_error_cd(pred, gt, mask)
        else:
            return self.geometric_error_11(pred, gt, mask)
    
    def tapering_mse(self, pred, gt, mask=None):
        if not self.free_axis:
            return super().tapering_mse(pred, gt, mask)

        # pred shape (B, P, 2)
        _pred = pred.sum(-1).unsqueeze(-1)
        _gt = gt.sum(-1).unsqueeze(-1)
        if mask is None:
            return torch.cdist(_pred, _gt, p=2)**2 / 2.0

        loss = nn.MSELoss(reduction='none')(_pred, _gt)
        return (loss * mask).sum() / (mask.sum() * 2 + 1e-6)
    
    def bending_mse(self, pred_k, pred_a, gt, mask=None):
        if not self.free_axis:
            return super().bending_mse(pred_k, pred_a, gt, mask)

        _pred_k = pred_k.sum(-1).unsqueeze(-1)
        _gt_k = gt[..., 0::2].sum(-1).unsqueeze(-1)
        
        # pred_k shape (B, P, 3)
        if mask is None:
            return torch.cdist(_pred_k, _gt_k, p=2)**2 / 3.0

        loss = nn.MSELoss(reduction='none')(_pred_k, _gt_k)
        return (loss * mask).sum() / (mask.sum() * 3 + 1e-6)
    
    def forward(self, batch, out_dict):
        gt_scale = batch['gt_scale'].cuda().float()
        gt_shape = batch['gt_shape'].cuda().float()
        gt_trans = batch['gt_trans'].cuda().float()
        gt_rotate = batch['gt_rotate_q'].cuda().float()
        gt_exist = batch['gt_exist'].cuda().float()
        if 'tapering' in out_dict:
            gt_tapering = batch['gt_tapering'].cuda().float()
        if 'bending_k' in out_dict and 'bending_a' in out_dict:
            gt_bending = batch['gt_bending'].cuda().float()
        
        B, P = gt_scale.shape[:2]
        gt_pts = batch['gt_sq_points'].cuda().float()
        gt_sq_etas = batch['gt_sq_etas'].cuda().float() # [B, P, N]
        gt_sq_omegas = batch['gt_sq_omegas'].cuda().float() # [B, P, N]

        rotation = mat2quat(out_dict['rotate'])
        with torch.no_grad():
            device = gt_scale.device
            C = torch.zeros((B, P, P), device=device)

            if self.c_geometric > 0:
                # For cost computation downsample etas and omegas
                n_pts = gt_sq_etas.shape[-1]
                gt_sq_etas_ds, gt_sq_omegas_ds, gt_pts_ds = gt_sq_etas, gt_sq_omegas, gt_pts
                if n_pts > 64:
                    idx = torch.linspace(0, n_pts - 1, steps=64, dtype=torch.int, device=gt_sq_etas.device)
                    gt_sq_etas_ds = gt_sq_etas.index_select(-1, idx)
                    gt_sq_omegas_ds = gt_sq_omegas.index_select(-1, idx)
                    gt_pts_ds = gt_pts.index_select(-2, idx)
                gt_sq_etas_ext = gt_sq_etas_ds.reshape(B, -1).unsqueeze(1).expand(-1, P, -1)
                gt_sq_omegas_ext = gt_sq_omegas_ds.reshape(B, -1).unsqueeze(1).expand(-1, P, -1)
                
                tapering_pred = out_dict['tapering'] if 'tapering' in out_dict else torch.zeros_like(out_dict['shape'])
                if 'bending_k' in out_dict and 'bending_a' in out_dict:
                    bending_k_pred = out_dict['bending_k']
                    bending_a_pred = out_dict['bending_a']
                else:
                    bending_k_pred = torch.zeros_like(out_dict['scale'])
                    bending_a_pred = torch.zeros_like(out_dict['scale'])

                pts = parametric_to_points_extended(
                    out_dict['trans'], 
                    out_dict['rotate'], 
                    out_dict['scale'], 
                    out_dict['shape'], 
                    tapering_pred, 
                    bending_k_pred, 
                    bending_a_pred,
                    gt_sq_etas_ext,
                    gt_sq_omegas_ext
                )
                pts = pts.view(B, P, P, -1, 3) # [B, P, P, N, 3]
                cost_geometric = self.c_geometric * self.geometric_error(pts, gt_pts_ds)
                C += cost_geometric

            if self.c_scale > 0:
                cost_scale = self.c_scale * self.scale_mse(out_dict['scale'], gt_scale)
                C += cost_scale

            if self.c_shape > 0:
                cost_shape = self.c_shape * self.shape_mse(out_dict['shape'], gt_shape)
                C += cost_shape

            if self.c_trans > 0:
                cost_trans = self.c_trans * self.trans_mse(out_dict['trans'], gt_trans)
                C += cost_trans

            if self.c_rot > 0:
                cost_rot = self.c_rot * self.rot_mse(rotation, gt_rotate)
                C += cost_rot

            if 'tapering' in out_dict and self.c_tapering > 0:
                cost_tapering = self.c_tapering * self.tapering_mse(out_dict['tapering'], gt_tapering)
                C += cost_tapering

            if 'bending_k' in out_dict and 'bending_a' in out_dict and self.c_bending > 0:
                cost_bending = self.c_bending * self.bending_mse(out_dict['bending_k'], out_dict['bending_a'], gt_bending)
                C += cost_bending

            if self.c_exist > 0:
                cost_exist = self.c_exist * torch.abs(out_dict['exist'].view(B, P, 1) - gt_exist.view(B, 1, P))
                C += cost_exist

            C_np = C.cpu().numpy()
            
            indices = []
            for b in range(B):
                row_ind, col_ind = linear_sum_assignment(C_np[b])
                indices.append(col_ind)
                
            indices = torch.tensor(np.array(indices), device=gt_scale.device)
            batch_idx = torch.arange(B, device=gt_scale.device).unsqueeze(1).expand(B, P)
            gt_pts = gt_pts[batch_idx, indices]
            gt_sq_etas = gt_sq_etas[batch_idx, indices]
            gt_sq_omegas = gt_sq_omegas[batch_idx, indices]
            gt_scale = gt_scale[batch_idx, indices]
            gt_shape = gt_shape[batch_idx, indices]
            gt_trans = gt_trans[batch_idx, indices]
            gt_rotate = gt_rotate[batch_idx, indices]
            gt_exist = gt_exist[batch_idx, indices]
            if 'tapering' in out_dict:
                gt_tapering = gt_tapering[batch_idx, indices]
            if 'bending_k' in out_dict and 'bending_a' in out_dict:
                gt_bending = gt_bending[batch_idx, indices]
        
        loss = 0
        loss_dict = {}            
        mask = (gt_exist > 0.5).float()

        # Existance (binary cross-entropy)
        exist_loss = nn.BCELoss()(out_dict['exist'], mask)
        loss += self.w_exist * exist_loss
        loss_dict['param_exist'] = exist_loss.item()
        
        # Geometric
        tapering_pred = out_dict['tapering'] if 'tapering' in out_dict else torch.zeros_like(out_dict['shape'])
        if 'bending_k' in out_dict and 'bending_a' in out_dict:
            bending_k_pred = out_dict['bending_k']
            bending_a_pred = out_dict['bending_a']
        else:
            bending_k_pred = torch.zeros_like(out_dict['scale'])
            bending_a_pred = torch.zeros_like(out_dict['scale'])

        pts = parametric_to_points_extended(
            out_dict['trans'], 
            out_dict['rotate'], 
            out_dict['scale'],
            out_dict['shape'],
            tapering_pred,
            bending_k_pred,
            bending_a_pred,
            gt_sq_etas,
            gt_sq_omegas
        )
        geometric_loss = self.geometric_error(pts, gt_pts, mask)
        loss += self.w_geometric * geometric_loss
        loss_dict['param_geometric'] = geometric_loss.item()
        
        # pc = batch['points'].cuda().float() # [B, N_pc, 3]
        # if self.w_sdf > 0:
        #     # SDF on Surface Points
        #     all_sdfs_surf = sdf_batch(pc, out_dict) #(B, N, M_surf)
        #     assign_probs = out_dict['assign_matrix'].transpose(1, 2) #(B, N, M_surf)
            
        #     relu_sdfs_trunc = torch.abs(all_sdfs_surf) + (out_dict['exist'] < 0.5).detach()
        #     weighted_relu_sdf = (assign_probs * relu_sdfs_trunc).sum(dim=1)
        #     L_sdf = weighted_relu_sdf.mean()
        #     loss += self.w_sdf * L_sdf
        #     loss_dict['sdf'] = L_sdf.item()
            
        # if self.w_cd > 0:
        #     B_dim, P_dim, N_sq, _ = pts.shape
        #     N_pc = pc.shape[1]
            
        #     pc_exp = pc.unsqueeze(1).expand(-1, P_dim, -1, -1).reshape(B_dim * P_dim, N_pc, 3)
        #     pts_exp = pts.reshape(B_dim * P_dim, N_sq, 3)
            
        #     D = torch.cdist(pc_exp, pts_exp, p=2.0) ** 2 # [B*P, N_pc, N_sq]
        #     pc_to_gt_sq = D.min(dim=2)[0].reshape(B_dim, P_dim, N_pc) # [B, P, N_pc]
            
        #     # Compute squared distances
        #     diff = pts.unsqueeze(3) - pc.unsqueeze(2)  # [B, P, S, N, 3]
        #     D = (diff ** 2).sum(-1)                            # [B, P, S, N]

        #     # Point-to-Primitive Chamfer
        #     pcl_to_prim_loss = D.min(dim=2)[0]      
        #     pcl_to_prim_loss = (pcl_to_prim_loss.transpose(-1,-2) * weights).sum(-1).mean()

        # Scale
        scale_loss = self.scale_mse(out_dict['scale'], gt_scale, mask)
        loss += self.w_scale * scale_loss
        loss_dict['param_scale'] = scale_loss.item()
        
        # Shape
        shape_loss = self.shape_mse(out_dict['shape'], gt_shape, mask)
        loss += self.w_shape * shape_loss
        loss_dict['param_shape'] = shape_loss.item()
        
        # Translation
        trans_loss = self.trans_mse(out_dict['trans'], gt_trans, mask)
        loss += self.w_trans * trans_loss
        loss_dict['param_trans'] = trans_loss.item()
        
        # Rotation
        rot_loss = self.rot_mse(rotation, gt_rotate, mask)
        loss += self.w_rot * rot_loss
        loss_dict['param_rot'] = rot_loss.item()

        # Tapering
        if 'tapering' in out_dict:
            tapering_loss = self.tapering_mse(out_dict['tapering'], gt_tapering, mask)
            loss += self.w_tapering * tapering_loss
            loss_dict['param_tapering'] = tapering_loss.item()

        # Bending
        if 'bending_k' in out_dict and 'bending_a' in out_dict:
            bend_loss = self.bending_mse(out_dict['bending_k'], out_dict['bending_a'], gt_bending, mask)
            loss += self.w_bending * bend_loss
            loss_dict['param_bending'] = bend_loss.item()
        
        # stop trainer from complaining about unused parameters (TODO should we use it?)
        sparsity_loss = self.get_sparsity_loss(out_dict['assign_matrix'])
        loss += self.w_sps * sparsity_loss
        
        loss += self.compute_common_losses(out_dict, loss_dict)
        
        loss_dict['all'] = loss.item()
        return loss, loss_dict

class LossGeometric(ParamLossGeometric):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.geometric_cd = getattr(cfg, 'geometric_cd', False)
        self.sampler = EqualDistanceSamplerSQ(n_samples=cfg.n_samples, D_eta=0.05, D_omega=0.05)
        self.n_samples_cost = getattr(cfg, 'n_samples_cost', 64)
        
        self.c_geometric = getattr(cfg, 'c_geometric', 1.0)
        self.w_geometric = getattr(cfg, 'w_geometric', 1.0)
        self.w_geometric_f = getattr(cfg, 'w_geometric_f', 1.0)
        self.c_z_align = getattr(cfg, 'c_z_align', 0.0)
        
        
    def sample(self, scale, shape):
        etas, omegas = self.sampler.sample_on_batch(
            scale.detach().cpu().numpy(),
            shape.detach().cpu().numpy()
        )
        # Make sure we don't get nan for gradients
        etas[etas == 0] += 1e-6
        omegas[omegas == 0] += 1e-6

        # Move to tensors
        etas = scale.new_tensor(etas)
        omegas = scale.new_tensor(omegas)
        return etas, omegas 

    def forward(self, batch, out_dict):
        gt_scale = batch['gt_scale'].cuda().float()
        gt_shape = batch['gt_shape'].cuda().float()
        gt_trans = batch['gt_trans'].cuda().float()
        gt_rotate = batch['gt_rotate'].cuda().float()
        gt_rotate_q = batch['gt_rotate_q'].cuda().float()
        gt_exist = batch['gt_exist'].cuda().float()
        gt_tapering = batch['gt_tapering'].cuda().float()
        gt_bending = batch['gt_bending'].cuda().float()
        
        B, P = gt_scale.shape[:2]
        
        # Sample and compute points for GT
        gt_sq_etas, gt_sq_omegas = self.sample(gt_scale, gt_shape)
        gt_pts = parametric_to_points_extended(
            gt_trans,
            gt_rotate,
            gt_scale,
            gt_shape,
            gt_tapering,
            gt_bending[..., 0::2],
            gt_bending[..., 1::2],
            gt_sq_etas,
            gt_sq_omegas
        )

        # Sample and compute points for Pred
        pred_sq_etas, pred_sq_omegas = self.sample(out_dict['scale'], out_dict['shape'])
        pred_pts = parametric_to_points_extended(
            out_dict['trans'], 
            out_dict['rotate'], 
            out_dict['scale'],
            out_dict['shape'],
            out_dict['tapering'],
            out_dict['bending_k'],
            out_dict['bending_a'],
            pred_sq_etas,
            pred_sq_omegas
        )

        # Hungarian matching
        with torch.no_grad():
            device = gt_scale.device
            C = torch.zeros((B, P, P), device=device)

            if self.c_geometric > 0:
                # For cost computation downsample points
                n_pts = gt_pts.shape[-2]
                gt_pts_ds = gt_pts
                pred_pts_ds = pred_pts
                if n_pts > 64:
                    idx = torch.linspace(0, n_pts - 1, steps=self.n_samples_cost, dtype=torch.int, device=gt_pts.device)
                    gt_pts_ds = gt_pts.index_select(-2, idx)
                    pred_pts_ds = pred_pts.index_select(-2, idx)
                
                pred_pts_ds_ext = pred_pts_ds.unsqueeze(2).expand(-1, -1, P, -1, -1) # [B, P_pred, P_gt, N_ds, 3]
                cost_geometric = self.c_geometric * self.geometric_error_cd(pred_pts_ds_ext, gt_pts_ds)
                C += cost_geometric

            if self.c_z_align > 0:
                z_pred = out_dict['rotate'][..., 2]  # (B, P_pred, 3)
                z_gt = gt_rotate[..., 2]            # (B, P_gt, 3)
                # pairwise dot product -> (B, P_pred, P_gt)
                dot = torch.einsum('bpi,bqi->bpq', z_pred, z_gt)
                cost_z = self.c_z_align * (1.0 - torch.abs(dot))
                C += cost_z

            if self.c_exist > 0:
                cost_exist = self.c_exist * torch.abs(out_dict['exist'].view(B, P, 1) - gt_exist.view(B, 1, P))
                C += cost_exist

            C_np = C.cpu().numpy()
            
            indices = []
            for b in range(B):
                row_ind, col_ind = linear_sum_assignment(C_np[b])
                indices.append(col_ind)
                
            indices = torch.tensor(np.array(indices), device=gt_scale.device)
            batch_idx = torch.arange(B, device=gt_scale.device).unsqueeze(1).expand(B, P)
            gt_pts = gt_pts[batch_idx, indices]
            gt_scale = gt_scale[batch_idx, indices]
            gt_shape = gt_shape[batch_idx, indices]
            gt_trans = gt_trans[batch_idx, indices]
            gt_rotate = gt_rotate[batch_idx, indices]
            gt_exist = gt_exist[batch_idx, indices]
            gt_tapering = gt_tapering[batch_idx, indices]
            gt_bending = gt_bending[batch_idx, indices]
        
        loss = 0
        loss_dict = {}            
        mask = (gt_exist > 0.5).float()

        # Existance (binary cross-entropy)
        exist_loss = nn.BCELoss()(out_dict['exist'], mask)
        loss += self.w_exist * exist_loss
        loss_dict['param_exist'] = exist_loss.item()
        
        # Geometric
        geometric_loss = self.geometric_error_cd(pred_pts, gt_pts, mask)
        loss += self.w_geometric * geometric_loss
        loss_dict['param_geometric'] = geometric_loss.item()
        
        with torch.no_grad():
            pred_xy = out_dict['scale'][..., 0:2]                 # (B, P, 2)
            gt_xy = gt_scale[..., 0:2]                            # (B, P, 2)
            gt_xy_swapped = gt_xy[..., [1, 0]]
            l1_orig = torch.abs(pred_xy - gt_xy).sum(dim=-1)      # (B, P)
            l1_swap = torch.abs(pred_xy - gt_xy_swapped).sum(dim=-1)
            swap_mask = (l1_swap < l1_orig)                       # (B, P)

            # build matched gt_scale (swap x,y where needed)
            gt_scale_matched = gt_scale.clone()
            new_xy = torch.where(swap_mask.unsqueeze(-1), gt_xy_swapped, gt_xy)
            gt_scale_matched[..., 0:2] = new_xy
        
        pred_pts_forced = parametric_to_points_extended(
            gt_trans, 
            out_dict['rotate'], 
            gt_scale,
            gt_shape,
            out_dict['tapering'],
            out_dict['bending_k'],
            out_dict['bending_a'],
            gt_sq_etas,
            gt_sq_omegas
        )
        geometric_loss_forced = self.geometric_error_cd(pred_pts_forced, gt_pts, mask)
        loss += self.w_geometric_f * geometric_loss_forced
        loss_dict['param_geometric_forced'] = geometric_loss_forced.item()
        
        # stop trainer from complaining about unused parameters (TODO fix this)
        loss += 0 * out_dict['assign_matrix'].mean()
        loss_dict['all'] = loss.item()
        return loss, loss_dict

class IoULoss(BaseLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.truncation = getattr(cfg, 'truncation', 0.05)
        self.w_iou = getattr(cfg, 'w_iou', 1.0)
        self.w_sdf = getattr(cfg, 'w_sdf', 1.0)
        self.w_bbox = getattr(cfg, 'w_bbox', 1.0)
        self.w_overlap = getattr(cfg, 'w_overlap', 1.0)
    
    def poles(self, scale, rotate, trans):
        B, N, _ = scale.shape
        local = torch.zeros(B, N, 6, 3, device=scale.device)
        local[:,:,0,0] =  scale[:,:,0]  # +x
        local[:,:,1,0] = -scale[:,:,0]  # -x
        local[:,:,2,1] =  scale[:,:,1]  # +y
        local[:,:,3,1] = -scale[:,:,1]  # -y
        local[:,:,4,2] =  scale[:,:,2]  # +z
        local[:,:,5,2] = -scale[:,:,2]  # -z

        # rotate
        poles_world = torch.matmul(
            rotate.unsqueeze(2),     # (B,N,1,3,3)
            local.unsqueeze(-1) # (B,N,6,3,1)
        ).squeeze(-1)

        # translate
        poles_world = poles_world + trans.unsqueeze(2)
        return poles_world

    def forward(self, batch, out_dict):
        pc, normals = batch['points'].cuda().float(), batch['normals'].cuda().float()
        points_iou, gt_occ = batch['points_iou'].cuda().float(), batch['occupancies'].cuda().bool()

        # pc: (B, M_surf, 3)
        loss = 0
        loss_dict = {}
        
        # 1. SDF on Surface Points
        all_sdfs_surf = sdf_batch(pc, out_dict) #(B, N, M_surf)
        exist_probs = out_dict['exist'].reshape(all_sdfs_surf.shape[0], all_sdfs_surf.shape[1]) # (B, N)
        exist_weights = exist_probs.unsqueeze(-1) # (B, N, 1)
        assign_probs = out_dict['assign_matrix'].transpose(1, 2) #(B, N, M_surf)
        
        temperature = 1e-2
        sdfs_surf_trunc = (torch.sigmoid(all_sdfs_surf / temperature) - 0.5) * 2 * self.truncation
        relu_sdfs_trunc = torch.abs(torch.nn.LeakyReLU()(sdfs_surf_trunc)) 
        weighted_relu_sdf = (assign_probs * relu_sdfs_trunc).sum(dim=1) #(B, M_surf)
        L_sdf = 16 * weighted_relu_sdf.mean()

        loss += self.w_sdf * L_sdf
        loss_dict['sdf'] = L_sdf.item()
        
        # 2. BBox Loss
        bbox_min = torch.min(pc, dim=1).values.unsqueeze(1).unsqueeze(2) # (B, 1, 1, 3)
        bbox_max = torch.max(pc, dim=1).values.unsqueeze(1).unsqueeze(2)
        margin = (bbox_max - bbox_min) * .025
        bbox_min -= margin
        bbox_max += margin
        
        poles = self.poles(out_dict['scale'], out_dict['rotate'], out_dict['trans']) # (B, N, 6, 3)
        violation = torch.relu(bbox_min - poles) + torch.relu(poles - bbox_max) # (B, N, 6, 3)
        dist = torch.linalg.norm(violation, dim=-1) # (B, N, 6)
        L_bbox = (dist * exist_weights).sum(dim=2).mean() # Mean over batch
        
        loss += self.w_bbox * L_bbox
        loss_dict['bbox'] = L_bbox.item()

        # 3. IoU Loss
        all_sdfs_iou = sdf_batch(points_iou, out_dict) # (B, N, M_iou)
        
        temperature_iou = 1e-3
        prob_prim = torch.sigmoid(-all_sdfs_iou / temperature_iou)
        p_occupied = exist_weights * prob_prim # (B, N, M_iou)
        
        # Union probability: P(union) = 1 - prod(1 - p_i)
        # implementation: 1 - exp(sum(log(1 - p_i)))
        # clamp p_occupied to max 1-eps to avoid log(0)
        p_occupied = torch.clamp(p_occupied, min=0.0, max=0.9999)
        pred_occ = 1.0 - torch.exp(torch.sum(torch.log(1.0 - p_occupied), dim=1)) # (B, M_iou)
        intersection = (pred_occ * gt_occ).sum(dim=1)
        union = (pred_occ + gt_occ - pred_occ * gt_occ).sum(dim=1)
        iou = (intersection + 1e-6) / (torch.clamp(union, min=1.0) + 1e-6)
        L_iou = -torch.log(iou).mean()
        
        loss += self.w_iou * L_iou
        loss_dict['iou'] = iou.mean().item()
        loss_dict['iou_loss'] = L_iou.item()

        # 4. Overlap/Regularization Loss
        temperature_overlap = 1e-3
        indicator = torch.sigmoid(-(all_sdfs_iou + self.truncation) / temperature_overlap)
        weighted_indicator = (exist_weights > 0.5) * indicator # (B, N, M)
        overlap_count = torch.relu(torch.sum(weighted_indicator, dim=1) - 1) # (B, M)
        L_overlap = overlap_count.mean()
        
        loss += self.w_overlap * L_overlap
        loss_dict['overlap'] = L_overlap.item()

        loss += self.compute_common_losses(out_dict, loss_dict)
        loss_dict['all'] = loss.item()
        return loss, loss_dict


class Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        loss_type = getattr(cfg, 'type', 'original')
        print(f"Initializing Loss: {loss_type}")
        
        if loss_type == 'original':
            self.impl = SuperDecLoss(cfg)
        elif loss_type == 'param':
            self.impl = ParamLoss(cfg)
        elif loss_type == 'param_geom':
            self.impl = ParamLossGeometric(cfg)
        elif loss_type == 'geom':
            self.impl = LossGeometric(cfg)
        elif loss_type == 'iou':
            self.impl = IoULoss(cfg)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, batch, out_dict):
        return self.impl(batch, out_dict)

