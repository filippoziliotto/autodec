import torch.nn as nn
import torch.nn.functional as F
import torch

class SuperDecHead(nn.Module):
    """Head for Superquadrics Prediction"""
    
    def __init__(self, emb_dims, ctx):
        super(SuperDecHead, self).__init__()
        self.emb_dims = emb_dims
        # small MLP heads: one hidden layer
        hidden_dim = max(emb_dims // 2, 64)
        self.scale_head = nn.Sequential(
            nn.Linear(emb_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        self.shape_head = nn.Sequential(
            nn.Linear(emb_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.rotation6d = getattr(ctx, 'rotation6d', False)
        rot_out = 6 if self.rotation6d else 4
        self.rot_head = nn.Sequential(
            nn.Linear(emb_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rot_out)
        )
        self.t_head = nn.Sequential(
            nn.Linear(emb_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        self.exist_head = nn.Sequential(
            nn.Linear(emb_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.extended = getattr(ctx, 'extended', False)
        if self.extended:
            self.tapering_head = nn.Sequential(
                nn.Linear(emb_dims, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            )
            self.bending_k_head = nn.Sequential(
                nn.Linear(emb_dims, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3)
            )
            self.bending_a_head = nn.Sequential(
                nn.Linear(emb_dims, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3)
            )
            # Initialize tapering and bending heads' final linear layers: tapering and bending_a to 0,
            # and initialize bending_k bias to -6 so sigmoid produces a small
            # initial bending value (sigmoid(-6) ~ 0.0025 -> near zero)
            for head in (self.tapering_head, self.bending_a_head):
                last_lin = [l for l in head if isinstance(l, nn.Linear)][-1]
                nn.init.zeros_(last_lin.weight)
                nn.init.zeros_(last_lin.bias)
            last_lin_k = [l for l in self.bending_k_head if isinstance(l, nn.Linear)][-1]
            nn.init.zeros_(last_lin_k.weight)
            nn.init.constant_(last_lin_k.bias, -6.0)


    def forward(self, x):
        scale_pre_activation = self.scale_head(x)
        scale = self.scale_activation(scale_pre_activation)
        
        shape_before_activation = self.shape_head(x)
        shape = self.shape_activation(shape_before_activation)

        if self.rotation6d:
            rotation = self.rot6d2mat(self.rot_head(x))
        else:
            q = F.normalize(self.rot_head(x), dim=-1, p=2)
            rotation = self.quat2mat(q)            

        translation = self.t_head(x)

        exist = self.exist_activation(self.exist_head(x))

        out_dict = {"scale": scale, "shape": shape, "rotate": rotation, "trans": translation, "exist": exist}
        if self.extended:
            tapering = self.tapering_activation(self.tapering_head(x))
            bending_k = self.bending_k_activation(self.bending_k_head(x), scale)
            bending_a = self.bending_a_activation(self.bending_a_head(x))
            out_dict.update({"tapering": tapering, "bending_k": bending_k, "bending_a": bending_a})
        return out_dict
        
    @staticmethod  
    def quat2mat(quat):
        """Normalize the quaternion and convert it to rotation matrix"""
        B = quat.shape[0]
        N = quat.shape[1]
        quat = F.normalize(quat, dim=2)
        quat = quat.contiguous().view(-1,4)
        w, x, y, z = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z
        rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                            2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                            2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, N, 3, 3)
        rotMat = rotMat.view(B,N,3,3)
        return rotMat
    
    @staticmethod
    def rot6d2mat(rot6d):
        """Convert 6D rotation representation to 3x3 rotation matrix.
        Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2018
        """
        B, N, _ = rot6d.shape
        rot6d = rot6d.view(-1, 6)
        
        a1 = rot6d[:, :3]
        a2 = rot6d[:, 3:]
        
        b1 = F.normalize(a1, dim=-1)
        b2 = F.normalize(a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        
        rotMat = torch.stack((b1, b2, b3), dim=-1)
        return rotMat.view(B, N, 3, 3)
    
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
        max_scale = tau * torch.logsumexp(scale/tau, keepdim=True, dim=-1)
        return (torch.sigmoid(x) * 0.95) / max_scale 
    
    @staticmethod 
    def bending_a_activation(x):
        return torch.sigmoid(x) * 2 * torch.pi