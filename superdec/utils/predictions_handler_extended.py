import numpy as np
import torch
from typing import Dict
import trimesh

from superdec.utils.visualizations import generate_ncolors
from superdec.utils.transforms import transform_to_primitive_frame

def extend_dict(outdict):
    if isinstance(outdict['scale'], torch.Tensor): cls = torch
    else: cls = np
    
    B, N, _ = outdict['scale'].shape
    if 'rescale' not in outdict:
        outdict['rescale'] = cls.ones((B))
    if 'recenter' not in outdict:
        outdict['recenter'] = cls.zeros((B, 3))
    if 'tapering' not in outdict: 
        outdict['tapering'] = cls.zeros((B, N, 2))
    if 'bending' not in outdict or outdict['bending'].shape[-1] != 6: 
        outdict['bending'] = cls.zeros((B, N, 6))
    return outdict

class PointCloud:
    def __init__(self, points, colors):
        self.points = points
        self.colors = colors

class PredictionHandler:
    def __init__(self, predictions: Dict[str, np.ndarray]):
        self.names = predictions['names']
        self.pc = predictions['pc']                # [B, N, 3]
        self.assign_matrix = predictions['assign_matrix']  # [B, N, P]
        self.scale = predictions['scale']               # [B, P, 3]
        self.rotation = predictions['rotation']         # [B, P, 3, 3]
        self.translation = predictions['translation']    # [B, P, 3]
        self.exponents = predictions['exponents']       # [B, P, 2]
        self.exist = predictions['exist']               # [B, P]
        self.colors = generate_ncolors(self.translation.shape[1])  # Generate colors for each object

        # extension
        self.tapering = predictions['tapering']
        self.bending = predictions['bending']
        self.rescale = predictions['rescale']
        self.recenter = predictions['recenter']
    
    def save_npz(self, filepath):
        """Save accumulated outputs to compressed npz file."""
        np.savez_compressed(
            filepath,
            names=np.array(self.names),
            pc=np.stack(self.pc),
            assign_matrix=np.stack(self.assign_matrix),
            scale=np.stack(self.scale),
            rotation=np.stack(self.rotation),
            translation=np.stack(self.translation),
            exponents=np.stack(self.exponents),
            exist=np.stack(self.exist),
            tapering=np.stack(self.tapering),
            bending=np.stack(self.bending),
            rescale=np.stack(self.rescale),
            recenter=np.stack(self.recenter),
        )
    
    @classmethod
    def from_npz(cls, path: str):
        data = np.load(path, allow_pickle=True)
        data_dict = {key: data[key] for key in data.files}
        return cls(extend_dict(data_dict))

    @classmethod # TODO test!!
    def from_outdict(cls, outdict, pcs, names):
        extend_dict(outdict)
        predictions = {
            'names': names, 
            'pc': pcs.cpu().numpy(), 
            'assign_matrix': outdict['assign_matrix'].cpu().numpy(), 
            'scale': outdict['scale'].cpu().numpy(), 
            'rotation': outdict['rotate'].cpu().numpy(),
            'translation': outdict['trans'].cpu().numpy(), 
            'exponents': outdict['shape'].cpu().numpy(), 
            'exist': outdict['exist'].cpu().numpy(),
            'tapering': outdict['tapering'].cpu().numpy(),
            'bending': outdict['bending'].cpu().numpy(),
            'rescale': outdict['rescale'].cpu().numpy(), 
            'recenter': outdict['recenter'].cpu().numpy(), 
        }
        return cls(predictions)

    def append_outdict(self, outdict, pcs, names):
        extend_dict(outdict)
        self.names = np.concatenate((self.names, names), axis=0)
        self.pc = np.concatenate((self.pc, pcs.cpu().numpy()), axis=0)
        self.assign_matrix = np.concatenate((self.assign_matrix, outdict['assign_matrix'].cpu().numpy()), axis=0)
        self.scale = np.concatenate((self.scale, outdict['scale'].cpu().numpy()), axis=0)
        self.rotation = np.concatenate((self.rotation, outdict['rotate'].cpu().numpy()), axis=0)
        self.translation = np.concatenate((self.translation, outdict['trans'].cpu().numpy()), axis=0)
        self.exponents = np.concatenate((self.exponents, outdict['shape'].cpu().numpy()), axis=0)
        self.exist = np.concatenate((self.exist, outdict['exist'].cpu().numpy()), axis=0)
        self.tapering = np.concatenate((self.tapering, outdict['tapering'].cpu().numpy()), axis=0)
        self.bending = np.concatenate((self.bending, outdict['bending'].cpu().numpy()), axis=0)
        self.rescale = np.concatenate((self.rescale, outdict['rescale'].cpu().numpy()), axis=0)
        self.recenter = np.concatenate((self.recenter, outdict['recenter'].cpu().numpy()), axis=0)

    def get_segmented_pc(self, index):
        if isinstance(self.assign_matrix, torch.Tensor):
            assign_matrix = assign_matrix.cpu().numpy()[index]
        else:
            assign_matrix = self.assign_matrix[index]
        P = assign_matrix.shape[1]
        segmentation = np.argmax(assign_matrix, axis=1)
        colors = generate_ncolors(P)
        colored_pc = colors[segmentation]

        points = self.pc[index]
        colors = colored_pc / 255.0
        pc_obj = PointCloud(points=points, colors=colors)
        return pc_obj
    
    def get_segmented_pcs(self):
        pcs = []
        B = self.scale.shape[0]
        for b in range(B):
            pc = self.get_segmented_pc(b)
            pcs.append(pc)
        return pcs

    def get_meshes(self, resolution: int = 100, colors=True):
        meshes = []
        B = self.scale.shape[0]
        for b in range(B):
            try:
                mesh = self.get_mesh(b, resolution, colors)
                meshes.append(mesh)
            except Exception as e:
                print(f"Error generating mesh for index {b}: {e}")
                meshes.append(None)
        return meshes

    def get_mesh(self, index, resolution: int = 100, colors=True):
        P = self.scale.shape[1]

        vertices = []
        faces = []
        if colors:
            v_colors = []
            f_colors = []
        os_vertices = 0
        for p in range(P):
            if self.exist[index, p] > 0.5:
                mesh = self._superquadric_mesh(
                    self.scale[index, p], self.exponents[index, p],
                    self.rotation[index, p], self.translation[index, p], self.tapering[index, p], self.bending[index, p], resolution
                )
                
                vertices_cur, faces_cur = mesh

                vertices.append(vertices_cur)
                faces.append(faces_cur + os_vertices) 

                if colors:
                    cur_color = self.colors[p]
                    v_colors.append(np.ones((vertices_cur.shape[0],3)) * cur_color) 
                    f_colors.append(np.ones((faces_cur.shape[0],3)) * cur_color)

                os_vertices += len(vertices_cur)
        vertices = np.concatenate(vertices)
        faces = np.concatenate(faces)
        if colors:
            v_colors = np.concatenate(v_colors)/255.0
            f_colors = np.concatenate(f_colors)/255.0
            mesh = trimesh.Trimesh(vertices, faces, face_colors=f_colors, vertex_colors=v_colors)
        else:
            mesh = trimesh.Trimesh(vertices, faces)
                
        return mesh

    def apply_bending_axis(self, x, y, z, val_kb, val_alpha, axis):
        if np.abs(val_kb) < 1e-3: return x, y, z
        
        if axis == 'z':
            u, v_coord, w = x, y, z
        elif axis == 'x':
            u, v_coord, w = y, z, x
        elif axis == 'y':
            u, v_coord, w = z, x, y
        
        # Precompute constants
        inv_kb = 1.0 / val_kb
        sin_alpha = np.sin(val_alpha)
        cos_alpha = np.cos(val_alpha)
        
        beta = np.arctan2(v_coord, u)
        r = np.sqrt(u**2 + v_coord**2) * np.cos(val_alpha - beta)
        gamma = w * val_kb
        rho = inv_kb - r        
        R = inv_kb - rho * np.cos(gamma)

        expr = (R - r)
        u = u + expr * cos_alpha
        v_coord = v_coord + expr * sin_alpha
        w = rho * np.sin(gamma)
        
        if axis == 'z':
            return u, v_coord, w
        elif axis == 'x':
            return w, u, v_coord
        elif axis == 'y':
            return v_coord, w, u

    def _superquadric_mesh(self, scale, exponents, rotation, translation, tapering, bending, N):
        def f(o, m):
            sin_o = np.sin(o)
            return np.sign(sin_o) * np.abs(sin_o)**m
        def g(o, m):
            cos_o = np.cos(o)
            return np.sign(cos_o) * np.abs(cos_o)**m

        u = np.linspace(-np.pi, np.pi, N, endpoint=True)
        v = np.linspace(-np.pi/2.0, np.pi/2.0, N, endpoint=True)
        u = np.tile(u, N)
        v = (np.repeat(v, N))
        if np.linalg.det(rotation) < 0:
            u = u[::-1]

        x = scale[0] * g(v, exponents[0]) * g(u, exponents[1])
        y = scale[1] * g(v, exponents[0]) * f(u, exponents[1])
        z = scale[2] * f(v, exponents[0])
        # Set poles to zero to account for numerical instabilities in f and g due to ** operator
        x[:N] = 0.0
        x[-N:] = 0.0

        kx, ky = tapering
        z_norm = z / scale[2]
        fx = kx * z_norm + 1
        fy = ky * z_norm + 1
        x = x*fx
        y = y*fy

        # Apply bending transformation
        x, y, z = self.apply_bending_axis(x, y, z, bending[4], bending[5], 'y')
        x, y, z = self.apply_bending_axis(x, y, z, bending[2], bending[3], 'x')
        x, y, z = self.apply_bending_axis(x, y, z, bending[0], bending[1], 'z')

        vertices = np.stack([x, y, z], axis=1) # Faster than concatenate expand_dims
        vertices = (rotation @ vertices.T).T + translation  

        # Create a grid of indices for the top-left corner of each quad
        i = np.arange(N - 1)
        j = np.arange(N - 1)
        J, I = np.meshgrid(j, i)
        
        # Indices for the quad vertices
        # p1: (i, j)     -> I*N + J
        # p2: (i, j+1)   -> I*N + (J+1)
        # p3: (i+1, j)   -> (I+1)*N + J
        # p4: (i+1, j+1) -> (I+1)*N + (J+1)
        p1 = I * N + J
        p2 = I * N + (J + 1)
        p3 = (I + 1) * N + J
        p4 = (I + 1) * N + (J + 1)
        t1 = np.stack([p1, p2, p3], axis=-1).reshape(-1, 3)
        t2 = np.stack([p3, p2, p4], axis=-1).reshape(-1, 3)        
        main_body_triangles = np.concatenate([t1, t2], axis=0)

        # Seam connection (wrapping u around): Connect last column to first column
        # Last vertex in row i: i*N + (N-1)
        # First vertex in row i: i*N
        # Last vertex in row i+1: (i+1)*N + (N-1)
        # First vertex in row i+1: (i+1)*N
        i_seam = np.arange(N - 1)
        p1_s = i_seam * N + (N - 1)
        p2_s = i_seam * N
        p3_s = (i_seam + 1) * N + (N - 1)
        p4_s = (i_seam + 1) * N
        t1_s = np.stack([p1_s, p2_s, p3_s], axis=-1)
        t2_s = np.stack([p3_s, p2_s, p4_s], axis=-1)
        seam_triangles = np.concatenate([t1_s, t2_s], axis=0)
        
        # Pole caps
        cap_triangles = np.array([
            [(N-1)*N+(N-1), (N-1)*N, (N-1)],
            [(N-1), (N-1)*N, 0]
        ])

        triangles = np.concatenate([main_body_triangles, seam_triangles, cap_triangles], axis=0)
        return np.array(vertices), np.array(triangles)
