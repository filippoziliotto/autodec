import os
from glob import glob

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.nn import fps

from superdec.utils.transforms import mat2quat
from superdec.data.transform import RotateAroundAxis3d, Scale3d, RandomMove3d, Compose, rotate_around_axis
from superdec.data.transform_occlusions import BackFaceCulling, RandomOcclusion, HRPOcclusion

SHAPENET_CATEGORIES = {
    "04379243": "table", "02958343": "car", "03001627": "chair", "02691156": "airplane",
    "04256520": "sofa", "04090263": "rifle", "03636649": "lamp", "03691459": "loudspeaker",
    "02933112": "cabinet", "03211117": "display", "04401088": "telephone", "02828884": "bench",
    "04530566": "watercraft"
}

def normalize_points(points):
    translation = points.mean(0)
    points = points - translation
    scale = 2 * np.max(np.abs(points))
    points = points / scale
    return points, translation, scale

def denormalize_points(points, translation, scale, z_up=False):
    scale = scale[:,None,None]
    translation = translation[:, None, :]
    points = points * scale + translation
    if z_up:
        points = rotate_around_axis(points.cpu().numpy(), axis=(1,0,0), angle = np.pi/2, center_point=np.zeros(3))
        points = torch.from_numpy(points)
    return points

def denormalize_outdict(outdict, translation, scale, z_up=False):
    scale = scale[:,None,None]
    translation = translation[:, None, :]
    outdict['scale'] = outdict['scale'] * scale
    outdict['trans'] = outdict['trans'] * scale + translation 
    if z_up:
        # transform sq by updating translation and rotation
        outdict['trans'] = torch.tensor(rotate_around_axis(outdict['trans'].cpu().numpy(), axis=(1,0,0), angle = np.pi/2, center_point=np.zeros(3)))
        outdict['rotate'] = outdict['rotate'].cpu().numpy()
        rot_x_90 = np.array([[1,0,0],[0,0,-1],[0,1,0]])
        for i in range(outdict['rotate'].shape[0]):
            outdict['rotate'][i] = rot_x_90 @ outdict['rotate'][i]
        outdict['rotate'] = torch.from_numpy(outdict['rotate'])
        
    return outdict

def get_transforms(split: str, cfg):
    if split != 'train' or 'trainer' not in cfg or not cfg.trainer.augmentations:
        return None

    return Compose([
        Scale3d(),
        RotateAroundAxis3d(rotation_limit=np.pi / 24, axis=(0, 0, 1)),
        RotateAroundAxis3d(rotation_limit=np.pi / 24, axis=(1, 0, 0)),
        RotateAroundAxis3d(rotation_limit=np.pi, axis=(0, 1, 0)),
        RandomMove3d(
            x_min=-0.1, x_max=0.1,
            y_min=-0.05, y_max=0.05,
            z_min=-0.1, z_max=0.1
        ),
    ])

def get_occlusion_transforms(split: str, cfg):
    if split != 'train' or 'trainer' not in cfg or not cfg.trainer.occlusions:
        return None
    
    return Compose([
        BackFaceCulling(p=0.5),
        RandomOcclusion(p=0.5),
        # HRPOcclusion(p=0.5), # kind of slow
    ])

class ScenesDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.gt = cfg.scenes_dataset.gt
        gt_suffix = "_gt" if self.gt else ""
        self.subfolder = f"pc{gt_suffix}"
        self.path = os.path.join(cfg.scenes_dataset.path)
        self.split = cfg.scenes_dataset.split
        self.z_up = True
        self.fps = cfg.scenes_dataset.fps if 'fps' in cfg.scenes_dataset else False
        self.scenes = self._load_scenes()
        self.models = self._gather_models()

    def _load_scenes(self):
        split_txt = f'{self.split}.txt'
        if not os.path.exists(os.path.join(self.path, split_txt)):
            print('Split %s does not exist.' % (split_txt))
        with open(os.path.join(self.path, split_txt), 'r') as f:
            scenes_names = f.read().split('\n')
        return scenes_names

    def _gather_models(self):
        models = []
        for s in self.scenes:
            try:
                scene_path = os.path.join(self.path, s, self.subfolder)
                model_ids = [os.path.splitext(f)[0] for f in os.listdir(scene_path) if f.endswith(".npz")]
                models.extend([{'scene': s, 'model_id': m} for m in model_ids])
            except FileNotFoundError:
                continue
        return models

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        model = self.models[idx]
        model_path = os.path.join(self.path, model['scene'], self.subfolder, f"{model['model_id']}.npz")
        
        pc_data = np.load(model_path)
        points_tmp = pc_data['points']

        n_points = points_tmp.shape[0]

        if n_points >= 4096:
            if self.fps:
                points_tensor = torch.from_numpy(points_tmp)
                ratio = 4096 / n_points
                indices = fps(points_tensor, ratio=ratio)
                indices = indices[:4096]
                points = points_tmp[indices.numpy()]
            else:
                idxs = np.random.choice(n_points, 4096, replace=False)
                points = points_tmp[idxs]
        else:
            idxs = np.random.choice(n_points, 4096)
            points = points_tmp[idxs]

        if self.z_up:
            points = rotate_around_axis(points, axis=(1,0,0), angle = -np.pi/2, center_point=np.zeros(3))

        points, translation, scale  = normalize_points(points)

        return {
            "points": torch.from_numpy(points),
            "translation": torch.from_numpy(translation),
            "scale": scale,
            "z_up": self.z_up,
            "point_num": points.shape[0],
            "model_id": model
        }

    def name(self):
        return 'ScenesDataset'

class Scene(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.gt = cfg.scene.gt
        gt_suffix = "_gt" if self.gt else ""
        self.path = os.path.join(cfg.scene.path, cfg.scene.name, f"pc{gt_suffix}")
        self.z_up = cfg.scene.z_up 
        self._gather_models()

    def _gather_models(self):
        self.models = [os.path.splitext(f)[0] for f in os.listdir(self.path) if f.endswith(".npz")]

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        model = self.models[idx]
        
        pc_data = np.load(os.path.join(self.path, f"{model}.npz"))
        points_tmp = pc_data['points']

        n_points = points_tmp.shape[0]

        if n_points >= 4096:
            idxs = np.random.choice(n_points, 4096, replace=False)
            points = points_tmp[idxs]
        else:
            idxs = np.random.choice(n_points, 4096)
            points = points_tmp[idxs]

        if self.z_up:
            points = rotate_around_axis(points, axis=(1,0,0), angle = -np.pi/2, center_point=np.zeros(3))

        points, translation, scale  = normalize_points(points)

        return {
            "points": torch.from_numpy(points),
            "translation": torch.from_numpy(translation),
            "scale": scale,
            "z_up": self.z_up,
            "point_num": points.shape[0],
            "model_id": model
        }

    def name(self):
        return 'Scene'


class ObjectDataset(Dataset):
    """Shared helpers for ShapeNet/ABO pointcloud-style datasets."""
    def __init__(self, split: str, cfg):
        self.normalize = getattr(cfg, 'normalize', False)
        self.use_fps = getattr(cfg, 'use_fps', False)
        self.load_occupancy = (split == 'val') or getattr(cfg, 'load_occupancy', False)
        
        self.valid_models = None
        self.gt_params_path = getattr(cfg, f'gt_{split}_path', None)
        self.gt_data, self.gt_mapping = {}, {}
        if self.gt_params_path is not None:
            self._load_gt_params()
            self.filter_cfg = getattr(cfg, 'filter', None)
            if self.filter_cfg is not None:
                self._load_filter()
    
    def _load_gt_params(self):
        if not os.path.exists(self.gt_params_path):
            print(f"Warning: GT params path {self.gt_params_path} not found.")
            return
        try:
            data = np.load(self.gt_params_path, allow_pickle=True)
            self.gt_data = {k: data[k] for k in data.files}
            names = self.gt_data['names']
            if names.ndim == 0: names = names.item()
            self.gt_mapping = {str(n): i for i, n in enumerate(names)}
            print(f"Loaded GT params from {self.gt_params_path} for {len(names)} models.")
        except Exception as e:
            print(f"Error loading GT params: {e}")
            self.gt_data = None

    def _load_filter(self):
        metrics_path = self.gt_params_path.replace('.npz', '_metrics.csv')
        if not os.path.exists(metrics_path):
            print(f"Warning: Metrics path {metrics_path} not found for filtering.")
            return
            
        try:
            metric = self.filter_cfg.get('metric')
            threshold = float(self.filter_cfg.get('threshold'))
            operator = self.filter_cfg.get('operator', '<=' if 'chamfer' in metric.lower() else '>=')
            
            df = pd.read_csv(metrics_path)
            if metric not in df.columns:
                print(f"Warning: Metric {metric} not found in {metrics_path}.")
                return
                
            total = len(df)            
            if operator == '<=':
                valid_df = df[df[metric] <= threshold]
            elif operator == '>=':
                valid_df = df[df[metric] >= threshold]
            elif operator == '<':
                valid_df = df[df[metric] < threshold]
            elif operator == '>':
                valid_df = df[df[metric] > threshold]
            else:
                print(f"Warning: Unknown operator {operator}.")
                return
                
            self.valid_models = set(valid_df['name'].astype(str))
            print(f"Filtered {total - len(self.valid_models)} models. {len(self.valid_models)} remaining based on {metric} {operator} {threshold}.")
        except Exception as e:
            print(f"Error loading metrics for filtering: {e}")

    def _load_pointcloud(self, model_path):
        # Load pointcloud and normals, prefer precomputed 4096 file on test
        if self.split == 'test' or self.use_fps:
            try:
                pc_data = np.load(os.path.join(model_path, "pointcloud_4096.npz"))
                points = pc_data["points"]
                normals = pc_data["normals"]
                return points, normals, pc_data
            except Exception:
                print(f"Error loading precomputed 4096 file for {model_path}")
                pass

        pc_data = np.load(os.path.join(model_path, "pointcloud.npz"))
        points = pc_data['points']
        normals = pc_data['normals']
        if self.transform_occlusions:
            t_data = self.transform_occlusions(points=points, normals=normals)
            points = t_data['points']
            normals = t_data['normals']

        n_points = points.shape[0]
        if self.split == 'test' or n_points >= 4096:
            idxs = np.random.choice(n_points, 4096, replace=False)
        else:
            idxs = np.random.choice(n_points, 4096)
        points = points[idxs]
        normals = normals[idxs]
        return points, normals, pc_data
    
    def _decompose_augmentation_matrix(self, res):
        aug_matrix = res['aug_matrix'].numpy()
        R_aug = aug_matrix[:3, :3]
        t_aug = aug_matrix[:3, 3]
        
        # Extract scale and normalize rotation matrix
        aug_scale = np.linalg.norm(R_aug, axis=1)
        R_aug_norm = R_aug / aug_scale[:, None]
        return R_aug_norm, t_aug, aug_scale

    def _add_occupancy_and_gt(self, res, model_id, model_path):
        # occupancy / points_iou
        if self.load_occupancy:
            try:
                points_iou_path = os.path.join(model_path, "points.npz")
                points_dict = np.load(points_iou_path)
                points_iou = points_dict['points']
                occ_tgt = points_dict['occupancies']
                if np.issubdtype(occ_tgt.dtype, np.uint8):
                    occ_tgt = np.unpackbits(occ_tgt)[:points_iou.shape[0]]

                if self.normalize:
                    translation_np = res['translation'].numpy() if isinstance(res['translation'], torch.Tensor) else res['translation']
                    scale_iou = res['scale']
                    points_iou = (points_iou - translation_np) / scale_iou

                R_aug, t_aug, s_aug = self._decompose_augmentation_matrix(res)
                points_iou = points_iou @ R_aug.T * s_aug + t_aug

                res.update({
                    'points_iou': torch.from_numpy(points_iou).float(),
                    'occupancies': torch.from_numpy(occ_tgt).bool()
                })
            except Exception:
                pass

        # GT params
        if str(model_id) in self.gt_mapping:
            idx = self.gt_mapping[str(model_id)]
            gt_scale = self.gt_data['scale'][idx]
            gt_shape = self.gt_data['exponents'][idx]
            gt_trans = self.gt_data['translation'][idx]
            gt_rotate = self.gt_data['rotation'][idx]
            gt_exist = self.gt_data['exist'][idx]
            gt_tapering = self.gt_data['tapering'][idx]
            gt_bending = self.gt_data['bending'][idx]

            translation_np = res['translation'].numpy() if isinstance(res['translation'], torch.Tensor) else res['translation']
            gt_trans = (gt_trans - translation_np) / res['scale']
            gt_scale = gt_scale / res['scale']

            R_aug, t_aug, s_aug = self._decompose_augmentation_matrix(res)
            gt_trans = gt_trans @ R_aug.T * s_aug + t_aug
            gt_rotate = R_aug @ gt_rotate
            gt_scale = gt_scale * s_aug

            rot_mat = torch.from_numpy(gt_rotate).float()
            res.update({
                'gt_scale': torch.from_numpy(gt_scale).float(),
                'gt_shape': torch.from_numpy(gt_shape).float(),
                'gt_trans': torch.from_numpy(gt_trans).float(),
                'gt_rotate': rot_mat,
                'gt_rotate_q': mat2quat(rot_mat.unsqueeze(0)).squeeze(0),
                'gt_exist': torch.from_numpy(gt_exist).float(),
                'gt_tapering': torch.from_numpy(gt_tapering).float(),
                'gt_bending': torch.from_numpy(gt_bending).float(),
            })
    
    def _get_model_path(self, idx):
        raise NotImplementedError(
            "Method _get_model_path is not implemented in class "
            + self.__class__.__name__
        )
    
    def __getitem__(self, idx):
        model_id, model_path = self._get_model_path(idx)
        points, normals, pc_data = self._load_pointcloud(model_path)

        if self.normalize:
            points, translation, scale  = normalize_points(points)
        else:
            translation = np.zeros(3)
            scale = 1.0

        if self.transform is not None:
            # Initialize accumulation matrix
            transform_info = {'matrix': np.eye(4)}
            t_data = self.transform(points=points, normals=normals, transform_info=transform_info)
            points = t_data['points']
            normals = t_data['normals']
            transform_matrix = t_data['transform_info']['matrix'] # 4x4
        else:
            transform_matrix = np.eye(4)

        res = {
            "points": torch.from_numpy(points),
            "normals": torch.from_numpy(normals),
            "translation": torch.from_numpy(translation),
            "scale": scale,
            "point_num": points.shape[0],
            "model_id": model_id,
            "idx": idx,
            "aug_matrix": torch.from_numpy(transform_matrix).float()
        }

        self._add_occupancy_and_gt(res, model_id, model_path)
        return res

class ShapeNet(ObjectDataset):
    def __init__(self, split: str, cfg):
        super().__init__(split, cfg.shapenet)
        self.split = split
        self.data_root = cfg.shapenet.path

        self.transform = get_transforms(split, cfg)
        self.transform_occlusions = get_occlusion_transforms(split, cfg)

        self.categories = self._load_categories(cfg.shapenet.categories)
        self.models = self._gather_models()

    def _load_categories(self, categories):
        if categories is None:
            return [d for d in os.listdir(self.data_root)
                    if os.path.isdir(os.path.join(self.data_root, d))]
        print(f"Categories for split '{self.split}': {', '.join(SHAPENET_CATEGORIES[c] for c in categories)}")
        return categories

    def _gather_models(self):
        models = []
        for c in self.categories:
            category_path = os.path.join(self.data_root, c)
            split_file = os.path.join(category_path, f'{self.split}.lst')
            if not os.path.exists(split_file):
                continue
            with open(split_file, 'r') as f:
                model_ids = [line.strip() for line in f if line.strip()]
            
            if self.valid_models is not None:
                model_ids = [m for m in model_ids if m in self.valid_models]
                
            models.extend([{'category': c, 'model_id': m} for m in model_ids])
        return models

    def __len__(self):
        return len(self.models)

    def _get_model_path(self, idx):
        model = self.models[idx]
        model_path = os.path.join(self.data_root, model['category'], model['model_id'])
        return model['model_id'], model_path
    
    def name(self):
        return 'ShapeNet'

class ABO(ObjectDataset):
    def __init__(self, split: str, cfg):
        super().__init__(split, cfg.abo)
        self.split = split
        self.data_root = cfg.abo.path

        self.transform = get_transforms(split, cfg)
        self.transform_occlusions = get_occlusion_transforms(split, cfg)

        self.models = self._gather_models()

    def _gather_models(self):
        if not os.path.exists(self.data_root):
            print(f"ABO data root not found: {self.data_root}")
            return []
        
        # List all subdirectories (ASINs)
        models = [d for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d))]
        models.sort() # Ensure deterministic order
        
        # Simple deterministic split: 80% train, 10% val, 10% test
        n_models = len(models)
        n_train = int(n_models * 0.8)
        n_val = int(n_models * 0.9)
        
        if self.split == 'train':
            models = models[:n_train]
        elif self.split == 'val':
            models = models[n_train:n_val]
        elif self.split == 'test':
            models = models[n_val:]
        else:
            print(f"Unknown split {self.split} for ABO, using all data")
            
        if self.valid_models is not None:
            models = [m for m in models if m in self.valid_models]
        
        return [{'model_id': m} for m in models]

    def _get_model_path(self, idx):
        model = self.models[idx]
        model_path = os.path.join(self.data_root, model['model_id'])
        return model['model_id'], model_path

    def __len__(self):
        return len(self.models)

    def name(self):
        return 'ABO'
