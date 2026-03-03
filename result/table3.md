
========== SuperDec Evaluation ==========
Config:
type: iou
small: false
num_epochs: 1000
device: cuda
optimization:
  tapering: true
  bending: true
  reorient: false
  pruning: false
  w_sdf: 0.0
  w_bbox: 0.0
  w_overlap: 0.0
dataloader:
  dataset: shapenet
  batch_size: 128
  num_workers: 8
shapenet:
  split: test
  path: data/ShapeNet
  categories: null
  load_occupancy: true
  use_fps: true

Processing batches: 100%|██████████| 69/69 [2:11:43<00:00, 114.54s/it]
Saving optimized results to data/output_npz/shapenet/iou_tb/test.npz...
Saving per-object metrics to data/output_npz/shapenet/iou_tb/test_metrics.csv...

----- Evaluation Results -----
          mean_chamfer_l1: 0.013747
          mean_chamfer_l2: 0.000407
                 mean_iou: 0.860458
             mean_f-score: 0.479885
          mean_f-score-15: 0.724104
          mean_f-score-20: 0.842870
       avg_num_primitives: 5.793852


========== SuperDec Evaluation ==========
Config:
type: iou
small: false
num_epochs: 1000
device: cuda
optimization:
  tapering: true
  bending: true
  reorient: false
  pruning: false
  w_overlap: 0.0
  w_bbox: 0.0
dataloader:
  dataset: shapenet
  batch_size: 128
  num_workers: 8
shapenet:
  split: test
  path: data/ShapeNet
  categories: null
  load_occupancy: true
  use_fps: true

Evaluating iou on shapenet test...
Loading data/output_npz/shapenet/shapenet_test.npz...
Will save results to data/output_npz/shapenet/iou_tb/test.npz...
Loaded 8751 objects from all categories out of 8751.
Processing batches: 100%|██████████| 69/69 [2:12:36<00:00, 115.31s/it]
Saving optimized results to data/output_npz/shapenet/iou_tb/test.npz...
Saving per-object metrics to data/output_npz/shapenet/iou_tb/test_metrics.csv...

----- Evaluation Results -----
          mean_chamfer_l1: 0.014066
          mean_chamfer_l2: 0.000721
                 mean_iou: 0.850866
             mean_f-score: 0.471407
          mean_f-score-15: 0.719132
          mean_f-score-20: 0.841676
       avg_num_primitives: 5.793852


========== SuperDec Evaluation ==========
Config:
type: iou
small: false
num_epochs: 1000
device: cuda
optimization:
  tapering: true
  bending: true
  reorient: false
  pruning: false
  w_overlap: 0.0
dataloader:
  dataset: shapenet
  batch_size: 128
  num_workers: 8
shapenet:
  split: test
  path: data/ShapeNet
  categories: null
  load_occupancy: true
  use_fps: true

Evaluating iou on shapenet test...
Loading data/output_npz/shapenet/shapenet_test.npz...
Will save results to data/output_npz/shapenet/iou_tb/test.npz...
Loaded 8751 objects from all categories out of 8751.
Processing batches: 100%|██████████| 69/69 [2:11:01<00:00, 113.94s/it]
Saving optimized results to data/output_npz/shapenet/iou_tb/test.npz...
Saving per-object metrics to data/output_npz/shapenet/iou_tb/test_metrics.csv...

----- Evaluation Results -----
          mean_chamfer_l1: 0.013529
          mean_chamfer_l2: 0.000358
                 mean_iou: 0.851024
             mean_f-score: 0.473468
          mean_f-score-15: 0.721891
          mean_f-score-20: 0.844490
       avg_num_primitives: 5.793852


========== SuperDec Evaluation ==========
Config:
type: iou
small: false
num_epochs: 1000
device: cuda
optimization:
  tapering: true
  bending: true
  reorient: false
  pruning: false
dataloader:
  dataset: shapenet
  batch_size: 128
  num_workers: 8
shapenet:
  split: test
  path: data/ShapeNet
  categories: null
  load_occupancy: true
  use_fps: true

Processing batches: 100%|██████████| 69/69 [2:10:03<00:00, 113.09s/it]
Saving optimized results to data/output_npz/shapenet/iou_tb/test.npz...
Saving per-object metrics to data/output_npz/shapenet/iou_tb/test_metrics.csv...

----- Evaluation Results -----
          mean_chamfer_l1: 0.013502
          mean_chamfer_l2: 0.000356
                 mean_iou: 0.850852
             mean_f-score: 0.474068
          mean_f-score-15: 0.722705
          mean_f-score-20: 0.845227
       avg_num_primitives: 5.793852