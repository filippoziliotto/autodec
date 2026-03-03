========== SuperDec Evaluation ==========
Config:
type: none
prefix: shapenet/shapenet_test
small: false
num_epochs: 1000
device: cuda
optimization:
  tapering: false
  bending: false
  reorient: false
dataloader:
  dataset: shapenet
  batch_size: 256
  num_workers: 4
shapenet:
  split: test
  path: data/ShapeNet
  categories: null
  load_occupancy: true
  use_fps: true

Evaluating none on shapenet/shapenet_test...
Loading data/output_npz/shapenet/shapenet_test.npz...
Loaded 8751 objects from all categories out of 8751.
Processing batches: 100%|██████████| 35/35 [11:02<00:00, 18.93s/it]
Saving per-object metrics to data/output_npz/shapenet/shapenet_test_optimized_none_metrics.csv...

----- Evaluation Results -----
          mean_chamfer_l1: 0.017680
          mean_chamfer_l2: 0.000499
                 mean_iou: 0.491245
             mean_f-score: 0.258831
          mean_f-score-15: 0.495139
          mean_f-score-20: 0.674159
       avg_num_primitives: 5.793852



========== SuperDec Evaluation ==========
Config:
type: cd
small: false
num_epochs: 1000
device: cuda
optimization:
  tapering: false
  bending: false
  reorient: false
  pruning: false
dataloader:
  dataset: shapenet
  batch_size: 32
  num_workers: 8
shapenet:
  split: test
  path: data/ShapeNet
  categories: null
  load_occupancy: true
  use_fps: true

Processing batches: 100%|██████████| 274/274 [7:32:26<00:00, 99.08s/it]
Saving optimized results to data/output_npz/shapenet/cd/test.npz...
Saving per-object metrics to data/output_npz/shapenet/cd/test_metrics.csv...

----- Evaluation Results -----
          mean_chamfer_l1: 0.015962
          mean_chamfer_l2: 0.000428
                 mean_iou: 0.503051
             mean_f-score: 0.362205
          mean_f-score-15: 0.606642
          mean_f-score-20: 0.754342
       avg_num_primitives: 5.793852


========== SuperDec Evaluation ==========
Config:
type: base
prefix: shapenet/shapenet_test
small: false
num_epochs: 1000
device: cuda
optimization:
  tapering: false
  bending: false
  reorient: false
dataloader:
  dataset: shapenet
  batch_size: 256
  num_workers: 4
shapenet:
  split: test
  path: data/ShapeNet
  categories: null
  load_occupancy: true
  use_fps: true

Evaluating base on shapenet/shapenet_test...
Loading data/output_npz/shapenet/shapenet_test.npz...
Loaded 8751 objects from all categories out of 8751.
Processing batches: 100%|██████████| 35/35 [21:54<00:00, 37.56s/it]
Saving optimized results to data/output_npz/shapenet/shapenet_test_optimized_base.npz...
Saving per-object metrics to data/output_npz/shapenet/shapenet_test_optimized_base_metrics.csv...

----- Evaluation Results -----
          mean_chamfer_l1: 0.020565
          mean_chamfer_l2: 0.001468
                 mean_iou: 0.483176
             mean_f-score: 0.358773
          mean_f-score-15: 0.591397
          mean_f-score-20: 0.744762
       avg_num_primitives: 5.793852



========== SuperDec Evaluation ==========
Config:
type: iou
small: false
num_epochs: 1000
device: cuda
optimization:
  tapering: false
  bending: false
  reorient: false
  pruning: false
dataloader:
  dataset: shapenet
  batch_size: 256
  num_workers: 4
shapenet:
  split: test
  path: data/ShapeNet
  categories: null
  load_occupancy: true
  use_fps: true

Evaluating iou on shapenet test...
Loading data/output_npz/shapenet/shapenet_test.npz...
Will save results to data/output_npz/shapenet/iou/test.npz...
Loaded 8751 objects from all categories out of 8751.
Processing batches: 100%|██████████| 35/35 [1:22:51<00:00, 142.04s/it]
Saving optimized results to data/output_npz/shapenet/iou/test.npz...
Saving per-object metrics to data/output_npz/shapenet/iou/test_metrics.csv...

----- Evaluation Results -----
          mean_chamfer_l1: 0.014585
          mean_chamfer_l2: 0.000411
                 mean_iou: 0.559141
             mean_f-score: 0.397528
          mean_f-score-15: 0.637725
          mean_f-score-20: 0.786554
       avg_num_primitives: 5.793852



========== SuperDec Evaluation ==========
Config:
type: iou
small: false
num_epochs: 1000
device: cuda
optimization:
  tapering: false
  bending: false
  reorient: true
  pruning: false
dataloader:
  dataset: shapenet
  batch_size: 256
  num_workers: 4
shapenet:
  split: test
  path: data/ShapeNet
  categories: null
  load_occupancy: true
  use_fps: true

Evaluating iou on shapenet test...
Loading data/output_npz/shapenet/shapenet_test.npz...
Will save results to data/output_npz/shapenet/iou_r/test.npz...
Loaded 8751 objects from all categories out of 8751.
Processing batches: 100%|██████████| 35/35 [1:24:34<00:00, 145.00s/it]
Saving optimized results to data/output_npz/shapenet/iou_r/test.npz...
Saving per-object metrics to data/output_npz/shapenet/iou_r/test_metrics.csv...

----- Evaluation Results -----
          mean_chamfer_l1: 0.014828
          mean_chamfer_l2: 0.000423
                 mean_iou: 0.556734
             mean_f-score: 0.391482
          mean_f-score-15: 0.630518
          mean_f-score-20: 0.780649
       avg_num_primitives: 5.793852