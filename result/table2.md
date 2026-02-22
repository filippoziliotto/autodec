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
Processing batches: 100%|██████████| 35/35 [1:22:29<00:00, 141.43s/it]
Saving optimized results to data/output_npz/shapenet/iou/test.npz...
Saving per-object metrics to data/output_npz/shapenet/iou/test_metrics.csv...

----- Evaluation Results -----
          mean_chamfer_l1: 0.014845
          mean_chamfer_l2: 0.000422
                 mean_iou: 0.556882
             mean_f-score: 0.389055
          mean_f-score-15: 0.629756
          mean_f-score-20: 0.780638
       avg_num_primitives: 5.793852



========== SuperDec Evaluation ==========
Config:
type: iou
small: false
num_epochs: 1000
device: cuda
optimization:
  tapering: true
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
Will save results to data/output_npz/shapenet/iou_t/test.npz...
Loaded 8751 objects from all categories out of 8751.
Processing batches: 100%|██████████| 35/35 [1:29:10<00:00, 152.88s/it]
Saving optimized results to data/output_npz/shapenet/iou_t/test.npz...
Saving per-object metrics to data/output_npz/shapenet/iou_t/test_metrics.csv...

----- Evaluation Results -----
          mean_chamfer_l1: 0.014220
          mean_chamfer_l2: 0.000384
                 mean_iou: 0.562896
             mean_f-score: 0.404594
          mean_f-score-15: 0.648251
          mean_f-score-20: 0.796985
       avg_num_primitives: 5.793852



========== SuperDec Evaluation ==========
Config:
type: iou
small: false
num_epochs: 1000
device: cuda
optimization:
  tapering: false
  bending: true
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
Will save results to data/output_npz/shapenet/iou_b/test.npz...
Loaded 8751 objects from all categories out of 8751.
Processing batches: 100%|██████████| 35/35 [1:51:43<00:00, 191.54s/it]
Saving optimized results to data/output_npz/shapenet/iou_b/test.npz...
Saving per-object metrics to data/output_npz/shapenet/iou_b/test_metrics.csv...

----- Evaluation Results -----
          mean_chamfer_l1: 0.014016
          mean_chamfer_l2: 0.000377
                 mean_iou: 0.828522
             mean_f-score: 0.413230
          mean_f-score-15: 0.657580
          mean_f-score-20: 0.804983
       avg_num_primitives: 5.793852
       


========== SuperDec Evaluation ==========
Config:
type: iou
prefix: shapenet/shapenet_test
small: false
num_epochs: 1000
device: cuda
optimization:
  tapering: true
  bending: true
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

Evaluating iou on shapenet/shapenet_test...
Loading data/output_npz/shapenet/shapenet_test.npz...
Loaded 8751 objects from all categories out of 8751.
Processing batches: 100%|██████████| 35/35 [2:04:06<00:00, 212.75s/it]
Saving optimized results to data/output_npz/shapenet/shapenet_test_optimized_iou.npz...
Saving per-object metrics to data/output_npz/shapenet/shapenet_test_optimized_iou_metrics.csv...

----- Evaluation Results -----
          mean_chamfer_l1: 0.013797
          mean_chamfer_l2: 0.000364
                 mean_iou: 0.839617
             mean_f-score: 0.417229
          mean_f-score-15: 0.663565
          mean_f-score-20: 0.810512
       avg_num_primitives: 5.793852



========== SuperDec Evaluation ==========
Config:
type: iou
prefix: shapenet/shapenet_test
small: false
num_epochs: 1000
device: cuda
optimization:
  tapering: true
  bending: true
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

Evaluating iou on shapenet/shapenet_test...
Loading data/output_npz/shapenet/shapenet_test.npz...
Loaded 8751 objects from all categories out of 8751.
Processing batches: 100%|██████████| 35/35 [2:03:56<00:00, 212.48s/it]
Saving optimized results to data/output_npz/shapenet/shapenet_test_optimized_iou.npz...
Saving per-object metrics to data/output_npz/shapenet/shapenet_test_optimized_iou_metrics.csv...

----- Evaluation Results -----
          mean_chamfer_l1: 0.013788
          mean_chamfer_l2: 0.000363
                 mean_iou: 0.838679
             mean_f-score: 0.418064
          mean_f-score-15: 0.662941
          mean_f-score-20: 0.809869
       avg_num_primitives: 5.793852