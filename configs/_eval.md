========== SuperDec Evaluation ==========
Config:
type: none
prefix: normalized/shapenet_test
small: true
num_epochs: 1000
device: cuda
optimization:
  tapering: false
  bending: false
  reorient: false

----- Evaluation Results -----
          mean_chamfer_l1: 0.017940
          mean_chamfer_l2: 0.000502
                 mean_iou: 0.479687
             mean_f-score: 0.249628
          mean_f-score-15: 0.484598
          mean_f-score-20: 0.669967
       avg_num_primitives: 5.843750


========== SuperDec Evaluation ==========
Config:
type: none
prefix: shapenet/shapenet_test
small: true
num_epochs: 1000
device: cuda
optimization:
  tapering: false
  bending: false
  reorient: false

----- Evaluation Results -----
          mean_chamfer_l1: 0.017373
          mean_chamfer_l2: 0.000466
                 mean_iou: 0.483669
             mean_f-score: 0.259548
          mean_f-score-15: 0.498542
          mean_f-score-20: 0.684692
       avg_num_primitives: 6.179688


========== SuperDec Evaluation ==========
Config:
type: cd
small: true
num_epochs: 1000
device: cuda
optimization:
  tapering: false
  bending: false
  reorient: false
  pruning: false

----- Evaluation Results -----
          mean_chamfer_l1: 0.015796
          mean_chamfer_l2: 0.000422
                 mean_iou: 0.495267
             mean_f-score: 0.321018
          mean_f-score-15: 0.549997
          mean_f-score-20: 0.707897
       avg_num_primitives: 6.179688


========== SuperDec Evaluation ==========
Config:
type: base
prefix: shapenet/shapenet_test
small: true
num_epochs: 1000
device: cuda
optimization:
  tapering: false
  bending: false
  reorient: false
  pruning: false

----- Evaluation Results -----
          mean_chamfer_l1: 0.019699
          mean_chamfer_l2: 0.001131
                 mean_iou: 0.468757
             mean_f-score: 0.363740
          mean_f-score-15: 0.593120
          mean_f-score-20: 0.747623
       avg_num_primitives: 6.179688


========== SuperDec Evaluation ==========
Config:
type: iou
prefix: shapenet/shapenet_test
small: true
num_epochs: 1000
device: cuda
optimization:
  tapering: false
  bending: false
  reorient: false

----- Evaluation Results -----
          mean_chamfer_l1: 0.014669
          mean_chamfer_l2: 0.000403
                 mean_iou: 0.540296
             mean_f-score: 0.393255
          mean_f-score-15: 0.631395
          mean_f-score-20: 0.784103
       avg_num_primitives: 6.179688


========== SuperDec Evaluation ==========
Config:
type: iou
prefix: shapenet/shapenet_test
small: true
num_epochs: 1000
device: cuda
optimization:
  tapering: false
  bending: false
  reorient: true
  pruning: false

----- Evaluation Results -----
          mean_chamfer_l1: 0.014656
          mean_chamfer_l2: 0.000406
                 mean_iou: 0.540576
             mean_f-score: 0.394703
          mean_f-score-15: 0.631822
          mean_f-score-20: 0.784900
       avg_num_primitives: 6.179688

========== SuperDec Evaluation ==========
Config:
type: iou
prefix: shapenet/shapenet_test
small: true
num_epochs: 1000
device: cuda
optimization:
  tapering: true
  bending: false
  reorient: false
  pruning: false

----- Evaluation Results -----
          mean_chamfer_l1: 0.014037
          mean_chamfer_l2: 0.000361
                 mean_iou: 0.545176
             mean_f-score: 0.403846
          mean_f-score-15: 0.647247
          mean_f-score-20: 0.800333
       avg_num_primitives: 6.179688

========== SuperDec Evaluation ==========
Config:
type: iou
prefix: shapenet/shapenet_test
small: true
num_epochs: 1000
device: cuda
optimization:
  tapering: false
  bending: true
  reorient: false
  pruning: false

----- Evaluation Results -----
          mean_chamfer_l1: 0.013840
          mean_chamfer_l2: 0.000358
                 mean_iou: 0.830638
             mean_f-score: 0.416265
          mean_f-score-15: 0.659507
          mean_f-score-20: 0.808990
       avg_num_primitives: 6.179688

========== SuperDec Evaluation ==========
Config:
type: iou
prefix: shapenet/shapenet_test
small: true
num_epochs: 1000
device: cuda
optimization:
  tapering: true
  bending: true
  reorient: false
  pruning: false

----- Evaluation Results -----
          mean_chamfer_l1: 0.013566
          mean_chamfer_l2: 0.000330
                 mean_iou: 0.841085
             mean_f-score: 0.416376
          mean_f-score-15: 0.663414
          mean_f-score-20: 0.814809
       avg_num_primitives: 6.179688


========== SuperDec Evaluation ==========
Config:
type: iou
prefix: shapenet/shapenet_test
small: true
num_epochs: 1000
device: cuda
optimization:
  tapering: true
  bending: true
  reorient: true
  pruning: false

----- Evaluation Results -----
          mean_chamfer_l1: 0.013578
          mean_chamfer_l2: 0.000332
                 mean_iou: 0.839025
             mean_f-score: 0.417450
          mean_f-score-15: 0.663110
          mean_f-score-20: 0.813497
       avg_num_primitives: 6.179688


========== SuperDec Evaluation ==========
Config:
type: iou
prefix: shapenet/shapenet_test
small: true
num_epochs: 1000
device: cuda
optimization:
  tapering: true
  bending: true
  reorient: true
  pruning: true

