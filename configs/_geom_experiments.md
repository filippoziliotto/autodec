# Geom 001 - Geom loss
1000 epochs 4e-4 lr
### Cost and Weights
  w_geometric: 10.0
  w_geometric_f: 10.0
  w_exist: 1.0

  c_geometric: 10.0 
  c_z_align: 0.0
  c_exist: 1.0

## Exp 011 - Geom loss
clear shape , 3e-4lr, fix loss (gt etas/omegas)

## Exp 021 - Geom loss
start from ckpt of 011 epoch 1000
  w_geometric: 10.0
  w_geometric_f: 0.0
  w_exist: 1.0

  c_geometric: 10.0 
  c_z_align: 0.0
  c_exist: 1.0

## Exp 031 - Geom loss
lr: 2e-4
start from ckpt of 011 epoch 1000
  w_geometric: 10.0
  w_geometric_f: 1.0
  w_exist: 1.0

  c_geometric: 10.0 
  c_z_align: 1.0
  c_exist: 1.0

# Exp 002 - Geom Loss with Occlusion
### Cost and Weights




## Results

| Experiment | mean_chamfer_l1 | mean_chamfer_l2 | mean_iou | mean_f-score | avg_num_primitives |
|---|---:|---:|---:|---:|---:|
| superdec      | 0.017680 | 0.000499 | 0.491245 | 0.258831 | 5.793852 |
| geom_033 (old)| 0.016635 | 0.000480 | 0.640249 | 0.288719 | 5.873729 |
| geom_011      | 0.016509 | 0.000479 | 0.643321 | 0.288373 | 5.854759 |
| geom_tb       | 0.016509 | 0.000479 | 0.643321 | 0.288373 | 5.854759 |