# Exp 031 - IoU loss
1000 epoch 4e-4 lr
### Cost and Weights
  type: iou
  w_sps: 1.2
  w_ext: 0.02

  truncation: 0.05
  w_sdf: 0.2
  w_bbox: 0.2
  w_iou: 0.2
  w_overlap: 5.0

## Exp 131 - IoU loss
finetune 031 with w_sps: 1.25
200 epoch 1e-4 lr

## Exp 231 - IoU loss
finetune 031 with w_sps: 1.3
200 epoch 1e-4 lr

## Exp 331 - IoU loss
finetune 031 with w_sps: 1.4
200 epoch 1e-4 lr


# Results
| Experiment | mean_chamfer_l1 | mean_chamfer_l2 | mean_iou | mean_f-score | avg_num_primitives |
|---|---:|---:|---:|---:|---:|
| superdec      | 0.017680 | 0.000499 | 0.491245 | 0.258831 | 5.793852 |
| iou_031       | 0.016638 | 0.000519 | 0.692310 | 0.303842 | 6.147069 |
| iou_031 + opt | 0.013658 | 0.000410 | 0.859409 | 0.440683 | 6.146612 |
| iou_231       | 0.016710 | 0.000537 | 0.692084 | 0.303356 | 5.983659 |
