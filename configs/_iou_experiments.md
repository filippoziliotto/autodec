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

## Exp 431 - IoU loss
finetune 031 with w_sps: 1.3 and w_overlap: 15.0
200 epoch 1e-4 lr

## Exp 531 - IoU loss
finetune 031 with w_sps: 1.3 and w_overlap: 35.0
200 epoch 1e-4 lr



# Exp 041 - IoU loss
init non zero extended 3e-4 lr

# Exp 141 - IoU loss
finetune 041 with w_sps: 1.3 and w_overlap: 35.0
200 epoch 1e-4 lr


# Exp 051 - IoU loss
init non zero extended 4e-4 lr

## Exp 151 - IoU loss
finetune 051 with w_sps: 1.4
200 epoch 1e-4 lr

# Results
| Experiment | mean_chamfer_l1 | mean_chamfer_l2 | mean_iou | mean_f-score | avg_num_primitives |
|---|---:|---:|---:|---:|---:|
| superdec      | 0.017680 | 0.000499 | 0.491245 | 0.258831 | 5.793852 |
| superdec+ opt | 0.013788 | 0.000363 | 0.838679 | 0.418064 | 6.793852 |
| iou_031       | 0.016638 | 0.000519 | 0.692310 | 0.303842 | 6.147069 |
| iou_031 + opt | 0.013658 | 0.000410 | 0.859409 | 0.440683 | 6.146612 |
| iou_231       | 0.016710 | 0.000537 | 0.692084 | 0.303356 | 5.983659 |
| iou_331       | 0.016862 | 0.000569 | 0.690834 | 0.301815 | 5.775340 |
| iou_331 + opt | 0.013884 | 0.000463 | 0.855200 | 0.437946 | 5.775226 |
| iou_331+opt(r)| 0.013883 | 0.000463 | 0.854621 | 0.438769 | 5.775226 |
| iou_041       | 0.016550 | 0.000572 | 0.691837 | 0.306772 | 5.956576 |



epoch 2000
Saving per-object metrics to data/output_npz/shapenet_iou_031/shapenet_iou_031/none/test_metrics.csv...
----- Evaluation Results -----
          mean_chamfer_l1: 0.016501
          mean_chamfer_l2: 0.000513
                 mean_iou: 0.698831
             mean_f-score: 0.310637
          mean_f-score-15: 0.567570
          mean_f-score-20: 0.745357
       avg_num_primitives: 6.361559