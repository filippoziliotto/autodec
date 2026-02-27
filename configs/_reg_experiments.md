# Exp 001 - BCE & Parameter loss
### Cost and Weights
  w_exist: 1.0
  w_scale: 8.0
  w_shape: 1.0
  w_trans: 8.0
  w_rot: 0.2
  w_tapering: 1.0
  w_bending: 0.05

  c_exist: 1.0
  c_scale: 50.0
  c_trans: 10.0
  c_rot: 1.0

## Exp 011 - BCE & Parameter loss
As before but using 6D rotation

## Exp 021 - BCE & Parameter loss
As before but single mlp head

## Exp 031 - BCE & Parameter loss
As before but multi mlp head (1 per param "group")

## Exp 041 - BCE & Parameter loss
multi mlp head but filter on     
  metric: iou
  threshold: 0.87

# Exp 002 - BCE & Parameter & Geometric loss
### Cost and Weights
cost uses only geometric and exist
  w_geometric: 1.0
  c_geometric: 1.0 
  c_exist: 1.0

  w_exist: 1.0
  w_scale: 8.0
  w_shape: 1.0
  w_trans: 8.0
  w_rot: 0.2
  w_tapering: 1.0
  w_bending: 0.05
  w_sps: 0.0

## Exp 012 - BCE & Parameter & Geometric loss
As before but CD instead of 1-1 matching of geometric points
no mse on rot (yes on the rest - otherwise degenerates)
plus mse on tapering and bending ks sum

## Exp 022 - BCE & Geometric loss
back to 1-1 geometric loss but remove rotation loss
and increase bending and scale weight 

## Exp 032 - BCE & Geometric loss & Assign Cd
introduced a pc to sq chamfer distance loss weighted on the assign matrix
this gives some training signal to the assign matrix

# Exp 003 - BCE & Geometric loss
cost & loss uses only geometric and exist
  w_geometric: 1.0
  c_geometric: 1.0 
  w_exist: 1.0
  c_exist: 1.0

  w_scale: 0.0
  w_shape: 0.0
  w_trans: 0.0
  w_rot: 0.0
  w_tapering: 0.0
  w_bending: 0.0
  w_sps: 0.0


## Exp 013 - BCE & Geometric loss
As before but CD instead of 1-1 matching of geometric points

## Exp 023 - BCE & Geometric loss
As before but sinkhorn instead of CD matching of geometric points


# Exp 004 - BCE & Parameter & Geometric loss & Occlusions
normalization and occlusions