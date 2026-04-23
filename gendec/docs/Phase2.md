# PHASE 2 — Unconditional Joint Generation of AutoDec Bottleneck Tokens `(E, Z)`

## Working title
**SQ-Flow Phase 2: Unconditional Generation of AutoDec Structured Object Tokens**

---

## 1. Goal

Phase 2 extends Phase 1 from **explicit scaffold-only generation** to **joint generation of the full AutoDec object bottleneck**.

Instead of learning a prior only over the explicit superquadric scaffold:

\[
\epsilon \rightarrow \hat{E}
\]

we now learn a prior over the **joint AutoDec token set**:

\[
\epsilon \rightarrow (\hat{E}, \hat{Z})
\]

where:

- \(E\) is the explicit superquadric scaffold
- \(Z\) is the per-primitive AutoDec residual latent

The output of Phase 2 is therefore **not only structure**, but **structure + per-part residual detail code**. This makes Phase 2 closer to the real AutoDec bottleneck used by the decoder.

---

## 2. What this Phase 2 is — and what it is not

This file describes the **simple joint Phase 2 design**:

- use a **frozen AutoDec encoder** as teacher
- extract both explicit scaffold tokens \(E\) and residual tokens \(Z\)
- order the primitive slots deterministically offline
- train an unconditional flow model directly on the **joint token tensor**
- optionally decode generated \((E, Z)\) using a **frozen AutoDec decoder** for qualitative and quantitative evaluation

This is **not** the more structured conditional design:

\[
p(Z \mid E)
\]

That conditional design is a possible later variant. The current Phase 2 is simpler:

\[
p(E, Z)
\]

modeled as one unconditional flow prior in the joint token space.

---

## 3. Relationship to Phase 1

### Phase 1
Phase 1 modeled only the explicit scaffold:

\[
E \in \mathbb{R}^{P \times 15}
\]

with per-primitive token:

- scale: 3
- shape: 2
- translation: 3
- rotation (6D): 6
- existence logit: 1

so:

\[
d_E = 15
\]

### Phase 2
Phase 2 augments each primitive token with the AutoDec residual latent:

\[
Z \in \mathbb{R}^{P \times d_Z}
\]

where:

- \(P = 16\) primitive slots
- \(d_Z\) is the AutoDec residual dimension
- default AutoDec residual dimension is typically:

\[
d_Z = 64
\]

Therefore the full joint token has dimension:

\[
d_{EZ} = d_E + d_Z = 15 + 64 = 79
\]

and one object is represented as:

\[
T = [E, Z] \in \mathbb{R}^{16 \times 79}
\]

A minibatch is:

\[
T \in \mathbb{R}^{B \times 16 \times 79}
\]

---

## 4. High-level data flow

The full Phase 2 data flow is:

```text
ShapeNet chair point cloud X [4096,3]
  -> frozen AutoDec encoder
  -> explicit scaffold E [16,15]
  -> residual tokens Z [16,64]
  -> deterministic offline ordering
  -> joint ordered tokens T = [E,Z] [16,79]
  -> normalization
  -> flow model training in token space

Sampling:
noise [B,16,79]
  -> Phase 2 flow model
  -> sampled joint tokens T_hat [B,16,79]
  -> split into E_hat and Z_hat
  -> optional frozen AutoDec decoder
  -> decoded point cloud X_hat
```

---

## 5. Teacher source: frozen AutoDec encoder

## 5.1 Input object

For each training example:

- category: **single-category ShapeNet**, e.g. chairs
- canonical object orientation from dataset preprocessing
- point cloud:

\[
X \in \mathbb{R}^{4096 \times 3}
\]

For batched teacher inference:

\[
X \in \mathbb{R}^{B \times 4096 \times 3}
\]

---

## 5.2 Frozen AutoDec encoder outputs

The frozen AutoDec encoder produces at least:

- `scale`: \([B, P, 3]\)
- `shape`: \([B, P, 2]\)
- `rotate`: \([B, P, 3, 3]\)
- `trans`: \([B, P, 3]\)
- `exist_logit`: \([B, P, 1]\)
- `exist`: \([B, P, 1]\)
- `assign_matrix`: \([B, N, P]\)
- `residual`: \([B, P, d_Z]\)

with:

- \(N = 4096\)
- \(P = 16\)
- default \(d_Z = 64\)

After stripping batch dimension for one object:

- `scale`: \([16, 3]\)
- `shape`: \([16, 2]\)
- `rotate`: \([16, 3, 3]\)
- `trans`: \([16, 3]\)
- `exist_logit`: \([16, 1]\)
- `exist`: \([16, 1]\)
- `assign_matrix`: \([4096, 16]\)
- `residual`: \([16, 64]\)

---

## 5.3 Rotation conversion

As in Phase 1, do **not** model 9D flattened rotation matrices directly in the generative prior.

Convert each teacher rotation matrix:

\[
R_j \in \mathbb{R}^{3 \times 3}
\]

to the 6D representation:

\[
r^{6d}_j \in \mathbb{R}^{6}
\]

using the first two matrix columns.

So the explicit scaffold token remains:

\[
e_j = [s_j, \epsilon_j, t_j, r^{6d}_j, a_j] \in \mathbb{R}^{15}
\]

---

## 5.4 Assignment mass for ordering

Compute primitive assignment mass from the teacher assignment matrix:

\[
m_j = \frac{1}{N} \sum_{i=1}^{N} M_{ij}
\]

where:

- \(M \in \mathbb{R}^{4096 \times 16}\)
- output mass vector:

\[
m \in \mathbb{R}^{16}
\]

---

## 5.5 Primitive volume for ordering

Approximate primitive volume using scales:

\[
V_j = s_{x,j} s_{y,j} s_{z,j}
\]

Output:

\[
V \in \mathbb{R}^{16}
\]

---

## 5.6 Deterministic ordering

Apply deterministic ordering **offline**, exactly once during dataset building.

Recommended Phase 2 ordering rule:

1. existence descending
2. assignment mass descending
3. primitive volume descending
4. translation x ascending

The same permutation must be applied consistently to:

- `scale`
- `shape`
- `rot6d`
- `trans`
- `exist_logit`
- `exist`
- `assign_matrix` columns
- `mass`
- `volume`
- `residual`

This is critical.

In Phase 2, the residual token \(Z_j\) is tied to the same primitive slot as the explicit token \(E_j\). Therefore, **any reordering applied to explicit tokens must also be applied to residual tokens**.

---

## 6. Final token representation for Phase 2

## 6.1 Explicit token

For each primitive:

\[
e_j \in \mathbb{R}^{15}
\]

with:

- scale: 3
- shape: 2
- translation: 3
- rot6d: 6
- exist_logit: 1

---

## 6.2 Residual token

For each primitive:

\[
z_j \in \mathbb{R}^{d_Z}
\]

with default:

\[
d_Z = 64
\]

---

## 6.3 Joint token

Concatenate explicit and residual channels:

\[
t_j = [e_j, z_j] \in \mathbb{R}^{15 + d_Z}
\]

Default:

\[
t_j \in \mathbb{R}^{79}
\]

Stack across the fixed primitive slots:

\[
T = [t_1, \dots, t_{16}] \in \mathbb{R}^{16 \times 79}
\]

For batched training:

\[
T \in \mathbb{R}^{B \times 16 \times 79}
\]

---

## 7. Dataset format

Each offline-saved teacher example should contain at least:

- `points`: \([4096, 3]\)
- `tokens_e`: \([16, 15]\)
- `tokens_z`: \([16, 64]\)
- `tokens_ez`: \([16, 79]\)
- `exist`: \([16, 1]\)
- `mass`: \([16]\)
- `volume`: \([16]\)
- `category_id`: string
- `model_id`: string

Recommended optional fields:

- raw unsorted teacher tensors
- active primitive count
- teacher scaffold preview
- frozen AutoDec decoder preview

---

## 8. Normalization

Do **not** train on raw joint tokens.

The joint token channels have different scales and distributions:

- explicit parameters live in physically meaningful ranges
- residual channels are learned latent values and may be zero-centered but not unit-scaled
- existence logits can have very different magnitude than the residual code

Therefore normalize channelwise.

## 8.1 Joint normalization statistics

Compute dataset-level mean and std over all training objects and all primitive slots:

\[
\mu_{EZ} \in \mathbb{R}^{79}, \quad \sigma_{EZ} \in \mathbb{R}^{79}
\]

with safe clamp:

\[
\sigma_{EZ} \leftarrow \max(\sigma_{EZ}, 10^{-6})
\]

Normalize:

\[
\tilde{T} = (T - \mu_{EZ}) / \sigma_{EZ}
\]

Training is done entirely in normalized joint token space:

\[
\tilde{T} \in \mathbb{R}^{B \times 16 \times 79}
\]

---

## 8.2 Important note on loss balancing

If you simply flatten all 79 channels and compute one uniform MSE, then the residual branch \(Z\) with 64 dimensions can dominate the objective relative to the explicit branch \(E\) with only 15 dimensions.

So the Phase 2 implementation should **not** use only a naive flattened MSE.

Instead, split losses into:

- explicit-token flow loss
- residual-token flow loss

and combine them with explicit weights.

This is a central Phase 2 design choice.

---

## 9. Flow formulation

The Phase 2 flow path is the same as Phase 1, but now over joint tokens.

## 9.1 Clean joint token

\[
\tilde{T}_0 \in \mathbb{R}^{B \times 16 \times 79}
\]

---

## 9.2 Noise token

Sample Gaussian noise of the same shape:

\[
\tilde{T}_1 \sim \mathcal{N}(0, I), \quad \tilde{T}_1 \in \mathbb{R}^{B \times 16 \times 79}
\]

---

## 9.3 Sample time

\[
t \sim \mathcal{U}(0,1), \quad t \in \mathbb{R}^{B}
\]

Broadcast to:

\[
t_{tok} \in \mathbb{R}^{B \times 1 \times 1}
\]

---

## 9.4 Interpolated noisy token

\[
\tilde{T}_t = (1 - t_{tok}) \tilde{T}_0 + t_{tok} \tilde{T}_1
\]

Output shape:

\[
\tilde{T}_t \in \mathbb{R}^{B \times 16 \times 79}
\]

---

## 9.5 Target velocity

For the linear path, the target velocity is:

\[
v^* = \tilde{T}_1 - \tilde{T}_0
\]

with shape:

\[
v^* \in \mathbb{R}^{B \times 16 \times 79}
\]

The network predicts:

\[
\hat{v}_\theta(\tilde{T}_t, t)
\]

with the same shape.

---

## 10. Model architecture

Phase 2 uses the **same overall model family** as Phase 1:

- token-wise Set Transformer flow model
- time conditioning
- global object token
- full self-attention over primitive slots

But the input and output token dimension is now 79 instead of 15.

---

## 10.1 Full forward signature

Input:

- `tt`: \([B, 16, 79]\)
- `t`: \([B]\)

Output:

- `v_hat_e`: \([B, 16, 15]\)
- `v_hat_z`: \([B, 16, 64]\)
- `v_hat`: \([B, 16, 79]\)

Recommended implementation: use a **shared Set Transformer backbone** with **two output heads**.

This is better than one single 79D head because it lets you:

- monitor explicit and residual prediction separately
- weight their losses separately
- avoid the residual branch silently dominating the whole objective

---

## 10.2 Module graph

```text
Tt [B,16,79]
  -> TokenProjection
  -> hidden tokens [B,16,H]
  -> Add time embedding
  -> Prepend learned global token
  -> L x SetTransformerBlock
  -> Drop global token
  -> ExplicitVelocityHead -> v_hat_e [B,16,15]
  -> ResidualVelocityHead -> v_hat_z [B,16,64]
  -> concat -> v_hat [B,16,79]
```

---

## 10.3 TokenProjection

### Input

\[
\tilde{T}_t \in \mathbb{R}^{B \times 16 \times 79}
\]

### Module

```text
Linear(79 -> H)
LayerNorm(H)
SiLU
Linear(H -> H)
```

### Output

\[
X_0 \in \mathbb{R}^{B \times 16 \times H}
\]

Recommended hidden size:

- `H = 384` or `H = 256`

Because the token dimension is much larger than Phase 1, `H=384` is a safer default if compute allows it.

---

## 10.4 TimeEmbedding

### Input

\[
t \in \mathbb{R}^{B}
\]

### Module

```text
SinusoidalEmbedding(t) -> [B, T]
Linear(T -> H)
SiLU
Linear(H -> H)
```

Recommended:

- `T = 128`

### Output

\[
e_t \in \mathbb{R}^{B \times H}
\]

Broadcast:

\[
e_t^{tok} \in \mathbb{R}^{B \times 16 \times H}
\]

Add to token features:

\[
X_1 = X_0 + e_t^{tok}
\]

---

## 10.5 GlobalObjectToken

A learnable parameter:

\[
g \in \mathbb{R}^{H}
\]

Expanded across the batch:

\[
G \in \mathbb{R}^{B \times 1 \times H}
\]

Concatenate with the 16 primitive tokens:

\[
X_2 = [G; X_1] \in \mathbb{R}^{B \times 17 \times H}
\]

---

## 10.6 SetTransformerBlock

Input:

\[
X \in \mathbb{R}^{B \times 17 \times H}
\]

Each block:

1. multi-head self-attention over all 17 tokens
2. residual + LayerNorm
3. feed-forward MLP
4. residual + LayerNorm

Exact structure:

```text
X -> MHA(X,X,X)
  -> residual add
  -> LayerNorm
  -> Linear(H -> 4H)
  -> SiLU
  -> Linear(4H -> H)
  -> residual add
  -> LayerNorm
```

Recommended defaults:

- `n_heads = 8`
- `n_blocks = 6`
- `dropout = 0.0` initially

Output after all blocks:

\[
X_L \in \mathbb{R}^{B \times 17 \times H}
\]

---

## 10.7 Split output heads

Drop the global token:

\[
X_{prim} = X_L[:,1:,:] \in \mathbb{R}^{B \times 16 \times H}
\]

### Explicit head

```text
Linear(H -> H)
SiLU
Linear(H -> 15)
```

Output:

\[
\hat{v}_E \in \mathbb{R}^{B \times 16 \times 15}
\]

### Residual head

```text
Linear(H -> H)
SiLU
Linear(H -> 64)
```

Output:

\[
\hat{v}_Z \in \mathbb{R}^{B \times 16 \times 64}
\]

### Concatenated velocity

\[
\hat{v} = [\hat{v}_E, \hat{v}_Z] \in \mathbb{R}^{B \times 16 \times 79}
\]

---

## 11. Losses

## 11.1 Split target velocity

Split the target velocity into explicit and residual channels:

\[
v^*_E \in \mathbb{R}^{B \times 16 \times 15}
\]

\[
v^*_Z \in \mathbb{R}^{B \times 16 \times 64}
\]

---

## 11.2 Explicit flow loss

\[
\mathcal{L}_E = \frac{1}{B \cdot 16 \cdot 15} \|\hat{v}_E - v^*_E\|_2^2
\]

This controls the quality of the explicit scaffold generation.

---

## 11.3 Residual flow loss

\[
\mathcal{L}_Z = \frac{1}{B \cdot 16 \cdot 64} \|\hat{v}_Z - v^*_Z\|_2^2
\]

This controls the quality of the residual latent generation.

---

## 11.4 Why split losses are required

If you use one single averaged MSE over 79 channels, the residual branch may dominate because:

- it has 64 channels
- it is more flexible than the explicit scaffold

That can cause the model to prioritize predicting the residual latent well while under-optimizing the explicit geometry.

So the default loss should explicitly separate the two branches.

---

## 11.5 Joint flow loss

Recommended default:

\[
\mathcal{L}_{flow} = \lambda_E \mathcal{L}_E + \lambda_Z \mathcal{L}_Z
\]

Default choice:

- \(\lambda_E = 1.0\)
- \(\lambda_Z = 1.0\)

Because each term is already normalized by its own channel count.

If the residual branch still dominates qualitatively, try:

- \(\lambda_E = 1.0\)
- \(\lambda_Z = 0.5\)

---

## 11.6 Existence auxiliary loss

As in Phase 1, reconstruct the model-implied clean token estimate:

\[
\hat{T}_0 = \tilde{T}_t - t_{tok} \hat{v}
\]

Split the explicit branch and take the existence-logit channel from the reconstructed clean token.

Unnormalize only that channel and compute BCE-with-logits against the binary teacher existence target:

\[
y_j = \mathbb{1}[\sigma(a_j) > 0.5]
\]

Output shape:

- predicted logits: \([B,16]\)
- binary targets: \([B,16]\)

Loss:

\[
\mathcal{L}_{exist}
\]

Recommended small weight:

- \(\lambda_{exist} = 0.05\)

---

## 11.7 Optional residual norm regularizer

This is optional.

Because \(Z\) is a learned latent and may become noisy, you may add a mild latent regularization term on the reconstructed clean residual:

\[
\mathcal{L}_{z\_norm} = \frac{1}{B \cdot 16 \cdot 64} \|\hat{Z}_0\|_2^2
\]

where \(\hat{Z}_0\) is the reconstructed clean residual branch.

This is not required for the first implementation and should remain off initially.

---

## 11.8 Total loss

Default Phase 2 loss:

\[
\mathcal{L} = \lambda_E \mathcal{L}_E + \lambda_Z \mathcal{L}_Z + \lambda_{exist} \mathcal{L}_{exist}
\]

with initial recommended values:

- \(\lambda_E = 1.0\)
- \(\lambda_Z = 1.0\)
- \(\lambda_{exist} = 0.05\)

---

## 12. Training loop

## 12.1 Batch contents

Each dataloader batch returns at least:

- `tokens_ez`: \([B,16,79]\)
- `tokens_e`: \([B,16,15]\)
- `tokens_z`: \([B,16,64]\)
- `exist`: \([B,16,1]\)
- optionally `points`: \([B,4096,3]\)
- normalization stats for unnormalization

---

## 12.2 Forward pass

### Step 1
Load normalized clean joint tokens:

\[
\tilde{T}_0 \in \mathbb{R}^{B \times 16 \times 79}
\]

### Step 2
Sample Gaussian noise:

\[
\tilde{T}_1 \in \mathbb{R}^{B \times 16 \times 79}
\]

### Step 3
Sample times:

\[
t \in \mathbb{R}^{B}
\]

### Step 4
Construct interpolated noisy tokens:

\[
\tilde{T}_t = (1-t)\tilde{T}_0 + t\tilde{T}_1
\]

### Step 5
Compute target velocity:

\[
v^* = \tilde{T}_1 - \tilde{T}_0
\]

### Step 6
Run model:

\[
\hat{v}_E, \hat{v}_Z = f_\theta(\tilde{T}_t, t)
\]

### Step 7
Concatenate if needed:

\[
\hat{v} = [\hat{v}_E, \hat{v}_Z]
\]

### Step 8
Compute losses:

- explicit flow loss
- residual flow loss
- optional existence loss

### Step 9
Backprop and optimizer step.

---

## 12.3 Optimizer and schedule

Recommended initial setup:

- optimizer: AdamW
- learning rate: `1e-4`
- weight decay: `1e-4`
- batch size: 64
- epochs: 300
- gradient clipping: `1.0`

If the joint prior is harder to train than Phase 1, a smaller learning rate like `5e-5` is also reasonable.

---

## 13. Sampling / inference

## 13.1 Initial state

Sample Gaussian noise:

\[
\tilde{T}_{1} \sim \mathcal{N}(0, I), \quad \tilde{T}_{1} \in \mathbb{R}^{B \times 16 \times 79}
\]

---

## 13.2 ODE integration

Use Euler integration for the first version.

Let:

\[
1 = t_0 > t_1 > \dots > t_K = 0
\]

At each step:

\[
\tilde{T}_{t_{k+1}} = \tilde{T}_{t_k} - \hat{v}_\theta(\tilde{T}_{t_k}, t_k) \cdot (t_k - t_{k+1})
\]

with default:

- `K = 50` steps

Output:

\[
\hat{\tilde{T}}_0 \in \mathbb{R}^{B \times 16 \times 79}
\]

Unnormalize to get:

\[
\hat{T}_0 \in \mathbb{R}^{B \times 16 \times 79}
\]

---

## 13.3 Split generated tokens

After unnormalization, split into:

### Explicit part

\[
\hat{E} \in \mathbb{R}^{B \times 16 \times 15}
\]

### Residual part

\[
\hat{Z} \in \mathbb{R}^{B \times 16 \times 64}
\]

Then further split \(\hat{E}\) into:

- scale: \([B,16,3]\)
- shape: \([B,16,2]\)
- trans: \([B,16,3]\)
- rot6d: \([B,16,6]\)
- exist_logit: \([B,16,1]\)

Convert 6D rotations to matrices:

\[
\hat{R} \in \mathbb{R}^{B \times 16 \times 3 \times 3}
\]

Compute existence:

\[
\hat{\alpha} = \sigma(\hat{a}) \in \mathbb{R}^{B \times 16 \times 1}
\]

Threshold to obtain active primitives.

---

## 14. Optional frozen AutoDec decoder bridge

The simplest and most meaningful evaluation of Phase 2 is to pass generated \((E,Z)\) through a **frozen AutoDec decoder**.

## 14.1 Decoder input outdict

Build a decoder input dictionary with:

- `scale`: \([B,16,3]\)
- `shape`: \([B,16,2]\)
- `rotate`: \([B,16,3,3]\)
- `trans`: \([B,16,3]\)
- `exist_logit`: \([B,16,1]\)
- `exist`: \([B,16,1]\)
- `residual`: \([B,16,64]\)

This exactly matches the fields needed by the AutoDec decoder.

---

## 14.2 Decoder output

The frozen AutoDec decoder produces at least:

- `surface_points`: \([B, M, 3]\)
- `decoded_offsets`: \([B, M, 3]\)
- `decoded_points`: \([B, M, 3]\)
- `decoded_weights`: \([B, M]\)

where typically:

\[
M = P \cdot S = 16 \cdot 256 = 4096
\]

So the final decoded point cloud is:

\[
\hat{X} \in \mathbb{R}^{B \times 4096 \times 3}
\]

before any inference-time pruning.

This gives the natural full Phase 2 generative pipeline:

\[
\epsilon \rightarrow (\hat{E}, \hat{Z}) \rightarrow \text{frozen AutoDec decoder} \rightarrow \hat{X}
\]

---

## 15. Suggested metrics

### Token-space metrics

Track separately:

- `flow_loss_e`
- `flow_loss_z`
- `exist_loss`
- per-field explicit MSE:
  - scale
  - shape
  - translation
  - rotation6d
  - existence

### Token plausibility metrics

After sampling:

- active primitive count
- positive scale fraction
- valid exponent fraction
- orthonormal rotation fraction
- residual norm statistics

### Decoded-geometry metrics

If the frozen decoder bridge is enabled:

- point-cloud Chamfer-L1/L2
- F-score
- nearest-neighbor plausibility against held-out test set
- decoded active point count after existence gating

---

## 16. Recommended code modules

### `phase2/data/build_teacher_dataset.py`
Responsibilities:

- load ShapeNet chairs
- run frozen AutoDec encoder
- extract `E` and `Z`
- convert rotation to 6D
- compute mass and volume
- apply deterministic ordering jointly to `E` and `Z`
- save teacher examples

### `phase2/data/dataset.py`
Responsibilities:

- load saved joint tokens
- normalize channelwise
- return `tokens_ez`, `tokens_e`, `tokens_z`, `exist`

### `phase2/models/time_embedding.py`
Responsibilities:

- sinusoidal time embedding
- MLP projection

### `phase2/models/set_transformer_flow.py`
Responsibilities:

- token projection from 79D to hidden dim
- global token
- transformer blocks
- explicit velocity head
- residual velocity head

### `phase2/models/rotation.py`
Responsibilities:

- matrix to 6D
- 6D to matrix

### `phase2/losses/flow_matching.py`
Responsibilities:

- build flow path in joint token space
- split explicit/residual branches
- compute `L_E`, `L_Z`, `L_exist`

### `phase2/eval/autodec_bridge.py`
Responsibilities:

- convert sampled tokens into AutoDec decoder outdict
- run frozen decoder
- save decoded results

### `phase2/train.py`
Responsibilities:

- optimizer
- scheduler
- checkpointing
- validation
- optional sample decoding

### `phase2/sample.py`
Responsibilities:

- Euler sampler in joint token space
- token unnormalization
- split `(E,Z)`
- optional decoder call
- visualization export

---

## 17. Minimal tensor contract

### Dataset output

- `tokens_ez`: \([B,16,79]\)
- `tokens_e`: \([B,16,15]\)
- `tokens_z`: \([B,16,64]\)
- `exist`: \([B,16,1]\)

### Flow training input

- `T0`: \([B,16,79]\)
- `T1`: \([B,16,79]\)
- `t`: \([B]\)
- `Tt`: \([B,16,79]\)
- `v_target`: \([B,16,79]\)

### Model forward

Input:

- `Tt`: \([B,16,79]\)
- `t`: \([B]\)

Output:

- `v_hat_e`: \([B,16,15]\)
- `v_hat_z`: \([B,16,64]\)
- `v_hat`: \([B,16,79]\)

### Sampling output

- generated explicit tokens: \([B,16,15]\)
- generated residual tokens: \([B,16,64]\)
- generated rotation matrices: \([B,16,3,3]\)
- generated active mask: \([B,16]\)

### Decoder bridge input

- `scale`: \([B,16,3]\)
- `shape`: \([B,16,2]\)
- `rotate`: \([B,16,3,3]\)
- `trans`: \([B,16,3]\)
- `exist_logit`: \([B,16,1]\)
- `exist`: \([B,16,1]\)
- `residual`: \([B,16,64]\)

### Decoder output

- `decoded_points`: \([B,4096,3]\)
- `decoded_offsets`: \([B,4096,3]\)
- `decoded_weights`: \([B,4096]\)

---

## 18. Recommended default hyperparameters

```text
P = 16
D_E = 15
D_Z = 64
D_token = 79
H = 384
L = 6
n_heads = 8
ff_mult = 4
batch_size = 64
lr = 1e-4
weight_decay = 1e-4
epochs = 300
n_sampling_steps = 50
exist_threshold = 0.5
lambda_E = 1.0
lambda_Z = 1.0
lambda_exist = 0.05
```

---

## 19. First Phase 2 milestone

The first working milestone is reached when the code can:

1. build a teacher dataset from frozen AutoDec encoder outputs
2. store ordered joint `(E,Z)` tokens per object
3. train a Set Transformer flow model on `[16,79]` joint tokens
4. sample new joint tokens from noise
5. split them back into explicit scaffold and residual code
6. pass them through a frozen AutoDec decoder
7. produce plausible chair-like decoded point clouds

That is the complete target for this simple joint Phase 2 design.

---

## 20. Main caveat of this Phase 2 design

This Phase 2 is intentionally simple, but it has one major caveat:

Because it generates \((E,Z)\) jointly from noise, it does **not explicitly enforce** the desirable hierarchy:

- explicit scaffold handles coarse structure
- residual handles only detail

So if trained carelessly, the residual branch may absorb too much of the generative burden.

That is why the split losses and frozen-decoder evaluation path are important. They let you monitor whether:

- the explicit scaffold remains meaningful
- the residual latent is helping, not replacing, the scaffold

A later conditional design \(p(Z \mid E)\) is more principled, but the current joint design is the simplest faithful extension of Phase 1.
