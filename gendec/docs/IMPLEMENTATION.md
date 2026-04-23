# IMPLEMENTATION — Phase 1 Superquadric Scaffold Flow Model

## 1. Objective

Implement **Phase 1** of the object-level generative pipeline:

\[
\epsilon \rightarrow \hat{E}
\]

where \(\hat{E}\) is a generated explicit superquadric scaffold for a single-category object (chair).

This document focuses only on **Phase 1** with the following fixed choices:

- deterministic primitive ordering
- token-wise Set Transformer flow model
- frozen SuperDec teacher outputs
- single-category ShapeNet chairs
- 4096-point canonical inputs

The goal is to learn a flow-matching prior over ordered superquadric tokens.

---

## 2. Training data pipeline

## 2.1 Raw input

For each training example:

- object category: **chair**
- dataset: **ShapeNet**
- input point cloud:

\[
X \in \mathbb{R}^{4096 \times 3}
\]

These point clouds must use exactly the same canonical preprocessing used by the frozen teacher pipeline.

---

## 2.2 Frozen teacher inference

Run a **frozen SuperDec** model on each point cloud.

Teacher outputs needed per object:

- `scale`: \([P, 3]\)
- `shape`: \([P, 2]\)
- `rotate`: \([P, 3, 3]\)
- `trans`: \([P, 3]\)
- `exist` or `exist_logit`: \([P, 1]\)
- `assign_matrix`: \([N, P]\)

with:

- \(N = 4096\)
- \(P = 16\)

---

## 2.3 Teacher-side post-processing

### 2.3.1 Rotation conversion

The generative model should not directly predict a flattened 3x3 rotation matrix.

Convert teacher rotations:

\[
R_j \in \mathbb{R}^{3 \times 3}
\]

to a **6D rotation representation**:

\[
r^{6d}_j \in \mathbb{R}^6
\]

using the first two columns of the rotation matrix:

\[
r^{6d}_j = \text{vec}(R_j[:,0:2])
\]

So for each primitive token we store rotation as 6 values.

---

### 2.3.2 Assignment mass

Compute primitive assignment mass:

\[
m_j = \frac{1}{N} \sum_{i=1}^{N} M_{ij}
\]

where:

- \(M \in \mathbb{R}^{N \times P}\)
- \(M_{ij}\) is soft assignment of point \(i\) to primitive \(j\)

Output:

\[
m \in \mathbb{R}^{P}
\]

This is used only for deterministic ordering and optional dataset diagnostics.

---

### 2.3.3 Primitive volume

Approximate primitive volume for ordering:

\[
V_j = s_{x,j} \cdot s_{y,j} \cdot s_{z,j}
\]

Output:

\[
V \in \mathbb{R}^{P}
\]

This does not need to be physically exact. It is only a tie-breaker.

---

### 2.3.4 Deterministic sorting

For each object, sort the 16 primitive slots using the following lexicographic key:

1. existence probability descending
2. assignment mass descending
3. volume descending
4. translation x ascending

After sorting, all primitive-dependent tensors must be reordered consistently:

- `scale`
- `shape`
- `rot6d`
- `trans`
- `exist_logit`
- `exist`
- `assign_matrix` columns
- `mass`
- `volume`

Important: this is done **offline** and stored in the training dataset.

---

## 2.4 Final Phase 1 token format

Each primitive token has dimension:

\[
d_E = 3 + 2 + 3 + 6 + 1 = 15
\]

Define:

\[
e_j = [s_j, \epsilon_j, t_j, r^{6d}_j, a_j] \in \mathbb{R}^{15}
\]

Stack all ordered primitive tokens:

\[
E \in \mathbb{R}^{P \times 15}
\]

with fixed:

\[
P = 16
\]

For batched training:

\[
E \in \mathbb{R}^{B \times 16 \times 15}
\]

---

## 2.5 Dataset file structure

Each offline saved example should contain at least:

- `points`: \([4096, 3]\)
- `tokens_e`: ordered explicit scaffold \([16, 15]\)
- `exist`: \([16, 1]\)
- `mass`: \([16]\)
- `volume`: \([16]\)
- `category_id`: scalar or string
- `model_id`: ShapeNet model identifier

Optional but recommended:

- unsorted teacher outputs
- active primitive count
- teacher visualization preview

---

## 3. Normalization of token channels

Do **not** train directly on raw mixed-scale parameters.

Different token dimensions have very different ranges:

- scales are bounded positive values
- exponents are bounded in a narrow interval
- translations may be centered around 0 but category-dependent
- 6D rotation values lie roughly in \([-1,1]\)
- existence logits may have large magnitude

So create a normalized training representation.

### 3.1 Per-channel normalization

Compute training-set statistics over all training objects and all primitive slots:

\[
\mu \in \mathbb{R}^{15}, \quad \sigma \in \mathbb{R}^{15}
\]

Then normalize:

\[
\tilde{E} = (E - \mu) / \sigma
\]

with safe clamp on \(\sigma\):

\[
\sigma \leftarrow \max(\sigma, 10^{-6})
\]

Training is done in normalized token space:

\[
\tilde{E} \in \mathbb{R}^{B \times 16 \times 15}
\]

Inference output is unnormalized before post-processing.

---

## 4. Flow matching formulation

## 4.1 Data variable

Let the clean ordered normalized scaffold be:

\[
\tilde{E}_0 \in \mathbb{R}^{B \times P \times d_E}
\]

with:

- \(P = 16\)
- \(d_E = 15\)

Sample noise:

\[
\tilde{E}_1 \sim \mathcal{N}(0, I)
\]

with the same shape.

Sample time:

\[
t \sim \mathcal{U}(0,1), \quad t \in \mathbb{R}^{B}
\]

Broadcast it to token dimension when needed.

---

## 4.2 Interpolation path

Use the rectified-flow linear interpolation path:

\[
\tilde{E}_t = (1 - t)\tilde{E}_0 + t\tilde{E}_1
\]

Implementation shape details:

- `E0`: \([B, 16, 15]\)
- `E1`: \([B, 16, 15]\)
- `t`: \([B]\)
- reshape `t` to \([B, 1, 1]\)
- `Et`: \([B, 16, 15]\)

---

## 4.3 Target velocity

For this linear path, the target velocity is:

\[
v^* = \tilde{E}_1 - \tilde{E}_0
\]

Shape:

\[
v^* \in \mathbb{R}^{B \times 16 \times 15}
\]

The model predicts:

\[
\hat{v}_\theta(\tilde{E}_t, t)
\]

with the same shape.

---

## 5. Model architecture

We use a **token-wise Set Transformer flow model**.

Input:

\[
\tilde{E}_t \in \mathbb{R}^{B \times 16 \times 15}
\]

Conditioning:

- scalar time \(t\)
- optional learnable global object token

Output:

\[
\hat{v}_\theta \in \mathbb{R}^{B \times 16 \times 15}
\]

---

## 5.1 High-level module graph

```text
Et [B,16,15]
  -> TokenProjection
  -> TokenEmbeddings [B,16,H]
  -> Add Time Conditioning
  -> Prepend Global Token
  -> L x SetTransformerBlock
  -> Drop Global Token
  -> VelocityHead
  -> v_hat [B,16,15]
```

Recommended default hidden size:

- \(H = 256\)
- number of blocks: 6
- heads per block: 8
- MLP expansion: 4H

---

## 5.2 Submodule: TokenProjection

### Purpose
Project each 15D scaffold token into model dimension.

### Input

\[
\tilde{E}_t \in \mathbb{R}^{B \times 16 \times 15}
\]

### Module

A small MLP:

```text
Linear(15 -> H)
LayerNorm(H)
SiLU
Linear(H -> H)
```

### Output

\[
X_0 \in \mathbb{R}^{B \times 16 \times H}
\]

Default:

\[
X_0 \in \mathbb{R}^{B \times 16 \times 256}
\]

---

## 5.3 Submodule: TimeEmbedding

### Purpose
Inject diffusion/flow time \(t\) into every primitive token.

### Input

\[
t \in \mathbb{R}^{B}
\]

### Module

Use sinusoidal time embedding followed by MLP:

```text
SinusoidalEmbedding(t) -> [B, T]
Linear(T -> H)
SiLU
Linear(H -> H)
```

Choose:

- `T = 128`
- final output dimension = `H`

### Output

\[
e_t \in \mathbb{R}^{B \times H}
\]

Broadcast to tokens:

\[
e_t^{tok} \in \mathbb{R}^{B \times 16 \times H}
\]

Then add:

\[
X_1 = X_0 + e_t^{tok}
\]

Output shape:

\[
X_1 \in \mathbb{R}^{B \times 16 \times H}
\]

---

## 5.4 Submodule: GlobalObjectToken

### Purpose
Provide one shared latent slot that aggregates global structure.

### Parameter

A learnable token:

\[
g \in \mathbb{R}^{H}
\]

### Expansion
Repeat for the batch:

\[
G \in \mathbb{R}^{B \times 1 \times H}
\]

Concatenate with token embeddings:

\[
X_2 = [G; X_1] \in \mathbb{R}^{B \times 17 \times H}
\]

This token participates in self-attention like any other token.

---

## 5.5 Submodule: SetTransformerBlock

### Purpose
Model interactions among primitive tokens.

### Input

\[
X \in \mathbb{R}^{B \times 17 \times H}
\]

### Structure per block

Each block contains:

1. Multi-head self-attention over all 17 tokens
2. Residual connection
3. LayerNorm
4. Feed-forward MLP
5. Residual connection
6. LayerNorm

Exact block:

```text
X -> MHA(X,X,X)
  -> residual add
  -> LayerNorm
  -> MLP(H -> 4H -> H)
  -> residual add
  -> LayerNorm
```

Recommended:

- 8 attention heads
- 6 blocks total
- dropout 0.0 or 0.1

### Output

After one block:

\[
X' \in \mathbb{R}^{B \times 17 \times H}
\]

After all \(L\) blocks:

\[
X_L \in \mathbb{R}^{B \times 17 \times H}
\]

---

## 5.6 Submodule: VelocityHead

### Purpose
Predict the velocity for each primitive token.

### Input
Drop the global token:

\[
X_{prim} = X_L[:, 1:, :] \in \mathbb{R}^{B \times 16 \times H}
\]

### Module

```text
Linear(H -> H)
SiLU
Linear(H -> 15)
```

### Output

\[
\hat{v}_\theta \in \mathbb{R}^{B \times 16 \times 15}
\]

This matches the target velocity shape exactly.

---

## 6. Optional architecture variants

These are allowed as ablations, but not the default.

### Variant A — no global token
Remove the global token and use only 16 primitive tokens.

### Variant B — shallower network
Use 4 blocks instead of 6.

### Variant C — larger hidden size
Use \(H=384\) instead of \(H=256\).

### Variant D — token dropout
Randomly drop a small fraction of tokens during training as regularization.

For the first implementation, keep the default architecture fixed.

---

## 7. Losses

## 7.1 Main loss: flow matching velocity regression

The main objective is mean squared error between predicted and target velocities.

### Input

- predicted velocity: \(\hat{v}_\theta \in \mathbb{R}^{B \times 16 \times 15}\)
- target velocity: \(v^* \in \mathbb{R}^{B \times 16 \times 15}\)

### Loss

\[
\mathcal{L}_{flow} = \frac{1}{B P d_E} \|\hat{v}_\theta - v^*\|_2^2
\]

This is the primary training loss.

---

## 7.2 Auxiliary loss: existence consistency

Although existence is just one channel of the token, it is useful to stabilize it with a separate auxiliary term.

### Extract

From clean teacher tokens:

- teacher existence logit: \(a_j\)
- teacher existence probability: \(\alpha_j = \sigma(a_j)\)

Define binary active label:

\[
y_j = \mathbb{1}[\alpha_j > 0.5]
\]

From predicted clean estimate, recover an estimate of \(\hat{E}_0\) from the current noisy state and predicted velocity:

\[
\hat{E}_0 = \tilde{E}_t - t \hat{v}_\theta
\]

Take the existence-logit channel from \(\hat{E}_0\), unnormalize that channel only, and apply BCE-with-logits against \(y_j\).

### Shape

- predicted existence logits: \([B, 16]\)
- binary targets: \([B, 16]\)

### Loss

\[
\mathcal{L}_{exist}
\]

Use a small weight only.

Recommended:

- \(\lambda_{exist} = 0.05\)

---

## 7.3 Optional auxiliary loss: ordering robustness loss

This is optional and can be skipped initially.

Goal: reduce overfitting to fragile tie-break behavior.

Approach:

- duplicate a batch example with tiny perturbation to low-priority sorting features
- enforce similar predicted denoised scaffolds

For v1, do **not** implement this. Keep deterministic ordering fixed and simple.

---

## 7.4 Total loss

Default total loss:

\[
\mathcal{L} = \mathcal{L}_{flow} + \lambda_{exist} \mathcal{L}_{exist}
\]

with:

- \(\lambda_{exist} = 0.05\)

If simplicity is preferred, train initially with only:

\[
\mathcal{L} = \mathcal{L}_{flow}
\]

and add `L_exist` only if existence quality is unstable.

---

## 8. Training loop

## 8.1 Batch contents

Each dataloader batch should return at least:

- `tokens_e`: \([B,16,15]\)
- optionally `points`: \([B,4096,3]\)
- optionally `exist`: \([B,16,1]\)
- metadata for debugging

---

## 8.2 Forward training pass

### Step 1
Load normalized clean scaffold:

\[
\tilde{E}_0 \in \mathbb{R}^{B \times 16 \times 15}
\]

### Step 2
Sample Gaussian noise:

\[
\tilde{E}_1 \in \mathbb{R}^{B \times 16 \times 15}
\]

### Step 3
Sample times:

\[
t \in \mathbb{R}^{B}
\]

### Step 4
Construct interpolated tokens:

\[
\tilde{E}_t = (1-t)\tilde{E}_0 + t\tilde{E}_1
\]

### Step 5
Compute target velocity:

\[
v^* = \tilde{E}_1 - \tilde{E}_0
\]

### Step 6
Run model:

\[
\hat{v}_\theta = f_\theta(\tilde{E}_t, t)
\]

### Step 7
Compute loss.

### Step 8
Backprop and optimizer step.

---

## 8.3 Optimizer and schedule

Recommended starting configuration:

- optimizer: AdamW
- learning rate: `1e-4`
- weight decay: `1e-4`
- batch size: 64 if memory allows
- epochs: 300
- gradient clipping: `1.0`

Optional:

- cosine LR schedule with warmup
- EMA of model weights for sampling

---

## 9. Inference and sampling

## 9.1 Sampling input

Sample initial noise:

\[
\tilde{E}_{t=1} \sim \mathcal{N}(0, I)
\]

Shape:

\[
\tilde{E}_{1} \in \mathbb{R}^{B \times 16 \times 15}
\]

---

## 9.2 ODE integration

Use Euler integration for the first version.

Let the time grid be:

\[
1 = t_0 > t_1 > \dots > t_T = 0
\]

For each step:

\[
\tilde{E}_{t_{k+1}} = \tilde{E}_{t_k} - \hat{v}_\theta(\tilde{E}_{t_k}, t_k) \cdot (t_k - t_{k+1})
\]

with:

- `T = 50` steps as a reasonable starting point

Output:

\[
\hat{\tilde{E}}_0 \in \mathbb{R}^{B \times 16 \times 15}
\]

Then unnormalize to get:

\[
\hat{E}_0 \in \mathbb{R}^{B \times 16 \times 15}
\]

---

## 9.3 Post-processing generated tokens

### Step 1 — split channels
For each primitive token, split into:

- scale: 3
- shape: 2
- translation: 3
- rot6d: 6
- exist_logit: 1

### Step 2 — convert rotation
Convert 6D rotation to matrix using Gram-Schmidt orthonormalization.

Output:

\[
R \in \mathbb{R}^{B \times 16 \times 3 \times 3}
\]

### Step 3 — threshold existence
Compute:

\[
\alpha = \sigma(a)
\]

Mark primitive active if:

\[
\alpha > 0.5
\]

### Step 4 — optional parameter clamping
For visualization safety only:

- scale clamp to positive minimum, e.g. `1e-3`
- exponent clamp to `[0.1, 2.0]`

If training normalization is correct, this should be rarely needed.

---

## 10. Phase 1 evaluation hooks

Even though this file is implementation-focused, the code should expose the following diagnostics.

## 10.1 Active primitive count

For a sampled batch:

\[
K_b = \sum_{j=1}^{16} \mathbb{1}[\sigma(a_{b,j}) > 0.5]
\]

Report mean and histogram.

---

## 10.2 Token statistics

For generated scaffolds, log per-channel means and variances for:

- scales
- exponents
- translations
- existence probabilities

Compare against the teacher training-set distribution.

---

## 10.3 Scaffold rendering

Implement a simple SQ renderer that turns generated tokens into sampled SQ surface points for visualization.

Per primitive:

- use scales + exponents + pose
- sample a fixed number of surface points, e.g. 256
- merge active primitives

Output visualization point cloud shape per object:

\[
X_{sq} \in \mathbb{R}^{K \cdot 256 \times 3}
\]

where \(K\) is the number of active primitives.

---

## 11. Recommended code modules

Suggested module split.

### `phase1/data/build_teacher_dataset.py`
Responsibilities:

- load ShapeNet chairs
- run frozen SuperDec
- compute mass and volume
- convert rotation to 6D
- sort primitives deterministically
- save `.pt` or `.npz` files

### `phase1/data/dataset.py`
Responsibilities:

- load saved teacher scaffolds
- apply normalization
- return `tokens_e`

### `phase1/models/time_embedding.py`
Responsibilities:

- sinusoidal time embedding
- MLP projection to hidden dim

### `phase1/models/set_transformer_flow.py`
Responsibilities:

- token projection
- global token
- transformer blocks
- velocity head

### `phase1/models/rotation.py`
Responsibilities:

- matrix to 6D
- 6D to matrix

### `phase1/losses/flow_matching.py`
Responsibilities:

- path construction
- target velocity
- flow loss
- optional existence auxiliary

### `phase1/train.py`
Responsibilities:

- optimizer
- schedule
- checkpointing
- validation sampling

### `phase1/sample.py`
Responsibilities:

- Euler sampler
- token unnormalization
- post-processing
- visualization export

---

## 12. Minimal tensor contract

This is the exact tensor contract the implementation should respect.

### Dataset output

- `tokens_e`: \([B,16,15]\)

### Flow training input

- `E0`: \([B,16,15]\)
- `E1`: \([B,16,15]\)
- `t`: \([B]\)
- `Et`: \([B,16,15]\)
- `v_target`: \([B,16,15]\)

### Model forward

Input:

- `Et`: \([B,16,15]\)
- `t`: \([B]\)

Output:

- `v_hat`: \([B,16,15]\)

### Sampling output

- generated tokens: \([B,16,15]\)
- generated rotation matrices: \([B,16,3,3]\)
- generated active mask: \([B,16]\)

---

## 13. Recommended default hyperparameters

```text
P = 16
D_token = 15
H = 256
L = 6
n_heads = 8
ff_mult = 4
batch_size = 64
lr = 1e-4
weight_decay = 1e-4
epochs = 300
n_sampling_steps = 50
exist_threshold = 0.5
lambda_exist = 0.05
```

---

## 14. First implementation milestone

The first working milestone is reached when the code can:

1. build a teacher scaffold dataset from ShapeNet chairs
2. train the Set Transformer flow model on ordered 15D scaffold tokens
3. sample new scaffold token sets from noise
4. convert them into superquadric assemblies
5. render plausible coarse chair structures

That is the complete implementation target for Phase 1.
