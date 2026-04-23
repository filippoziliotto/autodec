# PHASE 1 — Unconditional Object Scaffold Generation in Superquadric Space

## Working title
**SQ-Flow Phase 1: Unconditional Superquadric Scaffold Generation for 3D Objects**

---

## 1. Goal

Phase 1 learns a generative prior over the **explicit superquadric scaffold** of an object.

Given only noise, the model should generate a **set of superquadric primitives** that defines a plausible object-level structure. For this first stage, the output is **not** dense geometry and it is **not** texture. The output is a structured object scaffold:

- primitive scales
- primitive shape exponents
- primitive translations
- primitive rotations
- primitive existence logits

For this project version, we use:

- **deterministic primitive ordering**
- **token-wise Set Transformer flow model**
- **frozen SuperDec teacher outputs**
- **single-category ShapeNet training** (chairs only)
- **4096-point canonical inputs**

---

## 2. Why this phase exists

SuperDec can decompose a point cloud into a compact set of superquadrics, but it is not a generative model. It maps:

\[
X \rightarrow E
\]

where:

- \(X\) is the input point cloud
- \(E\) is the explicit superquadric scaffold

Phase 1 adds the missing generative prior:

\[
\epsilon \rightarrow \hat{E}
\]

This means we learn to sample plausible object structures **directly in superquadric space**.

The resulting scaffold is:

- compact
- interpretable
- editable
- suitable as input to later stages

This is the cleanest first generative step because it focuses only on **object structure**, without mixing in residual geometric detail.

---

## 3. Scope of Phase 1

### What Phase 1 does

Phase 1 generates the **explicit scaffold only**.

Each generated object is represented by a fixed number of primitive slots \(P\), where each slot contains:

- scale: 3 values
- shape exponents: 2 values
- translation: 3 values
- rotation: 6 values (continuous 6D rotation representation)
- existence logit: 1 value

So each primitive token has dimension:

\[
d_E = 3 + 2 + 3 + 6 + 1 = 15
\]

The full scaffold for one object is:

\[
E \in \mathbb{R}^{P \times 15}
\]

with fixed \(P = 16\).

### What Phase 1 does **not** do

Phase 1 does **not**:

- generate residual latent codes
- generate dense point clouds directly
- generate textures or appearance
- optimize reconstruction through the AutoDec decoder
- model multi-category object distributions

All of that is deferred to later phases.

---

## 4. Representation used in Phase 1

We model each object as an ordered set of \(P = 16\) primitive tokens.

Each token is:

\[
e_j = [s_j, \epsilon_j, t_j, r_j, a_j] \in \mathbb{R}^{15}
\]

where:

- \(s_j \in \mathbb{R}^3\): scale
- \(\epsilon_j \in \mathbb{R}^2\): SQ exponents
- \(t_j \in \mathbb{R}^3\): translation
- \(r_j \in \mathbb{R}^6\): 6D rotation representation
- \(a_j \in \mathbb{R}\): existence logit

The complete object scaffold is:

\[
E = [e_1, \dots, e_P] \in \mathbb{R}^{16 \times 15}
\]

At inference time, primitives with low existence are removed by thresholding the existence probability:

\[
\alpha_j = \sigma(a_j)
\]

---

## 5. Deterministic ordering

A major issue in primitive-based generation is that the scaffold is naturally a **set**, not a sequence. To train a simple fixed-slot generative model, we impose a deterministic ordering on the teacher scaffolds.

For every training object, the teacher output primitives are sorted using the following priority:

1. **existence probability descending**
2. **assignment mass descending**
3. **primitive volume descending**
4. **translation x-coordinate ascending** as final tie-breaker

This gives a stable canonical slot order for training.

Important:

- this ordering is a training convention, not a semantic part ordering
- slot 3 is not guaranteed to mean the same semantic part across all chairs
- the purpose is only to reduce permutation ambiguity enough for Phase 1 training

---

## 6. Data source for Phase 1

The Phase 1 training data is built offline using a **frozen SuperDec teacher**.

### Input data

We use:

- **ShapeNet chairs only**
- canonical object orientation from the dataset split/preprocessing
- **4096 input points per object**

### Offline teacher extraction

For each ShapeNet chair:

1. load the canonical object point cloud \(X \in \mathbb{R}^{4096 \times 3}\)
2. run the frozen SuperDec model
3. extract primitive parameters and assignment matrix
4. convert rotation to 6D representation
5. compute auxiliary values for ordering (assignment mass, volume)
6. sort primitives deterministically
7. save the ordered scaffold target

So the final training dataset is not raw point clouds alone. It is a dataset of teacher scaffolds:

\[
\mathcal{D}_{\text{phase1}} = \{E_i\}_{i=1}^{N_{train}}
\]

Optionally, for debugging and evaluation, also save:

- original point cloud \(X_i\)
- original unsorted teacher outputs
- active primitive count
- assignment mass per primitive

---

## 7. Model idea

Phase 1 uses a **token-wise Set Transformer flow model**.

The model takes a noisy scaffold token set at time \(t\):

\[
E_t \in \mathbb{R}^{P \times 15}
\]

and predicts the velocity field that moves the noisy tokens toward the clean teacher scaffold \(E_0\).

The model is:

- token-based, not point-based
- permutation-aware through self-attention across primitive tokens
- globally conditioned through a learnable object token
- trained with a standard flow matching objective

This is the right architecture because the object scaffold is a **small interacting set of parts**.

---

## 8. Training objective at a high level

For a clean scaffold \(E_0\), sample noise \(E_1 \sim \mathcal{N}(0, I)\). Then interpolate:

\[
E_t = (1 - t) E_0 + t E_1
\]

with \(t \sim \mathcal{U}(0,1)\).

The target velocity is:

\[
v^*(E_t, t) = E_1 - E_0
\]

The network predicts:

\[
\hat{v}_\theta(E_t, t)
\]

and is trained with mean squared error.

So the model learns how to transport noise into valid superquadric object scaffolds.

---

## 9. Output of Phase 1

At inference:

1. sample Gaussian noise in token space
2. integrate the learned flow from \(t=1\) to \(t=0\)
3. obtain a generated scaffold \(\hat{E}\)
4. convert 6D rotations to matrices if needed
5. threshold primitives by existence
6. visualize the resulting SQ object

The immediate output is a generated superquadric chair scaffold.

This scaffold can later be:

- rendered directly as a superquadric assembly
- decoded with \(Z=0\) through the AutoDec decoder for coarse geometry
- used as the conditioning scaffold for Phase 2 residual generation

---

## 10. Success criteria

Phase 1 is successful if it can generate scaffolds that are:

- valid (reasonable parameter ranges)
- sparse (roughly realistic active primitive counts)
- diverse
- structurally plausible for the target class
- editable at primitive level
- decodable into recognizable coarse chairs when using a scaffold-only decoder path

The key claim is:

> From pure noise, the model can generate compact and interpretable object scaffolds in superquadric space.

---

## 11. Why this is the right first stage

Phase 1 isolates the hardest structural question:

**Can we model the distribution of object part layouts directly in explicit geometric primitive space?**

This stage is deliberately narrow and clean:

- one category only
- no residuals yet
- no dense geometry prior yet
- no texture
- no conditioning

That makes it feasible, interpretable, and directly useful as the foundation for Phase 2.
