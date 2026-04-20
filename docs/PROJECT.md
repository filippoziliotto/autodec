# PROJECT: Structured 3D Autoencoder with Superquadric Bottleneck

> **Working title:** SuperDec-AE — Editable 3D Autoencoding with a Superquadric Bottleneck
>
> **Core claim:** We introduce a structured 3D autoencoding framework where explicit geometric primitives serve as the bottleneck, yielding representations that are compact, interpretable, editable, and reconstructively powerful.

---

## Notation

| Symbol | Meaning |
|---|---|
| N | Number of input points (4096) |
| X | Input point cloud, `X ∈ ℝ^{N×3}` |
| P | Maximum number of superquadric primitives (16) |
| P_act | Number of active primitives per object (~5–6) |
| S_dec | Number of surface samples per primitive for decoding (256) |
| S_sq | Number of surface samples per primitive for fitting loss |
| K_LM | Number of LM normalization samples (25 in SuperDec) |
| H | Feature dimension (128) |
| D | Number of Transformer decoder layers (3) |
| d | Residual latent dimension per primitive (64 default) |

---

## 1. Motivation and Problem Statement

### 1.1 What SuperDec does

SuperDec (Fedele et al., 2026) decomposes a 3D point cloud of an object into a compact set of superquadric primitives. Given an input point cloud `X ∈ ℝ^{N×3}` (N = 4096), it predicts ~5–6 superquadrics that approximate the object's geometry. Each superquadric is described by 11 parameters (3 scales + 2 shape exponents + 3 translation + 3 rotation) plus 1 existence score, totaling 12 numbers per primitive.

This mapping is **one-way**: point cloud → superquadrics.

### 1.2 What SuperDec does not do

SuperDec maps a point cloud to an explicit set of superquadric primitives, from which a coarse geometric approximation can be obtained by sampling the predicted surfaces. However, it does not learn a decoder that reconstructs high-fidelity dense geometry beyond this primitive approximation. As a result, thin structures, non-convex details, and high-frequency residual shape variation are not explicitly modeled.

### 1.3 What this project adds

We build a **3D autoencoder** whose bottleneck is the set of superquadric primitives, augmented by a small per-part residual latent. The encoder compresses a point cloud into interpretable geometric parts plus learned residual features. The decoder reconstructs a dense point cloud from both codes.

The pipeline is:

```
X ∈ ℝ^{N×3}  →  Encoder  →  (E ∈ ℝ^{P×12}, Z ∈ ℝ^{P×d})  →  Decoder  →  X̂ ∈ ℝ^{N'×3}
```

Unlike a standard autoencoder, the bottleneck is not a black box. Part of it (E) is a set of interpretable geometric primitives with known geometric meaning (scale, pose, shape exponents, existence), though not semantic meaning (e.g., "seat" or "leg") unless established by correspondence or manual inspection. Part of it (Z) is a small neural residual. Together, they reconstruct dense geometry.

### 1.4 Why this matters

| Capability | Standard AE | Pure SQ decomposition | **Ours** |
|---|---|---|---|
| Reconstruction quality | High | Low (lossy) | **High** |
| Interpretable latent | No | Yes | **Yes** |
| Editable (local, geometry-aware) | No | Yes (but no decoder) | **Yes** |
| Shape completion | Weak | N/A | **Promising (needs partial-input training)** |
| Compression | Moderate | Extreme | **Controllable** |

---

## 2. Background: Superquadrics

A superquadric in canonical pose is defined by 5 shape parameters. Its surface satisfies the implicit equation:

```
f(x) = ( |x/s_x|^(2/ε₂) + |y/s_y|^(2/ε₂) )^(ε₂/ε₁) + |z/s_z|^(2/ε₁) = 1
```

Note: absolute values are required for the expression to be well-defined over the reals when exponents are fractional.

Where:
- `(s_x, s_y, s_z)` — scales along principal semi-axes (3 params)
- `(ε₁, ε₂)` — shape exponents controlling roundness (2 params)
  - Both ≈ 1.0 → sphere
  - Both ≈ 0.1 → box
  - Intermediate → cylinder, pillow, octahedron

To place the superquadric in world coordinates: 3 translation + 3 rotation = 6 pose parameters.

**Total: 11 parameters per superquadric.**

We add 1 existence logit a ∈ ℝ (from which the existence probability α = σ(a) is derived), giving **12 parameters per primitive** (11 geometric + 1 existence logit).

**Rotation parameterization:** SuperDec uses 3 rotation parameters. The exact convention (Euler angles, axis-angle, or Rodrigues vector) must be verified from the codebase. If training from scratch, a continuous 6D rotation representation (Zhou et al., 2019) would be more stable, but since we initialize from pretrained SuperDec weights, we keep whatever parameterization the original code uses and document it precisely.

**Radial distance.** The radial distance from any point x ∈ ℝ³ to the superquadric surface has a closed-form expression:

```
d_r = |x_local| · |1 − f(x_local)^(−ε₁/2)|
```

where `x_local` is the point transformed into the **canonical coordinate frame** of the superquadric (i.e., after applying the inverse rotation and subtracting the translation). This is not a true Euclidean signed distance, but a radial approximation. It is used in SuperDec's LM optimization module (Eq. 10), **not** in the neural network training loss.

**Parametric surface sampling.** The surface of a superquadric can be sampled using the signed-power formulation:

```
c_ε(u) = sgn(cos u) · |cos u|^ε
s_ε(u) = sgn(sin u) · |sin u|^ε

p(η, ω) = R_j · [s_x · c_ε₁(η) · c_ε₂(ω),  s_y · c_ε₁(η) · s_ε₂(ω),  s_z · s_ε₁(η)]^T + t_j
```

where η ∈ [−π/2, π/2] and ω ∈ [−π, π]. The signed-power functions are required for correct behavior across all quadrants.

**Important:** Uniform sampling of (η, ω) does **not** produce uniform point density on the surface. SuperDec uses equal-distance sampling following Pilu et al. (1995). The decoder sampler should either use the same method or document and justify any cheaper approximation empirically.

---

## 3. Architecture

### 3.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  ENCODER (from SuperDec, pretrained)                            │
│                                                                 │
│  X ∈ ℝ^{N×3}                                                   │
│    ↓                                                            │
│  PVCNN encoder → F_PC ∈ ℝ^{N×H}                                │
│                                                                 │
│  SQ query tokens (sinusoidal init) → F⁰_SQ ∈ ℝ^{P×H}          │
│    ↓                                                            │
│  Transformer decoder (D=3 layers, self-attn + cross-attn)       │
│    → F_SQ ∈ ℝ^{P×H}                                            │
│    ↓                         ↓                    ↓             │
│  Seg head              Param head          Residual proj (NEW)  │
│  M ∈ ℝ^{N×P}          E ∈ ℝ^{P×12}       Z = MLP([F_SQ; pool(F_PC,M)]) │
│                                            Z ∈ ℝ^{P×d}          │
└─────────────────────────────────────────────────────────────────┘
                              ↓                    ↓
┌─────────────────────────────────────────────────────────────────┐
│  DECODER (new, trained from scratch)                            │
│                                                                 │
│  SQ surface sampler: E → X_sq ∈ ℝ^{(P·S_dec)×3} with soft w   │
│    ↓                                                            │
│  Offset network: (X_sq, E, Z) → Δ ∈ ℝ^{(P·S_dec)×3}           │
│    ↓                                                            │
│  X̂ = X_sq + w⊙Δ  (w gates offsets, not coordinates)            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Encoder (reused from SuperDec)

The encoder is the SuperDec architecture with one addition.

**Reused components (pretrained weights available):**

| Component | Input shape | Output shape | Parameters |
|---|---|---|---|
| PVCNN point encoder | X ∈ ℝ^{N×3} | F_PC ∈ ℝ^{N×H} | ~1.5M |
| SQ query initialization | — | F⁰_SQ ∈ ℝ^{P×H} | Sinusoidal, no learned params |
| Transformer decoder (D=3) | (F_PC, F⁰_SQ) | F_SQ ∈ ℝ^{P×H} | ~0.6M |
| Segmentation head | (F_PC, F_SQ) | M ∈ ℝ^{N×P} | Linear projection |
| Parameter head | F_SQ | E ∈ ℝ^{P×12} | Linear ℝ^H → ℝ^{12} |

The segmentation head computes:

```
M_ij = softmax_j( φ(F_PC) · F_SQ^T )    where φ: ℝ^H → ℝ^H
```

So `M ∈ ℝ^{N×P}` is a soft assignment matrix: each row sums to 1, assigning each point to superquadrics.

The parameter head is a linear layer predicting 12 values per query:

```
E_j = Linear(F_SQ_j)    for j = 1, ..., P
```

Where each row E_j contains: `[s_x, s_y, s_z, ε₁, ε₂, t_x, t_y, t_z, r₁, r₂, r₃, a_j]`, where the first 11 are geometric parameters and a_j is the existence logit (α_j = σ(a_j)).

**New component (trained from scratch):**

| Component | Input shape | Output shape | Parameters |
|---|---|---|---|
| Residual projection | (F_SQ ∈ ℝ^{P×H}, F_PC ∈ ℝ^{N×H}, M ∈ ℝ^{N×P}) | Z ∈ ℝ^{P×d} | ~40K params (41,152 for H=128, d=64) |

The residual code should encode the fine geometric detail that each superquadric misses. The primitive token F_SQ,j alone is optimized for fitting and segmentation, not for preserving high-frequency local detail. To address this, we first pool the local point features assigned to each primitive, then combine with the primitive token:

```
g_mean,j = (Σ_i m_ij · F_PC,i) / (Σ_i m_ij + ε)                 ∈ ℝ^H
g_max,j  = max_i masked_by(m_ij > ε, m_ij · F_PC,i)             ∈ ℝ^H
g_var,j  = (Σ_i m_ij · (F_PC,i − g_mean,j)^2) / (Σ_i m_ij + ε)  ∈ ℝ^H

Z_j = MLP([F_SQ,j; g_mean,j; g_max,j; g_var,j])                 ∈ ℝ^d
      (MLP: ℝ^{4H} → ℝ^H → ℝ^d)
```

This way each residual token is conditioned both on the primitive query state and on the actual local point features assigned to that primitive. Mean-only pooling was a bottleneck because it erased local feature variation; the implemented multi-statistic pooling preserves average evidence, strong feature responses, and local variance. The semicolon denotes concatenation, so the MLP input is ℝ^{4H} = ℝ^{512} for H=128.

**Design choice: per-part vs. global residual.** We use per-part residuals (one z_j per superquadric) rather than a single global z. This means each primitive carries its own fine-detail code, enabling part-level editing: you can modify one superquadric and its associated z_j without affecting other parts.

### 3.3 Decoder (new, trained from scratch)

The decoder has three stages: sampling, feature construction, and offset prediction.

#### Stage 1: SQ Surface Sampler (no learned parameters)

For each superquadric j, sample S_dec = 256 points on its surface using the signed-power parametric equation (see Section 2) with equal-distance sampling following Pilu et al. (1995).

**Soft gating (critical for training).** During training, we do **not** hard-threshold existence to mask out primitives. Instead, we derive a differentiable soft weight per primitive.

The parameter head predicts a raw logit `a_j ∈ ℝ` (not a probability). From this we compute:

```
α_j = σ(a_j)             ← existence probability used in L_exist
w_j = σ(a_j / τ)         ← soft gate with temperature τ (sharper than α_j)
```

**Important:** The soft weight w_j does **not** multiply the sampled 3D coordinates. Multiplying coordinates by w_j would shrink points toward the origin, distorting the geometry. Instead, w_j is used in two ways:

1. **Feature/offset gating:** The predicted offset Δ_i for points from primitive j is multiplied by w_j, so inactive primitives produce zero displacement but their scaffold points remain geometrically valid.
2. **Loss weighting:** The forward reconstruction term uses w_j as an importance weight (see Section 4.1).

Hard thresholding (keeping only primitives with α_j > threshold) is applied only at inference time.

**Adaptive sampling (optional improvement).** Allocating S_dec points uniformly per primitive is simple but geometrically biased: a tiny chair leg and a large tabletop both get 256 points. A better alternative is to distribute a fixed total point budget proportionally to the predicted assignment mass m̄_j or estimated surface area. For v1, we start with uniform allocation and revisit if reconstruction quality is uneven across parts.

```
Input:  E ∈ ℝ^{P×12}
Output: X_sq ∈ ℝ^{(P·S_dec)×3}       (coarse point cloud on SQ surfaces)
        w ∈ [0,1]^{P·S_dec}           (soft existence weights per point)
        part_ids ∈ {1,...,P}^{P·S_dec} (which SQ each point belongs to)
```

After soft gating, the effective output can be thought of as N' ≤ P·S_dec active points. This stage is fully differentiable with respect to the SQ parameters in E.

#### Stage 2: Feature Construction (learned split projections)

For each surface point x_sq,i ∈ X_sq, construct a feature vector by concatenating:

```
γ(p_i) = [p_i, sin(2^kπp_i), cos(2^kπp_i)]_{k=0}^{L−1}

f_i = [
  Proj_pos(γ(p_i)),
  Proj_E(E_j),
  Proj_Z(z_j),
  Proj_gate(w_i)
]
```

Where j = part_ids[i] is the superquadric that point i was sampled from.

```
Input:  X_sq ∈ ℝ^{(P·S_dec)×3}, E_dec ∈ ℝ^{P×18}, Z ∈ ℝ^{P×d}, w, part_ids
Output: F_dec ∈ ℝ^{(P·S_dec)×(4·C)}
```

`E_dec` uses the actual decoder packing: scale (3), shape (2), translation (3), flattened rotation matrix (9), and existence logit (1), for 18 values. The split projectors use `LayerNorm → Linear → ReLU` for position/Fourier features, E_dec, and Z, and `Linear → ReLU` for the gate. By default `C = max(4, hidden_dim // 4)`; setting `component_feature_dim=0` disables this and recovers the older raw concatenation path.

#### Stage 3: Offset Network (learned, the main decoder component)

This is the core new module. It takes the per-point features and predicts a 3D offset for each point.

**Architecture options (in order of complexity):**

**Option A — Simple MLP (starting point):**

```
F_dec ∈ ℝ^{(P·S_dec)×79}  →  MLP(79 → 256 → 256 → 3)  →  Δ ∈ ℝ^{(P·S_dec)×3}
```

Each point is processed independently. No cross-part communication. ~200K parameters.

Pros: Simple, fast, easy to debug.
Cons: Parts don't know about each other; may produce gaps or overlaps at part boundaries.

**Option B — stacked self-attention + cross-attention (implemented default):**

```
F_dec → Linear(4C → H)                    → point stream
T     → Linear(2C → H)                    → primitive tokens

repeat K times:
  within-primitive self-attention(points) → residual + LayerNorm
  cross-attention(points → primitives)    → residual + LayerNorm
  FFN(H → 4H → H)                         → residual + LayerNorm

MLP(H → H → 3) → Δ_raw ∈ ℝ^{(P·S_dec)×3}
```

Each surface point attends to all P primitive tokens, which concatenate projected explicit parameters E_j and residual features Z_j. The implemented default also applies self-attention within each primitive's S_dec sampled points. This gives local smoothing/coordination without the O((P·S_dec)^2) cost of global surface-point self-attention.

~500K parameters. This is the recommended architecture.

**Option C — Transformer decoder (most expressive):**

Multiple layers of self-attention among surface points + cross-attention to primitive tokens. Most expressive but expensive (O(N²) self-attention on P·S_dec points). Consider only if Option B underperforms.

#### Stage 4: Output

```
X̂ = X_sq + w ⊙ Δ    ∈ ℝ^{(P·S_dec)×3}
```

Where ⊙ denotes element-wise multiplication of each offset by its primitive's soft existence weight (broadcast across xyz). The final point cloud is the coarse SQ surface displaced by the gated offsets.

**Offset magnitude bounding.** The implementation supports the scale-aware cap

```
Δ = tanh(Δ_raw) · offset_cap · mean(scale_j)
```

repeated to each surface point from primitive j. The default AutoDec configs use `offset_cap: 0.3`; setting `offset_cap: null` restores the older unbounded behavior. This is preferred over the legacy scalar `offset_scale` because it prevents residual collapse while preserving scale-aware flexibility.

**Cardinality conventions:**
- **Training:** Always produce a fixed-size tensor X̂ ∈ ℝ^{(P·S_dec)×3} = ℝ^{4096×3}. Soft existence weights modulate the loss, not the tensor size. This avoids variable-length complications during batched training.
- **Inference / visualization:** Hard-threshold inactive primitives (α_j < threshold), remove their points, producing X̂ ∈ ℝ^{N'×3} with N' ≤ P·S_dec.
- **Evaluation:** Evaluate on the pruned output. If baselines use a fixed point count (e.g., 4096), resample the pruned output to match for fair comparison.

**Point density note.** Chamfer distance alone can tolerate clustered outputs. Since the decoder predicts offsets from sampled surfaces, it might fold or clump points locally. A light repulsion loss or local density regularizer may improve reconstruction quality. This is not mandatory for v1 but is worth remembering.

### 3.4 What we reuse vs. what we build

| Component | Source | Pretrained? | Trainable? |
|---|---|---|---|
| PVCNN encoder | SuperDec | ✅ Yes | Fine-tuned in Phase 2 |
| Transformer decoder | SuperDec | ✅ Yes | Fine-tuned in Phase 2 |
| Segmentation head | SuperDec | ✅ Yes | Fine-tuned in Phase 2 |
| Parameter head | SuperDec | ✅ Yes | Fine-tuned in Phase 2 |
| Residual projection | **New** | ❌ No | Trained from scratch (multi-stat pooling + MLP: ℝ^{4H} → ℝ^H → ℝ^d) |
| SQ surface sampler | **New** | N/A (no params) | N/A |
| Offset network | **New** | ❌ No | Trained from scratch |

### 3.5 Bottleneck size analysis

We report **both** the fixed-slot budget (all P=16 slots) and the active-only budget (serializing only active primitives).

**Fixed-slot budget (inference tensor size):**

| Code | Size per slot | Total (P=16) |
|---|---|---|
| Explicit E | 12 floats | 192 floats = 768 bytes |
| Residual Z | 64 floats | 1,024 floats = 4,096 bytes |
| **Total** | 76 floats | **1,216 floats = 4,864 bytes** |

**Active-only budget (serialized with P_act ≈ 6 + mask/order):**

| Code | Size per primitive | Total (P_act=6) |
|---|---|---|
| Explicit E | 12 floats | 72 floats = 288 bytes |
| Residual Z | 64 floats | 384 floats = 1,536 bytes |
| + Active mask | — | ~2 bytes |
| **Total** | 76 floats | **~456 floats = ~1,826 bytes** |

For comparison:
- Original point cloud: N × 3 = 12,288 floats = 49,152 bytes
- Standard AE latent (e.g., FoldingNet): 512 floats = 2,048 bytes (opaque)
- **Fixed-slot compression ratio: ~10×**
- **Active-only compression ratio: ~27×**

The reported compression depends on whether we serialize only active slots. Both numbers must be stated honestly.

---

## 4. Loss Functions

### 4.1 Reconstruction loss (L_recon)

Bidirectional Chamfer-L2 distance between the decoded and original point clouds, with soft existence weights.

During training, the decoder always produces a fixed-size output tensor X̂ ∈ ℝ^{(P·S_dec)×3} with associated weights w_i = w_{part_ids[i]} from the soft existence gate. The weights modulate the forward term but do not alter the coordinates:

```
Inputs:  X̂ ∈ ℝ^{(P·S_dec)×3},  X ∈ ℝ^{N×3},  w ∈ [0,1]^{P·S_dec}
Output:  scalar

Forward (weighted by existence):
  L_fwd = (1 / Σ_i w_i) Σ_i  w_i · min_k ‖x̂_i − x_k‖²

Backward (also weighted, to prevent inactive scaffold leakage):
  L_bwd = (1/N) Σ_k  min_i  (‖x_k − x̂_i‖² / (w_i + ε))

L_recon = L_fwd + L_bwd
```

Both terms are weight-aware. In the backward term, dividing by (w_i + ε) penalizes nearest-neighbor matches to low-weight (inactive) points, preventing them from cheaply satisfying coverage. Without this, inactive scaffold points could explain input points during training but would be pruned at inference, creating a train-test mismatch. An alternative formulation is to restrict the backward nearest-neighbor search to points with w_i above a soft threshold.

### 4.2 Superquadric fitting loss (L_sq)

We use a primitive fitting loss on the explicit branch to ensure that the superquadrics themselves remain geometrically meaningful. Concretely, for each predicted primitive we sample S_sq points on its surface and optimize a bidirectional loss between the input point cloud and the sampled primitive surfaces, weighted by the soft assignment matrix M. This follows the same structure as SuperDec's geometric supervision (Eq. 5-7) but uses squared Euclidean distances (Chamfer-L2) rather than the unsquared Euclidean distances in the original formulation. We use squared distances for consistency with our main L_recon objective.

```
Inputs:  X ∈ ℝ^{N×3},  E ∈ ℝ^{P×12},  M ∈ ℝ^{N×P}
         α_j = σ(a_j) derived from the existence logits in E
Output:  scalar

Let x'_js denote the s-th point sampled on the surface of the j-th superquadric,
and d(x_i, x'_js) the Euclidean distance between input point i and surface sample s.

Forward (X → SQ):
  L_{X→SQ} = (1/N) Σ_{i=1}^{N} Σ_{j=1}^{P}  m_ij · min_{s ∈ [S_sq]}  ‖x_i − x'_js‖²

Reverse (SQ → X):
  L_{SQ→X} = (1 / (S_sq · Σ_j α_j)) Σ_{j=1}^{P} α_j Σ_{s=1}^{S_sq}  min_{i ∈ [N]}  ‖x_i − x'_js‖²

L_sq = L_{X→SQ} + L_{SQ→X}
```

The full inherited SuperDec reconstruction loss is L_rec = L_{X→SQ} + L_{SQ→X} + L_N, where L_N is a normal-alignment term (Eq. 5 in SuperDec). **Decision:** For the first prototype (Phases 1–2), we implement L_sq = L_{X→SQ} + L_{SQ→X} only, omitting L_N to reduce complexity. For the paper-ready version, add L_N and report an ablation, since ShapeNet meshes provide normals and SuperDec uses L_N for convergence. A reviewer may otherwise note that we are not fully leveraging the original fitting signal.

**This loss is critical.** Without it, the model can learn trivial superquadrics and store everything in Z, collapsing into a standard autoencoder. L_sq forces the explicit branch to carry real geometric content independently of the decoder.

**Note on radial distance.** SuperDec's LM optimization module uses radial distance (Eq. 2, 10) rather than Euclidean distance for its residuals. The radial distance is cheaper to compute and does not require surface sampling. If desired, a radial-distance fitting term can be added as an **auxiliary loss** alongside the sampled-surface Chamfer, but it should not be conflated with the SuperDec neural training loss, which uses Euclidean distances to sampled surface points.

### 4.3 Parsimony loss (L_par)

Encourages the model to use fewer primitives. Inherited from SuperDec (Eq. 8).

```
Input:  M ∈ ℝ^{N×P}
Output: scalar

Let m̄_j = (1/N) Σ_{i=1}^{N} m_ij    (average assignment mass of primitive j)

L_par = ( (1/P) Σ_{j=1}^{P} √m̄_j )²
```

This is a 0.5-norm style sparsity surrogate that encourages point assignments to concentrate on fewer primitives. A simple average of m̄_j would not promote sparsity because M rows already sum to 1 — this squared-root-sum formulation is what makes the loss effective.

### 4.4 Existence loss (L_exist)

Teaches the model to explicitly flag which primitives are active. Inherited from SuperDec (Eq. 9).

```
Inputs:  a ∈ ℝ^P  (existence logits, the 12th column of E)
         α_j = σ(a_j)  (derived existence probabilities)
         α̂ ∈ ℝ^P  (derived binary targets from M)
Output:  scalar

m̄_j = (1/N) Σ_i m_ij
α̂_j = 1  if  m̄_j > ε_exist,  else 0

L_exist = (1/P) Σ_{j=1}^{P}  BCE(α_j, α̂_j)
```

Where ε_exist is a threshold. We follow the implementation's effective rule, equivalent to requiring roughly 24 assigned points; the paper notation is ambiguous (it defines m̄_j as normalized by N but lists ε_exist = 24, which is inconsistent if interpreted literally). The exact value must be verified against the original code.

### 4.5 Consistency loss (L_cons) — optional, currently disabled by default

Forces the decoder to produce a reasonable coarse shape even without the residual.

```
Inputs:  E ∈ ℝ^{P×12},  X ∈ ℝ^{N×3}
Output:  scalar

X̂_coarse = Decode(E, Z=0)   ← run decoder with residuals zeroed out
L_cons = Chamfer(X̂_coarse, X)
```

Requires a second forward pass through the decoder per training step.

The repository now implements the true `Z=0` decoder consistency path. `lambda_cons` is still set to 0.0 in the default configs, so the extra decoder pass is only requested when explicitly enabled.

**Cheaper alternative not currently implemented:** decoder-side residual dropout or Z-dropout during training. With probability p, the decoder receives `Z=0` for the normal reconstruction pass. This approximates the same pressure at lower cost than a second decoder pass, but it does not optimize both full-Z and zero-Z reconstructions in the same step.

**When to add:** Monitor `mean(‖Δ‖) / mean(‖X_sq‖)` during training. If this ratio grows too high, enable `lambda_cons` or add Z-dropout.

### 4.6 Total loss

```
Phase 1:
  L = L_recon [+ λ_cons · L_cons]

Phase 2:
  L = L_recon + λ_sq · L_sq + λ_par · L_par + λ_exist · L_exist [+ λ_cons · L_cons]
```

**Starting hyperparameters** (inherited from SuperDec where applicable):

| Weight | Value | Source |
|---|---|---|
| λ_sq | 1.0 | Tune based on balance with L_recon |
| λ_par | 0.06 | SuperDec default |
| λ_exist | 0.01 | SuperDec default |
| λ_cons | 0.0 default | Enable only as an ablation; tune empirically |

---

## 5. Training Plan

### 5.1 Dataset

**ShapeNet** (13 classes, same split as SuperDec / Choy et al.):
- Train/val/test splits predefined
- 4096 points per object via Farthest Point Sampling
- Objects pre-aligned in canonical orientation

For evaluation on real-world generalization: **ScanNet++** and **Replica** (same as SuperDec).

### 5.2 Training phases

#### Phase 0: SuperDec pretraining (already done)

Train the standard SuperDec model on ShapeNet. This produces pretrained weights for:
- PVCNN encoder
- Transformer decoder (D=3 layers)
- Segmentation head
- Parameter head

Use the publicly available SuperDec code and hyperparameters:
`P=16, S=4096, K_LM=25, H=128, D=3, ε_exist=24, λ_exist=0.01, λ_par=0.06`.

#### Phase 1: Decoder warm-up (freeze encoder)

**Goal:** Learn a working decoder without corrupting the pretrained encoder.

**What is frozen:** All encoder components (PVCNN, Transformer, segmentation head, parameter head).

**What is trained:**
- Residual projection: assignment-weighted mean/max/variance pooling + MLP(ℝ^{4H} → ℝ^H → ℝ^d)
- Offset network (MLP or cross-attention variant)

**Loss:** `L = L_recon` by default. `λ_cons · L_cons` can be added in Phase 1 because it depends on decoder outputs and trains the decoder/residual path, even while the SuperDec backbone is frozen.

Since the entire encoder is frozen in this stage, primitive-fitting losses on E and M are constant and do not provide useful gradients to any trainable parameter. This stage isolates decoder learning and avoids destabilizing the pretrained primitive decomposition.

**Optimizer:** Adam, lr = 1e-4 for new components.

**Expected duration:** ~100–200 epochs on ShapeNet.

**Success criterion:** Chamfer-L2 on validation set is significantly lower than SuperDec's SQ-only reconstruction error (0.047 × 10⁻² in-category).

#### Phase 2: Joint fine-tuning (unfreeze everything)

**Goal:** Allow the encoder to adapt to the autoencoder objective.

**What is trained:** Everything (encoder + decoder).

**Loss:** `L = L_recon + λ_sq · L_sq + λ_par · L_par + λ_exist · L_exist` by default, optionally plus `λ_cons · L_cons`.

L_sq, L_par, and L_exist are now meaningful because the encoder is trainable and these losses provide gradients to the encoder parameters.

**Optimizer:** Adam, with differential learning rates:
- Encoder: lr = 1e-5 (small, to avoid catastrophic forgetting)
- Decoder + residual projection: lr = 1e-4

**Expected duration:** ~100–200 epochs.

**Monitoring:** Track the following during training:
- L_recon (should decrease) and L_sq (should not increase significantly).
- Offset magnitude ratio: `mean(‖Δ‖) / mean(‖X_sq‖)`. Below 0.3 is healthy.
- Average active primitive count: `mean(Σ_j 𝟙[α_j > 0.5])`. Should stay near ~5–6.
- Primitive mass entropy: `−Σ_j m̄_j log m̄_j`. Low entropy = concentrated assignment = good.
- Scaffold vs. offset contribution: Chamfer(X_sq, X) vs. Chamfer(X̂, X). The gap is what the offset adds.

#### Phase 3 (optional): Activate L_cons or Z-dropout if needed

If the offset ratio `mean(‖Δ‖) / mean(‖X_sq‖)` grows too high during Phase 1 or Phase 2, enable the implemented `L_cons` and continue training, or add a cheaper Z-dropout regularizer as a follow-up change.

### 5.3 LM Optimization

We do **not** use LM refinement in the autoencoder pipeline. Although LM can improve primitive fitting in SuperDec, it would introduce a train-test mismatch here because the residual code Z is predicted jointly with the pre-refinement primitive code E. Refining E after encoding would make (E_LM, Z) inconsistent for the decoder, which was only ever trained on (E, Z) pairs from the encoder's direct output. Since the decoder is offset-based and starts from SQ surfaces, any change to E shifts the surface scaffold that Z was conditioned on.

We leave LM-based refinement as a separate evaluation-only ablation, clearly marked as such.

---

## 6. Evaluation Plan

### 6.1 Reconstruction quality

**Datasets:** ShapeNet (in-category and out-of-category), ScanNet++, Replica.

**Metrics:**
- Chamfer-L1 (×10²)
- Chamfer-L2 (×10²)
- F-Score at threshold τ = 0.01
- Normal consistency (if normals are available)

**Baselines:**

| Method | Type | Notes |
|---|---|---|
| FoldingNet (Yang et al., 2018) | Point cloud AE | Standard AE baseline, opaque latent |
| AtlasNet (Groueix et al., 2018) | Point cloud AE | Learned surface parameterization |
| PointFlow (Yang et al., 2019) | Point cloud generative | VAE-based, continuous normalizing flows |
| 3D-PointCapsNet (Zhao et al., 2019) | Point cloud AE | Capsule-based, some structure |
| SuperDec (pure SQ, no decoder) | Primitive decomposition | Our encoder-only ablation |
| **Ours (SuperDec-AE)** | Structured AE | Full pipeline |

### 6.2 Compression / bitrate analysis

**Key experiment:** Plot reconstruction error vs. representation size.

Baselines should be compared at **matched float budget**, not only at their default latent dimensions. For FoldingNet/AtlasNet, run at several latent dimensions and overlay on the same plot. Otherwise, any improvement could be attributed to using a larger latent rather than to the structured bottleneck.

| Method | Latent size (floats) — fixed slot | Latent size (floats) — active only |
|---|---|---|
| FoldingNet | 512 | 512 |
| AtlasNet | 1,024 | 1,024 |
| SuperDec (E only) | 192 | ~72 |
| **Ours (d=32)** | 704 | ~264 |
| **Ours (d=64)** | 1,216 | ~456 |
| **Ours (d=128)** | 2,240 | ~840 |

Sweep d ∈ {16, 32, 64, 128, 256} and plot Chamfer-L2 vs. total bottleneck size (reporting both fixed and active-only). This shows the trade-off between compression and fidelity.

### 6.3 Interpretability and editability

The model is expected to be more editable than a black-box AE, but editability is **local and geometry-aware**, not automatically semantic in the strong sense. Primitive slots are not semantically ordered across objects (slot 3 in one chair ≠ slot 3 in another).

**Primitive correspondence for cross-object edits.** Because primitive slots are not semantically ordered, cross-object interpolation and part swapping require an explicit correspondence step. For two shapes, we match active primitives with Hungarian assignment using a cost based on translation, scale, shape exponents, and assignment mass. We restrict these experiments to same-category objects to avoid degenerate correspondences.

**Primitive selection protocol.** For single-object editing (removal, scaling), we select the target primitive by either:
- Manual selection from a visualization of the decomposition, or
- Selecting the primitive with the largest assignment mass m̄_j in a user-specified spatial region.

This must be documented to avoid the appearance of cherry-picking.

**Qualitative experiments:**

1. **Part removal:** Delete one superquadric (set α_j = 0, zero out z_j), decode. Show that the rest of the object remains intact. Compare with removing a latent dimension in FoldingNet (expect: unpredictable global distortion).

2. **Part manipulation:** Scale up a chair seat (multiply s_x, s_y of the seat SQ), decode. Show that the result is a plausible chair with a wider seat.

3. **Part swap:** Take matched leg SQs from object A and matched seat SQs from object B (via Hungarian matching), combine, decode. Show meaningful compositionality. Restrict to same-category pairs.

4. **Interpolation:** For two same-category objects, establish primitive correspondence via Hungarian matching, then linearly interpolate (E_A, Z_A) → (E_B, Z_B), decode at intermediate points. Show that the morphing passes through geometrically valid intermediate shapes.

### 6.4 Shape completion (if time permits)

**Important caveat:** The encoder is trained on full point clouds, not partial ones. A structural prior helps, but strong completion claims require either partial-input training augmentation or at least dedicated robustness experiments. Random point dropout is not sufficient; structured masks or simulated occlusion should be used.

**Setup:** Apply structured masks (simulate single-view occlusion) removing 30%/50%/70% of points. Encode the partial point cloud. Decode.

**Hypothesis:** The SQ bottleneck acts as a structural prior — even with moderate masking, the model may infer the global part structure better than unstructured baselines. This claim is speculative until validated experimentally.

**Metrics:** Chamfer distance to the complete ground-truth shape, plotted vs. masking ratio.

---

## 7. Ablation Studies

### 7.1 Residual dimension d

Sweep d ∈ {0, 16, 32, 64, 128, 256}.
- d = 0 is the pure SQ decoder (no residual). Expected: coarse, high Chamfer.
- d = 256 is a very expressive residual. Expected: best Chamfer, but potential collapse risk.
- d = 64 is our default. Expected: good balance.

### 7.2 Decoder architecture

Compare Option A (MLP only) vs. Option B (MLP + cross-attention) vs. Option C (full Transformer decoder) on ShapeNet in-category.

### 7.3 Per-part vs. global residual

- Per-part: Z ∈ ℝ^{P×d} (each SQ has its own z_j). Better for editing.
- Global: z ∈ ℝ^{d_global} (one shared code). Potentially more efficient for reconstruction.

Compare reconstruction quality and editing quality.

### 7.4 Loss ablation

| Config | L_recon | L_sq | L_par | L_exist | L_cons | Expected result |
|---|---|---|---|---|---|---|
| Full | ✓ | ✓ | ✓ | ✓ | ✓ | Best overall |
| No L_sq | ✓ | ✗ | ✓ | ✓ | ✗ | SQ collapse: E becomes meaningless |
| No L_cons | ✓ | ✓ | ✓ | ✓ | ✗ | Likely fine if architecture is offset-based |
| No residual | ✓ | ✓ | ✓ | ✓ | ✗ | Pure SQ decoding, coarse output |
| Only L_recon | ✓ | ✗ | ✗ | ✗ | ✗ | Worst: total collapse, standard AE behavior |

### 7.5 Offset magnitude monitoring

Track `mean(‖Δ‖) / mean(‖X_sq‖)` across all ablations. Report this as a health metric. Values below 0.3 indicate the SQs carry the main structure; values above 1.0 indicate collapse.

### 7.6 LM as evaluation-only ablation

Run the encoder, apply LM to refine E, then decode with the original Z. Report Chamfer separately. This quantifies the E–Z mismatch problem discussed in Section 5.3.

---

## 8. Risks and Mitigations

### 8.1 Residual collapse (Z does everything)

**Risk:** The model ignores E and stores all information in Z.
**Mitigation:**
1. Offset-based decoder (architectural constraint: must start from SQ surfaces).
2. L_sq (loss-level constraint: SQs must fit the input independently).
3. Implement scale-aware offset caps, `tanh(Δ_raw) · cap · mean(scale_j)`, to bound residual displacement relative to primitive size.
4. L_cons or Z-dropout so decoding with Z=0 still works.
5. Monitor offset magnitude ratio during training.

### 8.2 SQ sampling discontinuities

**Risk:** The parametric SQ surface sampler has singularities at the poles (η = ±π/2) and when ε₁ or ε₂ are very small (sharp edges).
**Mitigation:** Clamp exponents to [0.1, 2.0]. Use equal-distance sampling (Pilu et al.) rather than naive uniform (η,ω). Add small noise to sampled points.

### 8.3 Editing artifacts (stale Z)

**Risk:** When editing E (e.g., moving a superquadric), the associated z_j may encode details that are coupled to the old SQ configuration, producing artifacts.
**Mitigation:**
1. Per-part residuals (z_j is local to each SQ, not global).
2. The offset decoder architecture constrains offsets to be relative to the SQ surface, so moderate changes to E produce moderate changes in the output.
3. For large edits, optionally decode with z_j = 0, then refine.

### 8.4 Primitive correspondence for cross-object operations

**Risk:** Unsupervised decomposition does not guarantee semantic alignment of primitive slots across objects. Slot 3 in one chair is not guaranteed to correspond to slot 3 in another.
**Mitigation:** Hungarian matching on (translation, scale, exponents, assignment mass). Restrict swap/interpolation to same-category objects. Document the matching protocol explicitly.

### 8.5 Scene-level extension (too ambitious for v1)

**Risk:** Extending to full scenes introduces instance segmentation noise, variable object counts, and layout complexity.
**Mitigation:** Keep the first paper object-level only. Scene-level results can be shown qualitatively (as SuperDec does) but should not be the core contribution.

---

## 9. Implementation Roadmap

### Step 1: Environment setup (Week 1)

- [ ] Clone SuperDec codebase
- [ ] Reproduce SuperDec results on ShapeNet (verify pretrained weights)
- [ ] Set up evaluation pipeline (Chamfer-L1, L2, F-Score)
- [ ] Verify ShapeNet data loading and FPS sampling
- [ ] Document the rotation parameterization used by SuperDec

### Step 2: Implement SQ surface sampler (Week 1–2)

- [ ] Implement differentiable signed-power parametric sampling on SQ surfaces
- [ ] Implement equal-distance sampling following Pilu et al.
- [ ] Implement soft gating via σ(a_j / τ) where a_j is the existence logit (no hard thresholding during training)
- [ ] Test: sample S_dec points per SQ, visualize, verify correctness
- [ ] Handle edge cases: exponent clamping, pole singularities
- [ ] Verify gradients flow through sampler to E

### Step 3: Implement residual projection (Week 2)

- [x] Implement weighted mean pooling: g_j = Σ m_ij F_PC,i / (Σ m_ij + ε)
- [x] Add masked max and weighted variance statistics
- [x] Implement MLP([F_SQ,j; g_mean,j; g_max,j; g_var,j]) → Z_j ∈ ℝ^d
- [ ] Verify output shape: Z ∈ ℝ^{P×d}
- [ ] Wire into the training loop

### Step 4: Implement offset decoder (Week 2–3)

- [x] Add Fourier features for sampled SQ surface coordinates
- [x] Add split projections for position, E_dec, Z, and gate
- [x] Implement stacked offset decoder with within-primitive self-attention, primitive cross-attention, and FFN blocks
- [x] Add primitive-scale offset cap: `tanh(raw) * offset_cap * mean(scale_j)`
- [x] Implement X̂ = X_sq + w⊙Δ (gated offsets)
- [ ] Implement L_recon (bidirectional weighted Chamfer-L2)
- [ ] Train Phase 1 (frozen encoder, L_recon only) and evaluate

### Step 5: Evaluate decoder variants (Week 3–4)

- [ ] Measure Chamfer-L1/L2 on ShapeNet val set
- [ ] Compare vs. SuperDec (SQ-only) and vs. FoldingNet
- [ ] Visualize reconstructions qualitatively
- [ ] Monitor offset magnitude ratio
- [ ] Compare the current stacked attention decoder against a simpler MLP baseline if needed

### Step 6: Residual-collapse regularization (Week 4–5)

- [x] Add true `L_cons` path using a second decoder pass with `Z=0`
- [ ] Keep `lambda_cons=0.0` by default and enable only as an ablation
- [ ] Optionally add training-time Z-dropout as a cheaper alternative
- [ ] Retrain Phase 1 and compare with and without `L_cons` / Z-dropout

### Step 7: Joint fine-tuning — Phase 2 (Week 5–6)

- [ ] Unfreeze encoder, train with full loss (L_recon + L_sq + L_par + L_exist)
- [ ] Implement L_sq as sampled-surface bidirectional Chamfer (NOT radial distance)
- [ ] Implement L_par with correct 0.5-norm formula
- [ ] Implement differential learning rates
- [ ] Monitor L_sq stability (should not increase much)
- [ ] Evaluate on ShapeNet in-category and out-of-category

### Step 8: Run ablations (Week 6–7)

- [ ] Sweep residual dimension d
- [ ] Loss ablation (with/without L_sq, with/without L_cons)
- [ ] Per-part vs. global residual
- [ ] LM evaluation-only ablation
- [ ] Compile results tables

### Step 9: Editing experiments (Week 7–8)

- [ ] Implement Hungarian matching for primitive correspondence
- [ ] Define and document primitive selection protocol
- [ ] Implement part removal, scaling, swapping
- [ ] Generate qualitative figures
- [ ] Implement interpolation in (E, Z) space for same-category pairs

### Step 10: Generalization to real-world data (Week 8–9)

- [ ] Evaluate on ScanNet++ objects (no fine-tuning)
- [ ] Evaluate on Replica objects (no fine-tuning)
- [ ] Optionally: show qualitative full-scene results via Mask3D pipeline

### Step 11: Write paper (Week 9–12)

- [ ] Draft method section
- [ ] Compile quantitative results (including matched-budget baselines)
- [ ] Create figures (architecture, qualitative comparisons, editing demos)
- [ ] Write related work, introduction, conclusion
- [ ] Internal review and revision

---

## 10. Summary of Tensor Shapes

Quick reference for the full forward pass:

```
Input:
  X ∈ ℝ^{N×3}                        ← input point cloud (N=4096)

Encoder:
  F_PC ∈ ℝ^{N×H}                     ← PVCNN point features (H=128)
  F⁰_SQ ∈ ℝ^{P×H}                   ← initial SQ query tokens (P=16)
  F_SQ ∈ ℝ^{P×H}                     ← refined SQ features (after Transformer)
  M ∈ ℝ^{N×P}                        ← soft segmentation matrix
  E ∈ ℝ^{P×12}                       ← SQ parameters (explicit latent)
  G ∈ ℝ^{P×H}                        ← weighted pool of local point features (each g_j ∈ ℝ^H)
  Z ∈ ℝ^{P×d}                        ← per-part residual via MLP([F_SQ,j; g_j]) (d=64)

Decoder:
  X_sq ∈ ℝ^{(P·S_dec)×3}            ← coarse point cloud on SQ surfaces (S_dec=256)
  w ∈ [0,1]^{P·S_dec}                ← soft existence weights (from logit a_j)
  part_ids ∈ {1,...,P}^{P·S_dec}     ← which SQ each surface point belongs to
  F_dec ∈ ℝ^{(P·S_dec)×(3+12+d)}    ← per-point features
  Δ ∈ ℝ^{(P·S_dec)×3}               ← predicted offsets (gated by w before adding)

Output:
  X̂ = X_sq + w⊙Δ ∈ ℝ^{(P·S_dec)×3} ← training (fixed size, weighted loss)
  X̂ ∈ ℝ^{N'×3}                       ← inference (pruned, N' ≤ P·S_dec)
```

---

## 11. Key Design Decisions Log

| Decision | Choice | Rationale |
|---|---|---|
| Decoder type | Offset-based (X̂ = X_sq + Δ) | Forces SQs to carry coarse structure architecturally |
| Residual scope | Per-part (Z ∈ ℝ^{P×d}) | Enables part-level editing |
| Residual dim d | 64 (default) | ~5× more capacity than E; sweep d as ablation |
| Offset network | MLP first, cross-attn if needed | Start simple, add complexity only if justified |
| Cross-attn keys/values | [E_j^proj; Z_j], not Z alone | Context should include both structural and residual info |
| LM optimization | **Not used** in AE pipeline | Creates (E_LM, Z) mismatch; separate ablation only |
| Existence gating | Soft weight w_j gates offsets and loss, NOT coordinates | Multiplying coordinates by w_j shrinks points toward origin |
| Existence head output | Raw logit a_j; α_j = σ(a_j), w_j = σ(a_j/τ) | Avoids double-sigmoid if head already outputs probability |
| Residual projection | MLP([F_SQ,j; g_j]) where g_j pools local point features | Pure linear of F_SQ,j underuses encoder; local evidence improves detail |
| Surface sampling | Equal-distance (Pilu et al.), signed-power formulation | Uniform (η,ω) gives non-uniform density; signed power needed for correctness |
| L_sq definition | Sampled-surface bidirectional Chamfer (SuperDec Eq. 6-7) | Radial distance is LM objective, not neural training loss |
| L_par definition | 0.5-norm surrogate (SuperDec Eq. 8) | Simple average doesn't promote sparsity |
| Phase 1 loss | L_recon only | L_sq is constant when encoder is frozen (zero gradient) |
| L_cons | Off by default, activate if offset ratio > 0.3 | Two other defenses (architecture + L_sq) may suffice |
| Compression reporting | Both fixed-slot and active-only budgets | Active-only is optimistic; both must be stated |
| Editing protocol | Hungarian matching + documented selection | Unsupervised slots are not semantically ordered |
| Scene-level | Qualitative only, not core contribution | Avoid inheriting instance segmentation complexity |
| Training | Initialize from pretrained SuperDec, Phase 1 → 2 | Avoid training the full pipeline from scratch |
