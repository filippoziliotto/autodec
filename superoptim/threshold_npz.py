"""
Threshold bending_k and tapering parameters in a superdec NPZ file.

For each primitive, if the absolute value of bending_k (any component) is below
--thresh_bending, all bending_k components for that primitive are zeroed out.
Similarly, if the L2 norm of tapering is below --thresh_tapering, tapering is zeroed.

Saves the modified file next to the original with a suffix appended to the name.
"""

import argparse
import os
import numpy as np
from superdec.utils.predictions_handler_extended import PredictionHandler


def apply_thresholds(handler: PredictionHandler, thresh_tapering: float, thresh_bending: float) -> np.ndarray:
    """Modify handler in-place: zero out tapering and bending_k below threshold, existing primitives only.
    Returns indices of objects where no primitive was modified."""
    existing = (handler.exist > 0.5).squeeze(-1)  # [B, P]
    max_scale = np.max(handler.scale, axis=-1)  # [B, P]
    existing = existing & (max_scale >= 0.1)  # exclude small primitives

    # bending shape: [B, P, 6] -> k at even indices (0, 2, 4), a at odd (1, 3, 5)
    bending = handler.bending  # [B, P, 6]
    bending_k = bending[..., 0::2]  # [B, P, 3]

    # zero bending per axis: if |k_i| * max_scale < threshold, zero out both k_i and a_i (existing only)
    axes_zeroed = np.zeros(bending_k.shape[:2], dtype=int)  # [B, P]
    for i in range(3):
        # mask_axis = existing & (np.abs(bending_k[..., i]) * max_scale < thresh_bending)  # [B, P]
        mask_axis = existing & (np.abs(bending_k[..., i]) < thresh_bending)  # [B, P]
        handler.bending[..., 2 * i][mask_axis] = 0.0    # zero k_i
        handler.bending[..., 2 * i + 1][mask_axis] = 0.0  # zero a_i
        axes_zeroed += mask_axis.astype(int)

    total_existing = int(existing.sum())
    for n in (3, 2, 1):
        count = int((axes_zeroed[existing] == n).sum())
        print(f"  {count}/{total_existing} existing primitives had {n} bending axis/axes zeroed (thresh={thresh_bending})")

    # Per-object: sum of |k| magnitude for primitives with exactly `active_axes` active bending axes
    for active_axes in (3, 2, 1):
        zeroed = 3 - active_axes
        prim_mask = (axes_zeroed == zeroed) & existing  # [B, P]
        # sum |k| across axes per primitive, then take the max over primitives per object
        sum_k = np.abs(bending_k).sum(axis=-1)  # [B, P]
        sum_k_masked = np.where(prim_mask, sum_k, 0.0)
        per_obj_mag = sum_k_masked.max(axis=1)  # [B]
        top10_idx = np.argsort(per_obj_mag)[::-1][:10]
        print(f"\n  Top 10 objects by max per-primitive sum-|k| ({active_axes}-axis-bending primitives):")
        print(f"    {'Index':<8} {'Max sum-|k|':<14} {'Name'}")
        for oi in top10_idx:
            if per_obj_mag[oi] == 0:
                break
            name = handler.names[oi] if hasattr(handler, 'names') else str(oi)
            print(f"    {oi:<8} {per_obj_mag[oi]:<14.4f} {name}")

    # zero tapering per primitive where max |tapering| < threshold (existing only)
    tapering = handler.tapering  # [B, P, 2]
    max_t = np.max(np.abs(tapering), axis=-1)  # [B, P]
    mask_t = existing & (max_t < thresh_tapering)
    handler.tapering[mask_t] = 0.0

    n_t = int(mask_t.sum())
    print(f"Zeroed tapering on {n_t}/{total_existing} existing primitives (thresh={thresh_tapering})")
    if n_t > 0:
        # bending indices: [kb_z=0, α_z=1, kb_x=2, α_x=3, kb_y=4, α_y=5]
        bent_kx = (np.abs(bending[~mask_t & existing, 2]) > 0).sum()
        bent_ky = (np.abs(bending[~mask_t & existing, 4]) > 0).sum()
        print(f"  combined tapering and bending on X: {bent_kx}")
        print(f"  combined tapering and bending on Y: {bent_ky}")
        # handler.bending[..., 2][mask_t] = 0.0    # zero k_i
        # handler.bending[..., 4][mask_t] = 0.0    # zero k_i

    # objects where at least 1 existing primitive had at least 1 parameter not zeroed
    has_nonzero = ((axes_zeroed < 3) | ~mask_t) & existing  # [B, P]
    not_thresholded = np.where(has_nonzero.any(axis=-1))[0]
    return not_thresholded


def make_output_path(input_path: str, suffix: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}{suffix}{ext}"


def main():
    parser = argparse.ArgumentParser(description="Threshold bending_k and tapering in a superdec NPZ file")
    parser.add_argument("npz", help="Path to the input NPZ file")
    parser.add_argument("--thresh_bending", type=float, default=0.5,#0.0005,
                        help="Zero bending_k for a primitive when max |k| < this value (default: 0.1)")
    parser.add_argument("--thresh_tapering", type=float, default=0.1,
                        help="Zero tapering for a primitive when max |k| < this value (default: 0.1)")
    parser.add_argument("--suffix", type=str, default="_thresholded",
                        help="Suffix appended to the output filename (default: _thresholded)")
    args = parser.parse_args()

    input_path = os.path.abspath(args.npz)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = make_output_path(input_path, args.suffix)

    print(f"Loading {input_path}")
    handler = PredictionHandler.from_npz(input_path)
    print(f"Loaded {len(handler.names)} models, {handler.scale.shape[1]} primitives each")

    not_thresholded = apply_thresholds(handler, args.thresh_tapering, args.thresh_bending)

    txt_path = make_output_path(input_path, args.suffix).replace(".npz", "_not_thresholded.txt")
    np.savetxt(txt_path, not_thresholded, fmt="%d")
    print(f"Saved {len(not_thresholded)} not-thresholded object indices to {txt_path}")

    print(f"Saving to {output_path}")
    handler.save_npz(output_path)
    print("Done.")


if __name__ == "__main__":
    main()
