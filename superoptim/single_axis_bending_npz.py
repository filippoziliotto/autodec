"""
For each existing primitive, keep only the bending axis with the largest |k|
and zero out the other two axes. Tapering is still thresholded as in threshold_npz.py.

Saves the modified file next to the original with a suffix appended to the name.
"""

import argparse
import os
import numpy as np
from superdec.utils.predictions_handler_extended import PredictionHandler


def apply_single_axis_bending(handler: PredictionHandler, thresh_tapering: float) -> None:
    """Modify handler in-place: keep only the max-|k| bending axis per primitive.
    Tapering is zeroed when max |tapering| < thresh_tapering. Small primitives excluded."""
    existing = (handler.exist > 0.5).squeeze(-1)  # [B, P]
    max_scale = np.max(handler.scale, axis=-1)     # [B, P]
    # existing = existing & (max_scale >= 0.1)       # exclude small primitives

    total_existing = int(existing.sum())

    # bending: [B, P, 6] -> k at even indices (0=z, 2=x, 4=y), alpha at odd
    bending_k = handler.bending[..., 0::2]  # [B, P, 3]

    # For each primitive, find the axis with the largest |k|
    best_axis = np.argmax(np.abs(bending_k), axis=-1)  # [B, P]

    for i in range(3):
        # Zero axes that are NOT the best axis, for existing primitives only
        mask = existing & (best_axis != i)
        handler.bending[..., 2 * i][mask] = 0.0      # zero k_i
        handler.bending[..., 2 * i + 1][mask] = 0.0  # zero a_i

    # Stats: how many primitives had each axis selected as dominant
    axis_names = ['z', 'x', 'y']
    for i, name in enumerate(axis_names):
        count = int((existing & (best_axis == i)).sum())
        print(f"  {count}/{total_existing} existing primitives kept bending on {name}-axis")

    # Zero tapering where max |tapering| < threshold
    # tapering = handler.tapering  # [B, P, 2]
    # max_t = np.max(np.abs(tapering), axis=-1)  # [B, P]
    # mask_t = existing & (max_t < thresh_tapering)
    # handler.tapering[mask_t] = 0.0

    # n_t = int(mask_t.sum())
    # print(f"Zeroed tapering on {n_t}/{total_existing} existing primitives (thresh={thresh_tapering})")


def make_output_path(input_path: str, suffix: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}{suffix}{ext}"


def main():
    parser = argparse.ArgumentParser(
        description="Keep only the dominant bending axis per primitive in a superdec NPZ file"
    )
    parser.add_argument("npz", help="Path to the input NPZ file")
    parser.add_argument("--thresh_tapering", type=float, default=0.1,
                        help="Zero tapering for a primitive when max |tapering| < this value (default: 0.1)")
    parser.add_argument("--suffix", type=str, default="_single_axis",
                        help="Suffix appended to the output filename (default: _single_axis)")
    args = parser.parse_args()

    input_path = os.path.abspath(args.npz)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = make_output_path(input_path, args.suffix)

    print(f"Loading {input_path}")
    handler = PredictionHandler.from_npz(input_path)
    print(f"Loaded {len(handler.names)} models, {handler.scale.shape[1]} primitives each")

    apply_single_axis_bending(handler, args.thresh_tapering)

    print(f"Saving to {output_path}")
    handler.save_npz(output_path)
    print("Done.")


if __name__ == "__main__":
    main()
