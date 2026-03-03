"""
Plot the distribution of superquadric shape exponents (ε1, ε2) stored in an NPZ file.

Usage:
    python -m superoptim.plot_shape_params <path/to/file.npz> [--out <path/to/output.png>]

Only primitives whose `exist` score > 0.5 are included.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from superdec.utils.predictions_handler_extended import PredictionHandler


def main():
    parser = argparse.ArgumentParser(description="Plot exponent distributions from an NPZ file.")
    parser.add_argument("npz", help="Path to the .npz file produced by SuperDec.")
    parser.add_argument(
        "--out", default=None,
        help="Output image path. Defaults to <npz_stem>_shape_params.png beside the npz."
    )
    args = parser.parse_args()

    if not os.path.isfile(args.npz):
        print(f"Error: file not found: {args.npz}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.npz} ...")
    handler = PredictionHandler.from_npz(args.npz)

    exponents = handler.exponents   # [B, P, 2]
    exist     = handler.exist       # [B, P]

    mask = (exist > 0.5).reshape(-1)           # (B*P,)
    exp_flat = exponents.reshape(-1, 2)        # (B*P, 2)
    eps_active = exp_flat[mask]                # (N_active, 2)
    eps1 = eps_active[:, 0]                    # (N_active,)
    eps2 = eps_active[:, 1]                    # (N_active,)

    n_objects    = exponents.shape[0]
    n_primitives = exponents.shape[1]
    n_active     = mask.sum()
    print(f"Objects: {n_objects}  |  Primitives/obj: {n_primitives}  |  Active primitives: {n_active}")

    # ------------------------------------------------------------------ #
    # Output path
    # ------------------------------------------------------------------ #
    if args.out is None:
        base = os.path.splitext(args.npz)[0]
        out_path = base + "_shape_params.png"
    else:
        out_path = args.out

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    bins = 60

    # -- ε1 histogram --
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(eps1, bins=bins, color="steelblue", edgecolor="white", linewidth=0.4)
    ax0.set_xlabel("ε₁", fontsize=12)
    ax0.set_ylabel("Count", fontsize=12)
    ax0.set_title("Distribution of ε₁", fontsize=13)
    ax0.axvline(eps1.mean(), color="tomato",   linestyle="--", label=f"mean={eps1.mean():.3f}")
    ax0.axvline(np.median(eps1), color="gold", linestyle=":",  label=f"median={np.median(eps1):.3f}")
    ax0.legend(fontsize=9)

    # -- ε2 histogram --
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(eps2, bins=bins, color="darkorange", edgecolor="white", linewidth=0.4)
    ax1.set_xlabel("ε₂", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Distribution of ε₂", fontsize=13)
    ax1.axvline(eps2.mean(), color="tomato",   linestyle="--", label=f"mean={eps2.mean():.3f}")
    ax1.axvline(np.median(eps2), color="gold", linestyle=":",  label=f"median={np.median(eps2):.3f}")
    ax1.legend(fontsize=9)

    # -- 2-D joint scatter / density --
    ax2 = fig.add_subplot(gs[1, :])
    h = ax2.hist2d(eps1, eps2, bins=bins, cmap="viridis")
    fig.colorbar(h[3], ax=ax2, label="Count")
    ax2.set_xlabel("ε₁", fontsize=12)
    ax2.set_ylabel("ε₂", fontsize=12)
    ax2.set_title("Joint distribution of (ε₁, ε₂)", fontsize=13)

    fig.suptitle(
        f"{os.path.basename(args.npz)}\n"
        f"{n_objects} objects · {n_active} active primitives",
        fontsize=11, y=0.98
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")

    # Print summary stats
    print("\n----- Shape parameter statistics (active primitives only) -----")
    for name, arr in [("ε₁", eps1), ("ε₂", eps2)]:
        print(f"  {name}:  min={arr.min():.4f}  max={arr.max():.4f}  "
              f"mean={arr.mean():.4f}  std={arr.std():.4f}  median={np.median(arr):.4f}")


if __name__ == "__main__":
    main()
