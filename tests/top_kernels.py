"""Inspect HYDRA kernels: build the (per-class) win-count matrices, rank the
most-used and the most class-discriminative kernels, and plot their shapes.

Builds a HydraTransform with count tracking enabled, streams synthetic windows
WITH labels through it in batches (so the global and per-class win-count matrices
accumulate exactly as they would over a real data split during feature
extraction), then reports:

  - the top-K kernels by global win count, and
  - the top-K kernels by class-discriminative win rate (the normalized
    within-group win fraction difference between the two classes).

The synthetic windows carry a weak class-dependent rhythm so some kernels really
do fire differently per class. Only needs torch + numpy (+ matplotlib for the
plot); no corpus required.

To rank kernels on the real EEG data instead, set `feature.track_counts=true` in
a normal training run (labels are already passed through during feature
extraction) and call `feature_extractor.top_discriminative_kernels(K)`.

Run:
    uv run python tests/top_kernels.py --top 12 --by max
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-windows", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--channels", type=int, default=19)
    p.add_argument("--timepoints", type=int, default=2000)
    p.add_argument("--n-groups", type=int, default=64)
    p.add_argument("--n-kernels", type=int, default=8)
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--by", choices=["max", "min", "total"], default="max")
    p.add_argument("--weighting", choices=["frequency", "magnitude"], default="frequency",
                   help="count how OFTEN a kernel wins, or its summed winning magnitude")
    p.add_argument("--score", choices=["difference", "ratio", "logodds"], default="difference",
                   help="how to combine the two classes' win fractions")
    p.add_argument("--signal", type=float, default=0.4, help="class-signal strength")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="tests/outputs")
    return p.parse_args(argv)


def _make_labelled_windows(n, channels, timepoints, signal, torch):
    """Synthetic windows with binary labels and a class-dependent rhythm."""
    labels = (torch.arange(n) % 2).long()  # balanced 0/1
    t = torch.linspace(0.0, 1.0, timepoints)
    x = torch.randn(n, channels, timepoints)
    for i in range(n):
        freq = 5.0 if labels[i].item() == 1 else 12.0
        x[i, :3, :] += signal * torch.sin(2 * torch.pi * freq * t)
    return x, labels


def _plot_discriminative(top, tag: str, out_dir: str) -> None:
    """Save a grid plot of the most discriminative kernels' shapes, if possible."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available; skipping plot)")
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    n = len(top)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2 * rows), squeeze=False)
    for i, info in enumerate(top):
        ax = axes[i // cols][i % cols]
        ax.plot(range(9), info.weight.tolist(), marker="o")
        ax.axhline(0.0, color="gray", linewidth=0.5)
        ax.set_title(
            f"#{info.rank} favors={info.favors} s={info.score:+.3f}\n"
            f"d{info.dilation} {info.representation} g{info.group}k{info.kernel}",
            fontsize=7,
        )
        ax.set_xticks(range(9))
        ax.tick_params(labelsize=6)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"Top {n} class-discriminative HYDRA kernels ({tag})")
    fig.tight_layout()
    out_file = out_path / f"top_discriminative_kernels_{tag}.png"
    fig.savefig(out_file, dpi=120)
    plt.close(fig)
    print(f"\nSaved plot to {out_file}")


def main(argv=None) -> int:
    args = parse(argv)
    try:
        import torch

        from src.models.components.hydra_transform import HydraTransform
    except ImportError as exc:
        print(f"This script needs torch + numpy: {exc}")
        print("Run it inside the project container / uv environment.")
        return 1

    torch.manual_seed(args.seed)
    transform = HydraTransform(
        n_timepoints=args.timepoints,
        n_channels=args.channels,
        k=args.n_kernels,
        g=args.n_groups,
        seed=args.seed,
        track_counts=True,
    )
    x, labels = _make_labelled_windows(
        args.n_windows, args.channels, args.timepoints, args.signal, torch
    )

    # Stream labelled windows through in batches so the global and per-class
    # win-count matrices accumulate across forward calls.
    for start in range(0, args.n_windows, args.batch_size):
        transform(x[start:start + args.batch_size], labels[start:start + args.batch_size])

    print("=" * 78)
    print("HYDRA kernel win-count matrices")
    print("=" * 78)
    print(f"  global shape         : {tuple(transform.win_counts_max.shape)}  "
          "(num_dilations, divisor, h, k)")
    print(f"  classes tracked      : {transform.class_labels()}")
    print(f"  total kernels        : {transform.win_counts_max.numel()}")

    # Check 1: every (example, dilation, repr, group, timepoint) has one argmax
    # winner; output length is T for raw and T-1 for diff.
    t = args.timepoints
    expected = args.n_windows * transform.h * transform.num_dilations * (2 * t - 1)
    total = int(transform.win_counts_max.sum().item())
    print(f"  total argmax events  : {total}  (expected {expected}) -> "
          f"{'OK' if total == expected else 'MISMATCH'}")

    # Check 2: the per-class matrices partition the global one.
    per_class_sum = sum(transform.win_counts_max_by_class.values())
    matches = bool(torch.equal(per_class_sum, transform.win_counts_max))
    print(f"  per-class sum==global: {'OK' if matches else 'MISMATCH'}")

    top = transform.top_kernels(args.top, by=args.by, weighting=args.weighting)
    print(f"\nTop {len(top)} kernels by GLOBAL '{args.by}/{args.weighting}' count:")
    print(f"  {'rank':>4} {'count':>14} {'dil':>5} {'repr':>5} {'group':>6} {'kernel':>6}")
    for info in top:
        print(f"  {info.rank:>4} {info.count:>14.2f} {info.dilation:>5} "
              f"{info.representation:>5} {info.group:>6} {info.kernel:>6}")

    disc = transform.top_discriminative_kernels(
        args.top, by=args.by, weighting=args.weighting, metric=args.score
    )
    a, b = transform.class_labels()[0], transform.class_labels()[1]
    print(f"\nTop {len(disc)} CLASS-DISCRIMINATIVE kernels "
          f"(by={args.by}/{args.weighting}, metric={args.score}, classes {a} vs {b}):")
    print(f"  {'rank':>4} {'score':>9} {'favors':>6} {'frac_' + str(a):>8} "
          f"{'frac_' + str(b):>8} {'dil':>5} {'repr':>5} {'grp':>4} {'ker':>4}  weights")
    for info in disc:
        weights = ", ".join(f"{v:+.2f}" for v in info.weight.tolist())
        print(f"  {info.rank:>4} {info.score:>+9.3f} {str(info.favors):>6} "
              f"{info.fractions[a]:>8.3f} {info.fractions[b]:>8.3f} "
              f"{info.dilation:>5} {info.representation:>5} {info.group:>4} "
              f"{info.kernel:>4}  [{weights}]")

    _plot_discriminative(disc, f"{args.by}_{args.weighting}_{args.score}", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
