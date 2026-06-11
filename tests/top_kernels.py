"""Inspect HYDRA kernels: build the (per-class) win-count matrices, rank the
most-used and the most class-discriminative kernels, and plot how they look in
both time (the dilated waveform, in ms) and frequency (the magnitude response,
in Hz).

Builds a HydraTransform with count tracking enabled, streams synthetic windows
WITH labels through it in batches (so the global and per-class win-count matrices
accumulate exactly as they would over a real data split during feature
extraction), then reports:

  - the top-K kernels by global win count, and
  - the top-K kernels by class-discriminative win rate (the normalized
    within-group win fraction difference between the two classes),

each with its peak frequency (Hz). The synthetic windows carry class-dependent
rhythms (10 Hz vs 25 Hz) so some kernels really do fire differently per class and
the peak-frequency histogram separates by favored class. Only needs torch + numpy
(+ matplotlib for the plots); no corpus required.

To rank kernels on the real EEG data instead, set `feature.track_counts=true` in
a normal training run; train.py then logs, saves, and plots these automatically.

Run:
    uv run python tests/top_kernels.py --top 12 --by max --weighting frequency --score difference
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
    p.add_argument("--sfreq", type=float, default=250.0, help="sampling rate in Hz")
    p.add_argument("--n-groups", type=int, default=64)
    p.add_argument("--n-kernels", type=int, default=8)
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--per-class", type=int, default=5,
                   help="discriminative kernels to show per class (balanced view)")
    p.add_argument("--by", choices=["max", "min", "total"], default="max")
    p.add_argument("--weighting", choices=["frequency", "magnitude"], default="frequency",
                   help="count how OFTEN a kernel wins, or its summed winning magnitude")
    p.add_argument("--score", choices=["difference", "ratio", "logodds"], default="difference",
                   help="how to combine the two classes' win fractions")
    p.add_argument("--signal", type=float, default=0.4, help="class-signal strength")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="tests/outputs")
    return p.parse_args(argv)


def _make_labelled_windows(n, channels, timepoints, signal, sfreq, torch):
    """Synthetic windows with binary labels and a class-dependent rhythm (in Hz)."""
    labels = (torch.arange(n) % 2).long()  # balanced 0/1
    time_s = torch.arange(timepoints) / sfreq
    x = torch.randn(n, channels, timepoints)
    for i in range(n):
        freq_hz = 25.0 if labels[i].item() == 1 else 10.0
        x[i, :3, :] += signal * torch.sin(2 * torch.pi * freq_hz * time_s)
    return x, labels


def _peak(info) -> str:
    return f"{info.peak_freq_hz:.1f}" if info.peak_freq_hz is not None else "-"


def main(argv=None) -> int:
    args = parse(argv)
    try:
        import torch

        from src.models.components import kernel_viz
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
        args.n_windows, args.channels, args.timepoints, args.signal, args.sfreq, torch
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
    print(f"  sampling rate (Hz)   : {args.sfreq}  (Nyquist {args.sfreq / 2:.1f} Hz)")
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

    top = transform.top_kernels(
        args.top, by=args.by, weighting=args.weighting, sfreq=args.sfreq
    )
    print(f"\nTop {len(top)} kernels by GLOBAL '{args.by}/{args.weighting}' count:")
    print(f"  {'rank':>4} {'count':>14} {'peakHz':>7} {'dil':>5} {'repr':>5} "
          f"{'group':>6} {'kernel':>6}")
    for info in top:
        print(f"  {info.rank:>4} {info.count:>14.2f} {_peak(info):>7} {info.dilation:>5} "
              f"{info.representation:>5} {info.group:>6} {info.kernel:>6}")

    a, b = transform.class_labels()[0], transform.class_labels()[1]
    per_class = transform.top_discriminative_kernels_per_class(
        max(args.per_class, 50), by=args.by, weighting=args.weighting,
        metric=args.score, sfreq=args.sfreq,
    )
    print(f"\nTop {args.per_class} CLASS-DISCRIMINATIVE kernels per class "
          f"(by={args.by}/{args.weighting}, metric={args.score}):")
    print(f"  {'favors':>6} {'rank':>4} {'score':>9} {'peakHz':>7} "
          f"{'frac_' + str(a):>8} {'frac_' + str(b):>8} {'dil':>5} {'repr':>5}")
    for c in (a, b):
        for info in per_class[c][:args.per_class]:
            print(f"  {str(info.favors):>6} {info.rank:>4} {info.score:>+9.3f} {_peak(info):>7} "
                  f"{info.fractions[a]:>8.3f} {info.fractions[b]:>8.3f} "
                  f"{info.dilation:>5} {info.representation:>5}")

    out = Path(args.out_dir)
    tag = f"{args.by}_{args.weighting}"
    dtag = f"{tag}_{args.score}"
    flat = per_class[a] + per_class[b]
    plot_set = {c: per_class[c][:args.per_class] for c in (a, b)}
    kernel_viz.plot_kernels(
        top, args.sfreq, out / f"top_kernels_{tag}.png", "Top HYDRA kernels (global)"
    )
    kernel_viz.plot_kernels_by_class(
        plot_set, args.sfreq, out / f"top_discriminative_kernels_{dtag}.png",
        "Top discriminative HYDRA kernels",
    )
    kernel_viz.plot_peak_freq_hist(
        flat, args.sfreq, out / f"discriminative_peak_freq_hist_{dtag}.png"
    )
    print(f"\nSaved plots to {args.out_dir}/ (if matplotlib is available)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
