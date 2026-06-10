"""Toy: one HYDRA group, end to end (gather -> sum -> conv -> argmax/min -> counts).

A tiny, instrumented walk-through of the multichannel HYDRA feature computation
that prints every intermediate tensor, so the gather/sum/conv/argmax steps can be
read off by hand. It uses the real ``HydraTransform`` internals (its kernels and
channel index sets), so the numbers match the production forward pass, and it
verifies that the manual walk reproduces ``HydraTransform.forward`` exactly.

Settings: C=4 channels, T=6 timepoints, g=4 groups (so divisor=2 and h=2 groups
per representation), k=2 kernels per group. Only needs torch + numpy
(``HydraTransform`` has no aeon dependency).

Run:
    uv run python tests/toy_hydra_group.py
"""

from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)


def _p(title: str, t: "torch.Tensor") -> None:  # noqa: F821 (torch imported in main)
    """Print a labelled tensor with its shape."""
    print(f"\n{title}   shape={tuple(t.shape)}")
    print(t)


def main() -> int:
    """Run the toy walk-through and return a process exit code."""
    try:
        import torch
        import torch.nn.functional as F

        from src.models.components.hydra_transform import HydraTransform
    except ImportError as exc:
        print(f"This toy needs torch + numpy: {exc}")
        print("Run it inside the project container / uv environment.")
        return 1

    torch.set_printoptions(precision=3, sci_mode=False)

    # Input: 1 window, 4 channels, 6 timepoints. Readable integers so the channel
    # sums in step 3 can be checked by eye.
    x = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # ch0
                [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],  # ch1
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],  # ch2
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],  # ch3
            ]
        ]
    )

    print("=" * 78)
    print("HYDRA toy: C=4, T=6, g=4 (=> divisor=2, h=2 groups per representation), k=2")
    print("=" * 78)
    _p("X (N, C, T)", x)

    hydra = HydraTransform(n_timepoints=6, n_channels=4, k=2, g=4, seed=0)
    print("\n--- setup (HydraTransform.__init__) ---")
    print(f"  num_dilations          = {hydra.num_dilations}  dilations={hydra.dilations.tolist()}")
    print(f"  paddings               = {hydra.paddings.tolist()}")
    print(f"  divisor (raw + diff)   = {hydra.divisor}")
    print(f"  h (groups per repr.)   = {hydra.h}")
    print(f"  k (kernels per group)  = {hydra.k}")
    print(f"  W (kernels) shape      = {tuple(hydra.W.shape)}  (num_dilations, divisor, k*h, 1, 9)")
    print(f"  channels summed/group  = {hydra.idx[0].shape[-1]}  (= clip(C//2, 2, max_num_channels))")

    # Walk dilation 0, RAW representation (diff_index = 0), verbosely.
    dilation_index, diff_index = 0, 0
    d = hydra.dilations[dilation_index].item()
    p = hydra.paddings[dilation_index].item()
    print("\n" + "=" * 78)
    print(f"WALK: dilation index {dilation_index} (dilation={d}, padding={p}), RAW representation")
    print("=" * 78)

    # 1) gather: each group's fixed random channel subset.
    idx = hydra.idx[dilation_index][diff_index]  # (h, channels_per_group)
    _p("1) idx[dilation][raw]  (h groups, channels per group)", idx)
    print(f"   -> group 0 sums channels {idx[0].tolist()}; group 1 sums channels {idx[1].tolist()}")

    gathered = x[:, idx]  # (N, h, channels_per_group, T)
    _p("2) gather X[:, idx]  (N, h, channels_per_group, T)", gathered)

    # 3) sum across the channel-subset axis -> one mixed series per group.
    mixed = gathered.sum(2)  # (N, h, T)
    _p("3) sum over the channel subset -> mixed series per group  (N, h, T)", mixed)
    print(f"   check group 0: X[ch{idx[0,0].item()}] + X[ch{idx[0,1].item()}] = {mixed[0, 0].tolist()}")

    # 4) grouped conv: group i's mixed series is convolved with group i's k kernels.
    weight = hydra.W[dilation_index][diff_index]  # (k*h, 1, 9)
    conv = F.conv1d(mixed, weight, dilation=d, padding=p, groups=hydra.h)  # (N, h*k, T)
    _p("4) conv1d(mixed, W, groups=h)  (N, h*k, T)", conv)

    z = conv.view(1, hydra.h, hydra.k, -1)  # (N, h, k, T)
    _p("5) view as (N, h, k, T): the k competing kernels per group", z)

    # 6) competition: argmax / argmin over the k kernels at each timepoint.
    max_values, max_indices = z.max(2)  # (N, h, T)
    min_values, min_indices = z.min(2)  # (N, h, T)
    _p("6a) argmax kernel per (group, time)", max_indices)
    _p("6b) winning max values", max_values)
    _p("6c) argmin kernel per (group, time)", min_indices)
    print("   (with k=2 the two kernels are simply each other's max and min at every timepoint)")

    # 7) counts: max is soft (sum of winning values), min is hard (count of wins).
    count_max = torch.zeros(1, hydra.h, hydra.k)
    count_max.scatter_add_(-1, max_indices, max_values)
    count_min = torch.zeros(1, hydra.h, hydra.k)
    count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))
    _p("7a) count_max (per group, per kernel: SUM of winning max-values)", count_max)
    _p("7b) count_min (per group, per kernel: COUNT of argmin wins)", count_min)
    print(f"\n   Each group emits [count_max(k), count_min(k)] = 2k = {2 * hydra.k} numbers.")

    # Verify the instrumented walk reproduces HydraTransform.forward(X). With a
    # single dilation, forward stacks [raw_max, raw_min, diff_max, diff_min].
    diff_x = torch.diff(x)  # (N, C, T-1)
    idx_d = hydra.idx[dilation_index][1]
    mixed_d = diff_x[:, idx_d].sum(2)
    conv_d = F.conv1d(mixed_d, hydra.W[dilation_index][1], dilation=d, padding=p, groups=hydra.h)
    z_d = conv_d.view(1, hydra.h, hydra.k, -1)
    dmax_v, dmax_i = z_d.max(2)
    dmin_v, dmin_i = z_d.min(2)
    count_max_d = torch.zeros(1, hydra.h, hydra.k)
    count_max_d.scatter_add_(-1, dmax_i, dmax_v)
    count_min_d = torch.zeros(1, hydra.h, hydra.k)
    count_min_d.scatter_add_(-1, dmin_i, torch.ones_like(dmin_v))

    manual = torch.cat([count_max, count_min, count_max_d, count_min_d], 1).view(1, -1)
    feats = hydra(x)

    print("\n" + "=" * 78)
    print("VERIFY against HydraTransform.forward(X)")
    print("=" * 78)
    print(f"  forward output shape   = {tuple(feats.shape)}")
    print(f"  expected n_features    = num_dilations*divisor*2*h*k = "
          f"{hydra.num_dilations * hydra.divisor * 2 * hydra.h * hydra.k}")
    print(f"  manual walk == forward : {bool(torch.equal(manual, feats))}")

    # Single-channel contrast: no gather/sum, and a plain conv (groups=1).
    print("\n" + "=" * 78)
    print("CONTRAST: single-channel path (C=1) skips gather/sum and uses groups=1")
    print("=" * 78)
    hydra_uni = HydraTransform(n_timepoints=6, n_channels=1, k=2, g=4, seed=0)
    x_uni = x[:, :1]  # (N, 1, T): just one channel, fed in directly
    conv_uni = F.conv1d(x_uni, hydra_uni.W[0][0], dilation=1, padding=hydra_uni.paddings[0].item())
    print(f"  input is used directly (no idx, no sum): X_uni shape {tuple(x_uni.shape)}")
    print(f"  conv1d(X_uni, W, groups=1) -> {tuple(conv_uni.shape)}  (k*h kernels on the one series)")
    print(f"  same downstream (N, h, k, T) structure: {tuple(conv_uni.view(1, hydra_uni.h, hydra_uni.k, -1).shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
