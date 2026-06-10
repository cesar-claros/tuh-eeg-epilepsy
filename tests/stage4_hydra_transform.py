"""Stage 4 - HYDRA feature transform.

INPUT  : a window batch X, float32 tensor (batch, channels, timepoints).
         (Synthetic here, but the same shape the dataloaders emit.)
OUTPUT : a feature matrix F, float32 tensor (batch, n_features), where each row
         is the concatenation of per-group "which kernel won how often" counts
         across dilations, for both the raw signal and its first difference.

This stage needs no corpus. It also:
  - shows the lazy build of the underlying HydraTransform (kernels / dilations),
  - verifies the feature-dimension formula
        n_features = num_dilations * divisor * 2 * h * k,
  - confirms the seed behaviour (same seed -> identical, different -> different).

Run:
    uv run python tests/stage4_hydra_transform.py --batch 8 --n-groups 64 --n-kernels 8
"""

from __future__ import annotations

import argparse

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)


def _banner(t: str) -> None:
    print("\n" + "=" * 88 + "\n" + t + "\n" + "=" * 88)


def _sec(t: str) -> None:
    print("\n" + t + "\n" + "-" * max(len(t), 8))


def _kv(k: str, v) -> None:
    print(f"  {k:<28}: {v}")


def _desc_tensor(name: str, t, n: int = 6) -> None:
    _sec(f"[tensor] {name}")
    _kv("shape", tuple(t.shape))
    _kv("dtype", t.dtype)
    _kv("numel", t.numel())
    if t.numel() == 0:
        return
    tf = t.detach().float()
    _kv("min/max", f"{tf.min():.4g} / {tf.max():.4g}")
    _kv("mean/std", f"{tf.mean():.4g} / {tf.std():.4g}")
    _kv("zeros", f"{(t == 0).float().mean() * 100:.1f}%")
    _kv(f"first {n}", [round(float(v), 4) for v in tf.flatten()[:n]])


def _need(import_fn, pkgs: str):
    try:
        return import_fn()
    except ImportError as exc:
        _sec("missing dependency")
        _kv("error", exc)
        print(f"  # needs: {pkgs}; run `uv sync` in code/ (add aeon if not declared)")
        return None


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--channels", type=int, default=19)
    p.add_argument("--timepoints", type=int, default=512)
    p.add_argument("--n-groups", type=int, default=64)
    p.add_argument("--n-kernels", type=int, default=8)
    p.add_argument("--max-num-channels", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    _banner("STAGE 4 - HYDRA feature transform")

    def _imports():
        import torch
        from src.models.components.hydra_transformer import HydraTransformer
        return torch, HydraTransformer

    mods = _need(_imports, "torch, numpy, aeon")
    if mods is None:
        return 1
    torch, HydraTransformer = mods

    torch.manual_seed(args.seed)
    x = torch.randn(args.batch, args.channels, args.timepoints)

    _sec("INPUT")
    _desc_tensor("X (window batch)", x)

    fe = HydraTransformer(
        n_kernels=args.n_kernels,
        n_groups=args.n_groups,
        max_num_channels=args.max_num_channels,
        n_jobs=1,
        random_state=args.seed,
    )
    _kv("lazy-built yet?", fe._hydra is not None)  # noqa: SLF001

    f = fe(x)

    _sec("internal HydraTransform (after lazy build)")
    h = fe._hydra  # noqa: SLF001
    _kv("num_dilations", h.num_dilations)
    _kv("dilations", h.dilations.tolist())
    _kv("divisor (raw + diff)", h.divisor)
    _kv("groups-per-divisor h", h.h)
    _kv("kernels-per-group k", h.k)
    _kv("W (kernels) shape", tuple(h.W.shape))
    _kv("idx (channel picks) len", len(h.idx))

    _banner("OUTPUT")
    _desc_tensor("F (HYDRA features)", f)

    _sec("feature-dimension formula")
    expected = h.num_dilations * h.divisor * 2 * h.h * h.k
    _kv("num_dilations*divisor*2*h*k", expected)
    _kv("F.shape[1]", f.shape[1])
    _kv("match", expected == f.shape[1])

    _sec("seed behaviour")
    f_same = HydraTransformer(n_kernels=args.n_kernels, n_groups=args.n_groups,
                              max_num_channels=args.max_num_channels,
                              random_state=args.seed)(x)
    f_diff = HydraTransformer(n_kernels=args.n_kernels, n_groups=args.n_groups,
                              max_num_channels=args.max_num_channels,
                              random_state=args.seed + 1)(x)
    _kv("same seed -> identical", bool(torch.equal(f, f_same)))
    _kv("different seed -> differs", not bool(torch.equal(f, f_diff)))

    _sec("-> flows to Stage 5")
    print("  # These sparse non-negative counts are standardised by the")
    print("  # _SparseScaler before the linear classifier sees them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
