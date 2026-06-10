"""Stage 5 - Sparse scaling of HYDRA features.

INPUT  : a HYDRA feature matrix F (batch, n_features) - sparse, non-negative
         counts. Produced here by running Stage 4's transform on synthetic
         windows so the distribution is realistic.
OUTPUT : the scaled matrix Fs (same shape) from `_SparseScaler.fit_transform`,
         plus the fitted statistics (mu, sigma, epsilon) the scaler stores.

The scaler square-roots the counts, standardises per feature, and (with
mask=True) keeps originally-zero entries at zero, so the sparsity pattern is
preserved. No corpus needed.

Run:
    uv run python tests/stage5_sparse_scaler.py --batch 96 --n-groups 32
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
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--channels", type=int, default=19)
    p.add_argument("--timepoints", type=int, default=512)
    p.add_argument("--n-groups", type=int, default=32)
    p.add_argument("--n-kernels", type=int, default=8)
    p.add_argument("--exponent", type=int, default=4)
    p.add_argument("--no-mask", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    _banner("STAGE 5 - Sparse scaling of HYDRA features")

    def _imports():
        import torch
        from src.models.components.hydra_transformer import HydraTransformer, _SparseScaler
        return torch, HydraTransformer, _SparseScaler

    mods = _need(_imports, "torch, numpy, aeon")
    if mods is None:
        return 1
    torch, HydraTransformer, _SparseScaler = mods

    torch.manual_seed(args.seed)
    x = torch.randn(args.batch, args.channels, args.timepoints)
    x2 = torch.randn(args.batch, args.channels, args.timepoints)  # a 2nd "split"
    fe = HydraTransformer(n_kernels=args.n_kernels, n_groups=args.n_groups,
                          random_state=args.seed)
    f = fe(x)
    f2 = fe(x2)

    _sec("INPUT")
    _kv("mask", not args.no_mask)
    _kv("exponent", args.exponent)
    _desc_tensor("F (HYDRA features in)", f)

    scaler = _SparseScaler(mask=not args.no_mask, exponent=args.exponent)
    fs = scaler.fit_transform(f)

    _sec("fitted statistics")
    _desc_tensor("scaler.mu", scaler.mu)
    _desc_tensor("scaler.sigma", scaler.sigma)
    _desc_tensor("scaler.epsilon", scaler.epsilon)

    _banner("OUTPUT")
    _desc_tensor("Fs (scaled features out)", fs)

    _sec("effect of scaling")
    # On the columns that are ever active, standardised features are ~0 mean / ~1 std.
    active_cols = (f != 0).any(dim=0)
    if active_cols.any():
        active = fs[:, active_cols]
        _kv("active columns", int(active_cols.sum()))
        _kv("Fs[active] col-mean (avg)", f"{active.mean(0).mean():.4g}")
        _kv("Fs[active] col-std  (avg)", f"{active.std(0).mean():.4g}")
    zeros_in = (f == 0)
    zeros_out = (fs == 0)
    _kv("zero pattern preserved", bool((zeros_in <= zeros_out).all()) if not args.no_mask
        else "n/a (mask off)")

    _sec("transform a second batch (train -> test reuse)")
    fs2 = scaler.transform(f2)
    _desc_tensor("scaler.transform(F2)", fs2)

    _sec("-> flows to Stage 6")
    print("  # In the real pipeline the scaler is step 0 of make_pipeline(scaler,")
    print("  # classifier); the classifier consumes Fs directly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
