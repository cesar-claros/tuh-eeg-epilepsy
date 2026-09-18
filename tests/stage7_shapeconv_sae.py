"""Stage 7 - ShapeConv sparse autoencoder (learned shapelet dictionary).

INPUT  : window batches X, float32 tensor (batch, channels, timepoints), served by
         a DataLoader of synthetic EEG-like rows: AR(1) background plus a planted
         spike-and-wave template at random positions in half of the windows.
OUTPUT : a feature matrix F, float32 tensor (batch, 2 * n_atoms): per-atom event
         counts summed over channels, then the per-atom best cosine match.

This stage needs no corpus. It also:
  - fits the dictionary without labels (k-means init, TopK reconstruction),
  - checks that one atom recovers the planted template (max cross-correlation
    over lags) and that its event count separates windows with / without events,
  - confirms the feature dimension 2 * n_atoms and the seed behaviour of the init.

Run:
    uv run python tests/stage7_shapeconv_sae.py --n-windows 64 --epochs 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
        print(f"  # needs: {pkgs}; run `uv sync` in code/")
        return None


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-windows", type=int, default=64)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--timepoints", type=int, default=2560, help="10 s at 256 Hz")
    p.add_argument("--template-len", type=int, default=64)
    p.add_argument("--atom-len", type=int, default=96, help="longer than the template, so any offset fits")
    p.add_argument("--events-per-channel", type=int, default=3)
    p.add_argument("--amplitude", type=float, default=8.0, help="event peak in background-std units")
    p.add_argument("--n-atoms", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="tests/outputs/stage7", help="where the checkpoint round trip writes")
    return p.parse_args(argv)


def _spike_wave(np, length: int):
    """Unit-norm, zero-mean spike (narrow) followed by a slow wave of opposite sign."""
    t = np.linspace(0.0, 1.0, length)
    template = np.exp(-(((t - 0.2) / 0.03) ** 2)) - 0.5 * np.exp(-(((t - 0.6) / 0.15) ** 2))
    template = template - template.mean()
    return template / np.linalg.norm(template)


def _synthetic(np, args):
    """AR(1) background rows with the template planted in the second half of the windows."""
    rng = np.random.default_rng(args.seed)
    n, c, t = args.n_windows, args.channels, args.timepoints
    noise = rng.standard_normal((n, c, t)).astype(np.float32)
    x = np.empty_like(noise)
    x[..., 0] = noise[..., 0]
    for i in range(1, t):
        x[..., i] = 0.95 * x[..., i - 1] + noise[..., i]
    x /= x.std(axis=-1, keepdims=True)
    template = _spike_wave(np, args.template_len)
    scale = args.amplitude / template.max()
    y = np.zeros(n, dtype=np.int64)
    y[n // 2:] = 1
    for w in np.flatnonzero(y):
        for ch in range(c):
            for start in rng.integers(0, t - args.template_len, size=args.events_per_channel):
                x[w, ch, start:start + args.template_len] += scale * template
    return x, y, template


def main(argv=None) -> int:
    args = parse(argv)
    _banner("STAGE 7 - ShapeConv sparse autoencoder")

    def _imports():
        import numpy as np
        import torch
        from sklearn.metrics import roc_auc_score
        from torch.utils.data import DataLoader, TensorDataset

        from src.models.components.shapeconv_sae import CHECKPOINT_NAME, AtomSpec, ShapeConvSAE, TrainSpec

        return np, torch, roc_auc_score, DataLoader, TensorDataset, AtomSpec, ShapeConvSAE, TrainSpec, CHECKPOINT_NAME

    mods = _need(_imports, "torch, numpy, scikit-learn, polars, loguru")
    if mods is None:
        return 1
    np, torch, roc_auc_score, DataLoader, TensorDataset, AtomSpec, ShapeConvSAE, TrainSpec, CHECKPOINT_NAME = mods

    x_np, y_np, template = _synthetic(np, args)
    x, y = torch.from_numpy(x_np), torch.from_numpy(y_np)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch, shuffle=False)

    _sec("INPUT")
    _desc_tensor("X (window batch)", x)
    _kv("windows with events", int(y.sum()))
    _kv("events per channel", args.events_per_channel)
    _kv("template length", args.template_len)

    spec = AtomSpec(n_atoms=args.n_atoms, atom_len=args.atom_len, rho_min=0.5)
    train_spec = TrainSpec(
        epochs=args.epochs, lr=args.lr, topk=args.topk, crop_len=256, crops_per_row=16,
        crop_batch=512, lambda_div=0.01, n_init_samples=2000,
    )
    sae = ShapeConvSAE(spec=spec, train_spec=train_spec, random_state=args.seed, device="cpu")
    init_atoms = sae.atoms_numpy.copy()
    sae.fit_unsupervised(loader, val_dataloader=loader)

    _sec("training history")
    for record in sae.history:
        _kv(
            f"epoch {int(record['epoch'])}",
            f"loss_rec={record['loss_rec']:.4f} val={record['val_loss_rec']:.4f} dead={int(record['n_dead'])}",
        )

    _sec("template recovery (max cross-correlation over lags)")
    atoms = sae.atoms_numpy
    xcorr = np.array([np.correlate(a, template, mode="full").max() for a in atoms])
    best_atom = int(xcorr.argmax())
    _kv("per-atom max xcorr", [round(float(v), 3) for v in xcorr])
    _kv("best atom / xcorr", f"{best_atom} / {xcorr[best_atom]:.3f}")
    _kv("recovered (> 0.9)", bool(xcorr[best_atom] > 0.9))

    _banner("OUTPUT")
    f = sae(x)
    _desc_tensor("F (SAE features)", f)
    _kv("2 * n_atoms", 2 * args.n_atoms)
    _kv("F.shape[1] match", f.shape[1] == 2 * args.n_atoms)

    _sec("best atom separates windows with / without events")
    counts = f[:, best_atom].numpy()
    _kv("mean count, no events", f"{counts[y_np == 0].mean():.2f}")
    _kv("mean count, events", f"{counts[y_np == 1].mean():.2f}")
    _kv("count AUROC", f"{roc_auc_score(y_np, counts):.3f}")
    _kv("best-cosine AUROC", f"{roc_auc_score(y_np, f[:, args.n_atoms + best_atom].numpy()):.3f}")

    _sec("seed behaviour (initial atoms, before fitting)")
    same = ShapeConvSAE(spec=spec, train_spec=train_spec, random_state=args.seed, device="cpu")
    diff = ShapeConvSAE(spec=spec, train_spec=train_spec, random_state=args.seed + 1, device="cpu")
    _kv("same seed -> identical", bool(np.array_equal(init_atoms, same.atoms_numpy)))
    _kv("different seed -> differs", not np.array_equal(init_atoms, diff.atoms_numpy))

    _sec("checkpoint round trip (save_artifacts -> pretrained=...)")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sae.train_subjects = ["synthetic"]
    sae.save_artifacts(out_dir)
    reloaded = ShapeConvSAE(spec=spec, train_spec=train_spec, device="cpu", pretrained=out_dir / CHECKPOINT_NAME)
    _kv("fitted after load", reloaded.fitted)
    reloaded.fit_unsupervised(loader)
    _kv("fit skipped (epochs kept)", len(reloaded.history) == len(sae.history))
    _kv("identical features", bool(torch.equal(reloaded(x), f)))
    _kv("train_subjects", reloaded.train_subjects)

    _sec("-> flows to Stage 5")
    print("  # Event counts are sparse non-negative, like HYDRA counts; the")
    print("  # _SparseScaler standardises them before the linear classifier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
