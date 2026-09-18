"""Stage 7 - ShapeConv sparse autoencoder (learned shapelet dictionary).

INPUT  : three INDEPENDENT synthetic draws (train / val / test) of window batches X,
         float32 (batch, channels, timepoints): AR(1) background plus a planted
         spike-and-wave template of RANDOM POLARITY at random positions in half of
         the windows (signed codes must cover both polarities with one atom).
OUTPUT : a feature matrix F, float32 tensor (batch, 3 * n_atoms): per-atom
         channel-activation counts, per-atom peak |coefficient|, per-atom max |cosine|.

Deterministic invariants FAIL the run (exit code 1):
  - feature dimension 3 * n_atoms and equality across chunk sizes,
  - same seed -> same fitted atoms (two short single-threaded fits, tolerance 1e-4;
    multi-threaded CPU convolutions are not bit-reproducible), different seed -> differs,
  - checkpoint round trip: restored spec, identical features, skipped second fit,
  - a checkpoint without provenance is refused.
Stochastic recovery GATE (decision of 2026-09-18 after Phase 1 campaign 1; fails the
run only with --strict): event-level matching on the test draw. The atom, its
activation-to-onset offset and its polarity sign are calibrated on the VALIDATION
draw (best validation F1); on the test draw, one-to-one matching within --match-tol
samples must reach precision > 0.9 and recall > 0.9.
Reported DIAGNOSTICS (never fail the run): the single-atom template correlation
(max |xcorr| in the encoded domain; demoted from a gate because feature splitting
carves one event across atoms, so no single atom need equal the waveform), the
window AUROCs of the best atom's count and peak features, the per-atom table,
recall on isolated versus close events, the dictionary-level event reconstruction
correlation, timing error, count MAE, duplicates and false alarms per channel-minute.

Stress conditions (targeted runs, not a grid): --amplitude 4 / 2 (event peak in
background std), --event-free (no events: reports false alarms per channel-minute
only; the gate does not apply), --band-pass (corpus-matched 1-45 Hz zero-phase
filter after planting), --second-morphology (half the events use a sharp wave;
recall is reported per template), --pair-gap G (every event gets a partner G
samples later on the same channel), --artifacts (a broad bump on a random channel
in 30 percent of ALL windows), --synchronous (each event on every channel at the
same onset with a random polarity per channel).

Run:
    uv run python tests/stage7_shapeconv_sae.py --n-windows 64 --epochs 20 --strict
    uv run python tests/stage7_shapeconv_sae.py --pre-emphasis none      # raw rows: background dominates
    uv run python tests/stage7_shapeconv_sae.py --mode topk --topk 2 --strict
    uv run python tests/stage7_shapeconv_sae.py --amplitude 4 --band-pass --artifacts --strict
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

REFERENCE_XCORR = 0.9   # diagnostic reference, not a gate
REFERENCE_AUROC = 0.9   # diagnostic reference, not a gate
RECOVERY_MIN_PR = 0.9   # the gate: event precision and recall
SFREQ = 256.0
REPRO_ATOL = 1e-4
REPRO_EPOCHS = 5


def _banner(t: str) -> None:
    print("\n" + "=" * 88 + "\n" + t + "\n" + "=" * 88)


def _sec(t: str) -> None:
    print("\n" + t + "\n" + "-" * max(len(t), 8))


def _kv(k: str, v) -> None:
    print(f"  {k:<32}: {v}")


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


class _Checks:
    """Collects named pass/fail results; deterministic failures fail the run."""

    def __init__(self) -> None:
        self.failed: list[str] = []
        self.soft_failed: list[str] = []

    def hard(self, name: str, ok: bool) -> None:
        _kv(name, "PASS" if ok else "FAIL")
        if not ok:
            self.failed.append(name)

    def soft(self, name: str, ok: bool, detail: str) -> None:
        _kv(name, f"{'PASS' if ok else 'BELOW TOLERANCE'} ({detail})")
        if not ok:
            self.soft_failed.append(name)

    @staticmethod
    def diagnostic(name: str, ok: bool, detail: str) -> None:
        """Report against a reference value without affecting the outcome."""
        _kv(name, f"{'above' if ok else 'BELOW'} reference ({detail})")


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-windows", type=int, default=64, help="windows per draw")
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--timepoints", type=int, default=2560, help="10 s at 256 Hz")
    p.add_argument("--template-len", type=int, default=64)
    p.add_argument("--atom-len", type=int, default=96, help="longer than the template, so any offset fits")
    p.add_argument("--events-per-channel", type=int, default=3)
    p.add_argument("--amplitude", type=float, default=8.0, help="event peak in background-std units")
    p.add_argument("--n-atoms", type=int, default=8)
    p.add_argument("--mode", choices=("shrink", "topk"), default="shrink")
    p.add_argument("--thresh", type=float, default=3.0, help="initial soft threshold (shrink)")
    p.add_argument("--lam", type=float, default=0.5, help="L1 weight (shrink)")
    p.add_argument("--topk", type=int, default=3, help="code entries per crop (topk)")
    p.add_argument("--amp-min", type=float, default=4.0, help="extraction: count activations with |a| > amp_min")
    p.add_argument("--pre-emphasis", choices=("diff", "none"), default="diff")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--match-tol", type=int, default=6, help="event matching tolerance in samples (engineering choice)")
    p.add_argument("--event-free", action="store_true", help="plant no events; report false alarms only")
    p.add_argument("--band-pass", action="store_true", help="1-45 Hz zero-phase Butterworth at 256 Hz after planting")
    p.add_argument("--second-morphology", action="store_true", help="half of the events use a sharp-wave template")
    p.add_argument("--pair-gap", type=int, default=0, help="if > 0, every event gets a partner this many samples later")
    p.add_argument("--artifacts", action="store_true", help="broad bump (100 samples, 4 std) in 30%% of all windows")
    p.add_argument("--synchronous", action="store_true", help="each event appears on every channel at the same onset")
    p.add_argument("--strict", action="store_true", help="also fail on the stochastic recovery checks")
    p.add_argument("--out-dir", default="tests/outputs/stage7", help="where the checkpoint round trip writes")
    return p.parse_args(argv)


def _spike_wave(np, length: int, sharp: bool = False):
    """Unit-norm, zero-mean spike (narrow) or sharp wave (broader) followed by a slow wave of opposite sign."""
    t = np.linspace(0.0, 1.0, length)
    width = 0.09 if sharp else 0.03
    template = np.exp(-(((t - 0.2) / width) ** 2)) - 0.5 * np.exp(-(((t - 0.6) / 0.15) ** 2))
    template = template - template.mean()
    return template / np.linalg.norm(template)


def _synthetic(np, args, seed: int):
    """One independent draw: AR(1) background rows, template of random polarity in half the windows.

    Returns the rows, the window labels, the template, and the planted events as an
    integer array with columns (window, channel, onset sample, polarity).
    """
    rng = np.random.default_rng(seed)
    n, c, t = args.n_windows, args.channels, args.timepoints
    noise = rng.standard_normal((n, c, t)).astype(np.float32)
    x = np.empty_like(noise)
    x[..., 0] = noise[..., 0]
    for i in range(1, t):
        x[..., i] = 0.95 * x[..., i - 1] + noise[..., i]
    x /= x.std(axis=-1, keepdims=True)
    templates = [_spike_wave(np, args.template_len)]
    if args.second_morphology:
        templates.append(_spike_wave(np, args.template_len, sharp=True))
    y = np.zeros(n, dtype=np.int64)
    if not args.event_free:
        y[n // 2:] = 1
    last_onset = t - args.template_len - max(args.pair_gap, 0)
    events = []

    def plant(w: int, ch: int, start: int, kind: int) -> None:
        polarity = int(rng.choice([-1, 1]))
        template = templates[kind]
        x[w, ch, start:start + args.template_len] += polarity * (args.amplitude / template.max()) * template
        events.append((w, ch, int(start), polarity, kind))

    for w in np.flatnonzero(y):
        # Synchronous fields draw the onsets once per window and plant them on every channel.
        for ch in ([None] if args.synchronous else range(c)):
            for start in rng.integers(0, last_onset, size=args.events_per_channel):
                kind = int(rng.integers(len(templates)))
                for target in (range(c) if ch is None else (ch,)):
                    plant(w, target, int(start), kind)
                    if args.pair_gap > 0:
                        plant(w, target, int(start) + args.pair_gap, kind)
    if args.artifacts:
        for w in range(n):
            if rng.random() < 0.3:
                onset = int(rng.integers(0, t - 100))
                x[w, rng.integers(c), onset:onset + 100] += 4.0 * np.hanning(100)
    if args.band_pass:
        from scipy.signal import butter, sosfiltfilt

        sos = butter(4, [1.0, 45.0], btype="bandpass", fs=SFREQ, output="sos")
        x = sosfiltfilt(sos, x, axis=-1).astype(np.float32)
    return x, y, templates[0], np.array(events, dtype=np.int64).reshape(-1, 5)


def _match_events(np, pred, true, tol: int):
    """Greedy one-to-one matching of predicted onsets to true onsets within ``tol`` samples.

    ``pred`` and ``true`` are integer arrays with columns (window, channel, sample).
    Pairs are taken in order of increasing timing error within each (window,
    channel). Returns the matched index arrays into ``pred`` and ``true``.
    """
    matched_p, matched_t = [], []
    keys = {tuple(k) for k in np.unique(np.concatenate([pred[:, :2], true[:, :2]]), axis=0)}
    for w, c in sorted(keys):
        pi = np.flatnonzero((pred[:, 0] == w) & (pred[:, 1] == c))
        ti = np.flatnonzero((true[:, 0] == w) & (true[:, 1] == c))
        if len(pi) == 0 or len(ti) == 0:
            continue
        error = np.abs(pred[pi, 2][:, None] - true[ti, 2][None, :])
        pairs = np.argwhere(error <= tol)
        used_p, used_t = set(), set()
        for a, b in pairs[np.argsort(error[pairs[:, 0], pairs[:, 1]], kind="stable")]:
            if a in used_p or b in used_t:
                continue
            used_p.add(a)
            used_t.add(b)
            matched_p.append(pi[a])
            matched_t.append(ti[b])
    return np.asarray(matched_p, dtype=np.int64), np.asarray(matched_t, dtype=np.int64)


def _calibrate_atom(np, acts, events, atom_len: int):
    """Offset (true onset minus activation sample) and polarity sign of one atom, from its nearest true events."""
    offsets, signs = [], []
    for w, c, sample, coef in acts:
        mask = (events[:, 0] == w) & (events[:, 1] == c)
        if not mask.any():
            continue
        nearest = np.argmin(np.abs(events[mask, 2] - sample))
        gap = events[mask, 2][nearest] - sample
        if abs(gap) <= atom_len:
            offsets.append(gap)
            signs.append(np.sign(coef) * events[mask, 3][nearest])
    if not offsets:
        return None
    return int(np.median(offsets)), int(np.sign(np.mean(signs)) or 1)


def _event_metrics(np, acts, events, offset: int, sign: int, tol: int, shape, atom_len: int):
    """Event-level scores of one atom's activations against the planted events.

    ``acts`` has columns (window, channel, sample, coefficient); ``events`` has
    (window, channel, onset, polarity); ``shape`` is (windows, channels, samples).
    """
    n_windows, n_channels, n_times = shape
    pred = np.column_stack([acts[:, 0], acts[:, 1], acts[:, 2] + offset]).astype(np.int64)
    true = events[:, :3]
    mp, mt = _match_events(np, pred, true, tol)
    n_pred, n_true, n_match = len(pred), len(true), len(mp)
    unmatched = np.setdiff1d(np.arange(n_pred), mp)
    duplicates = 0
    for i in unmatched:
        mask = (true[:, 0] == pred[i, 0]) & (true[:, 1] == pred[i, 1])
        if mask.any() and np.abs(true[mask, 2] - pred[i, 2]).min() <= tol:
            duplicates += 1
    minutes = n_windows * n_channels * n_times / SFREQ / 60.0
    true_counts = np.bincount(true[:, 0], minlength=n_windows)
    pred_counts = np.bincount(pred[:, 0], minlength=n_windows)
    polarity_ok = float(np.mean(np.sign(acts[mp, 3]) * sign == events[mt, 3])) if n_match else float("nan")
    # An event is "close" when another true event on the same channel starts within atom_len
    # samples: NMS keeps one activation per neighbourhood, so close pairs are where misses are expected.
    isolated = np.ones(n_true, dtype=bool)
    for i in range(n_true):
        same = (true[:, 0] == true[i, 0]) & (true[:, 1] == true[i, 1])
        same[i] = False
        if same.any() and np.abs(true[same, 2] - true[i, 2]).min() <= atom_len:
            isolated[i] = False
    matched_true = np.zeros(n_true, dtype=bool)
    matched_true[mt] = True
    per_template = {}
    if n_true and events.shape[1] > 4:
        per_template = {int(k): float(matched_true[events[:, 4] == k].mean()) for k in np.unique(events[:, 4])}
    return {
        "recall_by_template": per_template,
        "precision": n_match / n_pred if n_pred else float("nan"),
        "recall": n_match / n_true if n_true else float("nan"),
        "recall_isolated": float(matched_true[isolated].mean()) if isolated.any() else float("nan"),
        "recall_close": float(matched_true[~isolated].mean()) if (~isolated).any() else float("nan"),
        "n_close": int((~isolated).sum()),
        "timing_error_samples": float(np.abs(pred[mp, 2] - true[mt, 2]).mean()) if n_match else float("nan"),
        "count_mae_per_window": float(np.abs(true_counts - pred_counts).mean()),
        "duplicates": duplicates,
        "false_alarms_per_channel_minute": (len(unmatched) - duplicates) / minutes,
        "polarity_accuracy": polarity_ok,
        "n_true": n_true,
        "n_pred": n_pred,
    }


def _event_reconstruction_corr(torch, np, sae, x, events, atom_len: int) -> float:
    """Mean correlation between the sparse reconstruction and the encoded row around each true event.

    A dictionary-level recovery measure: it credits an event that is carved across
    several atoms, unlike the single-atom template correlation. The segment runs
    from ``onset - atom_len // 2`` to ``onset + atom_len + atom_len // 2`` in the
    encoded row (under ``diff`` the onset index is the same up to one sample).
    """
    rows = sae._rows(x)
    n_channels = x.shape[1]
    atoms = sae._project(sae.atoms)
    correlations = []
    with torch.no_grad():
        for start, code, _ in sae._row_chunks(x):
            recon = torch.nn.functional.conv_transpose1d(code, atoms).squeeze(1)
            for local in range(code.shape[0]):
                row = start + local
                w, c = divmod(row, n_channels)
                for onset in events[(events[:, 0] == w) & (events[:, 1] == c), 2]:
                    lo = max(int(onset) - atom_len // 2, 0)
                    hi = min(int(onset) + atom_len + atom_len // 2, rows.shape[1])
                    a, b = rows[row, lo:hi], recon[local, lo:hi]
                    a, b = a - a.mean(), b - b.mean()
                    denom = float(a.norm() * b.norm())
                    correlations.append(float((a * b).sum()) / denom if denom > 0 else 0.0)
    return float(np.mean(correlations)) if correlations else float("nan")


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
    checks = _Checks()

    def _loader(seed: int):
        x_np, y_np, template, events = _synthetic(np, args, seed)
        x, y = torch.from_numpy(x_np), torch.from_numpy(y_np)
        return DataLoader(TensorDataset(x, y), batch_size=args.batch, shuffle=False), x, y_np, template, events

    train_loader, x_train, _, template, _ = _loader(args.seed)
    val_loader, x_val, _, _, events_val = _loader(args.seed + 1)
    _, x_test, y_test, _, events_test = _loader(args.seed + 2)

    _sec("INPUT (train draw; val and test are independent draws)")
    _desc_tensor("X (window batch)", x_train)
    _kv("windows with events per draw", 0 if args.event_free else int(args.n_windows - args.n_windows // 2))
    stress_flags = ("event_free", "band_pass", "second_morphology", "artifacts", "synchronous")
    stress = [k for k in stress_flags if getattr(args, k)]
    if args.pair_gap:
        stress.append(f"pair_gap {args.pair_gap}")
    _kv("stress", ", ".join(stress) or "none")
    _kv("events per channel", args.events_per_channel)
    _kv("template length / atom length", f"{args.template_len} / {args.atom_len}")
    _kv("mode / pre-emphasis", f"{args.mode} / {args.pre_emphasis}")

    spec = AtomSpec(
        n_atoms=args.n_atoms, atom_len=args.atom_len, mode=args.mode, thresh=args.thresh,
        topk=args.topk, amp_min=args.amp_min, pre_emphasis=args.pre_emphasis,
    )
    train_spec = TrainSpec(
        epochs=args.epochs, lr=args.lr, lam=args.lam, crop_len=256, crops_per_row=16,
        crop_batch=512, n_init_samples=2000,
    )
    provenance = {"train_subjects": ["synthetic-train"], "data": {"signal_mode": "synthetic"}}

    def _fit(seed: int, epochs: int = args.epochs, monitor: bool = True):
        schedule = TrainSpec(**{**train_spec.__dict__, "epochs": epochs})
        model = ShapeConvSAE(spec=spec, train_spec=schedule, random_state=seed, device="cpu")
        model.fit_unsupervised(train_loader, val_dataloader=val_loader if monitor else None, provenance=provenance)
        return model

    sae = _fit(args.seed)

    _sec("training history")
    for record in sae.history:
        _kv(
            f"epoch {int(record['epoch'])}",
            f"residual {record['residual_frac']:.1%} (val {record['val_residual_frac']:.1%}) "
            f"starts/crop {record['active_per_crop']:.2f} objective {record['objective']:.4f} "
            f"thresh/response {record['median_thresh_over_response']:.2f} "
            f"dead={int(record['n_dead'])} reseeded={int(record['n_reseeded'])}",
        )
    if args.mode == "shrink":
        _kv("learned thresholds", [round(float(v), 2) for v in sae.thresholds])
    _kv("val response std per atom", [round(float(v), 2) for v in sae.atom_response_std])

    _banner("OUTPUT")
    f = sae(x_test)
    _desc_tensor("F (SAE features, test draw)", f)

    _sec("deterministic invariants")
    checks.hard("F.shape[1] == 3 * n_atoms", f.shape[1] == 3 * args.n_atoms)
    sae.chunk_rows = 1
    checks.hard("chunk_rows=1 gives the same F", bool(torch.allclose(sae(x_test), f, atol=1e-5)))
    sae.chunk_rows = 16
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    fit_a, fit_b = _fit(args.seed, REPRO_EPOCHS, monitor=False), _fit(args.seed, REPRO_EPOCHS, monitor=False)
    torch.set_num_threads(n_threads)
    max_diff = float(np.abs(fit_a.atoms_numpy - fit_b.atoms_numpy).max())
    checks.hard(f"same seed -> same fitted atoms (1 thread, {REPRO_EPOCHS} ep, max diff {max_diff:.1e})",
                max_diff <= REPRO_ATOL)
    seed_a = ShapeConvSAE(spec=spec, train_spec=train_spec, random_state=args.seed, device="cpu").atoms_numpy
    seed_b = ShapeConvSAE(spec=spec, train_spec=train_spec, random_state=args.seed + 1, device="cpu").atoms_numpy
    checks.hard("different seed -> different initial atoms", not np.array_equal(seed_a, seed_b))

    _sec("checkpoint round trip (save_artifacts -> pretrained=...)")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sae.save_artifacts(out_dir)
    other_spec = AtomSpec(
        n_atoms=args.n_atoms, atom_len=args.atom_len, mode=args.mode, pre_emphasis=args.pre_emphasis,
        rho_min=0.35, amp_min=2.0,
    )
    reloaded = ShapeConvSAE(spec=other_spec, train_spec=train_spec, device="cpu", pretrained=out_dir / CHECKPOINT_NAME)
    checks.hard("saved spec restored on load", reloaded.spec == sae.spec)
    checks.hard("fitted after load", reloaded.fitted)
    checks.hard("provenance restored", reloaded.train_subjects == ["synthetic-train"])
    reloaded.fit_unsupervised(train_loader, provenance={"train_subjects": ["other"]})
    checks.hard("second fit skipped, provenance kept", reloaded.train_subjects == ["synthetic-train"])
    checks.hard("identical features after reload", bool(torch.equal(reloaded(x_test), f)))
    overridden = ShapeConvSAE(
        spec=other_spec, train_spec=train_spec, device="cpu", pretrained=out_dir / CHECKPOINT_NAME, override_spec=True
    )
    checks.hard("override_spec keeps config gate", overridden.spec.rho_min == 0.35)
    checkpoint = torch.load(out_dir / CHECKPOINT_NAME, weights_only=True)
    checkpoint["provenance"] = {}
    torch.save(checkpoint, out_dir / "no_provenance.pt")
    try:
        ShapeConvSAE(spec=spec, train_spec=train_spec, device="cpu", pretrained=out_dir / "no_provenance.pt")
        refused = False
    except ValueError:
        refused = True
    checks.hard("checkpoint without provenance refused", refused)

    _sec("recovery diagnostics (held-out test draw; references, not gates)")
    atoms = sae.atoms_numpy
    target = np.diff(template) if args.pre_emphasis == "diff" else template
    target = (target - target.mean()) / np.linalg.norm(target - target.mean())
    xcorr = np.array([np.abs(np.correlate(a, target, mode="full")).max() for a in atoms])
    best_atom = int(xcorr.argmax())
    _kv("per-atom max |xcorr| (encoded domain)", [round(float(v), 3) for v in xcorr])
    signal_xcorr = np.abs(np.correlate(sae.atoms_signal_domain[best_atom], template, mode="full")).max()
    _kv("best atom, signal-domain |xcorr|", f"{signal_xcorr:.3f}")
    checks.diagnostic(f"single-atom template |xcorr| (ref {REFERENCE_XCORR})", bool(xcorr[best_atom] > REFERENCE_XCORR),
                      f"atom {best_atom}, |xcorr| {xcorr[best_atom]:.3f}")
    k = args.n_atoms
    counts, peaks, cosines = f[:, best_atom].numpy(), f[:, k + best_atom].numpy(), f[:, 2 * k + best_atom].numpy()
    if y_test.min() != y_test.max():
        _kv("mean count, no events / events", f"{counts[y_test == 0].mean():.2f} / {counts[y_test == 1].mean():.2f}")
        auc_count, auc_peak = roc_auc_score(y_test, counts), roc_auc_score(y_test, peaks)
        checks.diagnostic(f"count AUROC (ref {REFERENCE_AUROC})", auc_count > REFERENCE_AUROC, f"{auc_count:.3f}")
        checks.diagnostic(f"peak |a| AUROC (ref {REFERENCE_AUROC})", auc_peak > REFERENCE_AUROC, f"{auc_peak:.3f}")
        _kv("max |cosine| AUROC", f"{roc_auc_score(y_test, cosines):.3f}")
    else:
        _kv("window AUROCs", "not defined (one class)")
    _kv("events table (an event window)", sae.events(x_test[-1:]).head(5))

    _sec(f"event-level matching, THE GATE (calibrated on val, scored on test, tolerance {args.match_tol} samples)")
    shape = tuple(x_test.shape)
    acts_val, acts_test = sae.events(x_val).to_numpy(), sae.events(x_test).to_numpy()
    if args.event_free:
        minutes = shape[0] * shape[1] * shape[2] / SFREQ / 60.0
        per_atom = [float((acts_test[:, 2] == atom).sum()) / minutes for atom in range(args.n_atoms)]
        _kv("false alarms per channel-minute, per atom", [round(v, 3) for v in per_atom])
        _kv("false alarms per channel-minute, total", f"{sum(per_atom):.3f}")
        _kv("gate", "not applicable (no events planted)")
        _sec("RESULT")
        _kv("deterministic failures", checks.failed or "none")
        return 1 if checks.failed else 0
    calibration = {}
    for atom in range(args.n_atoms):
        acts = acts_val[acts_val[:, 2] == atom][:, [0, 1, 3, 4]]
        fit = _calibrate_atom(np, acts, events_val, args.atom_len) if len(acts) else None
        if fit is None:
            continue
        val_scores = _event_metrics(np, acts, events_val, fit[0], fit[1], args.match_tol, shape, args.atom_len)
        p_, r_ = val_scores["precision"], val_scores["recall"]
        f1 = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0.0
        calibration[atom] = (f1, fit[0], fit[1])
    for atom in range(args.n_atoms):
        f1, offset, sign = calibration.get(atom, (float("nan"), 0, 0))
        _kv(f"atom {atom:2d}: |xcorr| / val F1 / offset / sign",
            f"{xcorr[atom]:.3f} / {f1:.3f} / {offset:+d} / {sign:+d}")
    _kv("event reconstruction |corr| (test, all atoms)",
        f"{_event_reconstruction_corr(torch, np, sae, x_test, events_test, args.atom_len):.3f}")
    if calibration:
        chosen, (val_f1, offset, sign) = max(calibration.items(), key=lambda kv: (kv[1][0], xcorr[kv[0]]))
        _kv("atom / offset / sign (val)", f"{chosen} / {offset} / {sign:+d}  (val F1 {val_f1:.3f})")
        acts = acts_test[acts_test[:, 2] == chosen][:, [0, 1, 3, 4]]
        scores = _event_metrics(np, acts, events_test, offset, sign, args.match_tol, shape, args.atom_len)
        for key in ("recall_isolated", "recall_close", "n_close", "recall_by_template", "timing_error_samples",
                    "count_mae_per_window", "duplicates", "false_alarms_per_channel_minute", "polarity_accuracy",
                    "n_true", "n_pred"):
            _kv(key, f"{scores[key]:.3f}" if isinstance(scores[key], float) else scores[key])
        checks.soft(f"event precision (> {RECOVERY_MIN_PR})", scores["precision"] > RECOVERY_MIN_PR,
                    f"{scores['precision']:.3f}")
        checks.soft(f"event recall (> {RECOVERY_MIN_PR})", scores["recall"] > RECOVERY_MIN_PR,
                    f"{scores['recall']:.3f}")
    else:
        checks.soft("event matching calibrated", False, "no atom activated near a validation event")

    _sec("-> flows to Stage 5")
    print("  # Counts are sparse non-negative, like HYDRA counts; the _SparseScaler")
    print("  # standardises them (and the peak / cosine blocks) before the classifier.")

    _sec("RESULT")
    _kv("deterministic failures", checks.failed or "none")
    _kv("gate below tolerance", checks.soft_failed or "none")
    if checks.failed or (args.strict and checks.soft_failed):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
