"""Stage 6 - Classifier fit + window/subject scoring (mini end-to-end).

INPUT  : scaled HYDRA features Fs (batch, n_features), integer labels y, and a
         metadata DataFrame carrying `subject` + `epilepsy`. Built here from
         synthetic windows pushed through Stages 4-5 so the whole handoff chain
         is visible.
OUTPUT : a fitted `make_pipeline(scaler, classifier)`, its decision_function
         scores, window-level accuracy, and the subject-level accuracy obtained
         by averaging signed scores per subject - exactly what
         `Trainer._get_scores` reports and what `eval.py` reloads from joblib.

The metrics are illustrative only (synthetic data), never a real result. No
corpus needed.

Run:
    uv run python tests/stage6_classifier_scoring.py --model logistic --n-subjects 8
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


def _desc(name: str, a) -> None:
    import numpy as np

    a = np.asarray(a)
    _sec(f"[array] {name}")
    _kv("shape", a.shape)
    _kv("dtype", a.dtype)
    if a.size:
        af = a.astype("float64")
        _kv("min/max", f"{af.min():.4g} / {af.max():.4g}")
        _kv("mean/std", f"{af.mean():.4g} / {af.std():.4g}")
        _kv("first 6", [round(float(v), 4) for v in af.flatten()[:6]])


def _need(import_fn, pkgs: str):
    try:
        return import_fn()
    except ImportError as exc:
        _sec("missing dependency")
        _kv("error", exc)
        print(f"  # needs: {pkgs}; run `uv sync` in code/ "
              "(add aeon/scikit-learn/pandas if not declared)")
        return None


def _synth(n_windows, n_channels, n_timepoints, n_subjects, seed, signal):
    """Synthetic windowed batch with one class per subject (see Stage 2 output)."""
    import numpy as np
    import pandas as pd
    import torch

    rng = np.random.RandomState(seed)
    subjects = [f"sub-{i:03d}" for i in range(n_subjects)]
    subj_class = {s: i % 2 for i, s in enumerate(subjects)}
    win_subjects = [subjects[i % n_subjects] for i in range(n_windows)]
    y = np.array([subj_class[s] for s in win_subjects], dtype=np.int64)
    x = rng.randn(n_windows, n_channels, n_timepoints).astype("float32")
    t = np.linspace(0.0, 1.0, n_timepoints, dtype="float32")
    for i in range(n_windows):
        freq = 5.0 if y[i] == 1 else 12.0
        x[i, :3, :] += signal * np.sin(2.0 * np.pi * freq * t)[None, :]
    meta = pd.DataFrame({"subject": win_subjects,
                         "epilepsy": [bool(subj_class[s]) for s in win_subjects]})
    return torch.from_numpy(x), torch.from_numpy(y), meta


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=["logistic", "ridge"], default="logistic")
    p.add_argument("--n-windows", type=int, default=96)
    p.add_argument("--n-subjects", type=int, default=8)
    p.add_argument("--channels", type=int, default=19)
    p.add_argument("--timepoints", type=int, default=512)
    p.add_argument("--n-groups", type=int, default=32)
    p.add_argument("--n-kernels", type=int, default=8)
    p.add_argument("--signal", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    _banner("STAGE 6 - Classifier fit + window/subject scoring")

    def _imports():
        import numpy as np
        import pandas as pd
        import torch
        from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
        from sklearn.pipeline import make_pipeline

        from src.models.components.hydra_transformer import HydraTransformer, _SparseScaler
        return (np, pd, torch, LogisticRegressionCV, RidgeClassifierCV,
                make_pipeline, HydraTransformer, _SparseScaler)

    mods = _need(_imports, "torch, numpy, pandas, scikit-learn, aeon")
    if mods is None:
        return 1
    (np, pd, _torch, LogisticRegressionCV, RidgeClassifierCV,
     make_pipeline, HydraTransformer, _SparseScaler) = mods

    x, y, meta = _synth(args.n_windows, args.channels, args.timepoints,
                        args.n_subjects, args.seed, args.signal)
    fe = HydraTransformer(n_kernels=args.n_kernels, n_groups=args.n_groups,
                          random_state=args.seed)
    f = fe(x)

    _sec("INPUT")
    _kv("classifier", args.model)
    _kv("F (features) shape", tuple(f.shape))
    _kv("y shape / classes", f"{tuple(y.shape)}  {sorted(set(y.tolist()))}")
    _kv("n_subjects", args.n_subjects)
    _kv("meta columns", list(meta.columns))

    scaler = _SparseScaler(mask=True, exponent=4)
    if args.model == "logistic":
        clf = LogisticRegressionCV(cv=5, Cs=np.logspace(-3, 3, 10),
                                   n_jobs=-1, max_iter=500)
    else:
        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=5)
    pipe = make_pipeline(scaler, clf)
    pipe.fit(f, y)

    _banner("OUTPUT")
    _sec("fitted pipeline")
    _kv("steps", [type(pipe[i]).__name__ for i in range(len(pipe))])
    _kv("pipe[0] (scaler)", type(pipe[0]).__name__)
    _kv("pipe[1] (classifier)", type(pipe[1]).__name__)
    print("  # eval.py recovers exactly these two via scaler, model = pipe[0], pipe[1]")

    decision = pipe.decision_function(f)
    _desc("decision_function(F)", decision)

    # Window-level accuracy (what pipeline.score reports).
    window_acc = pipe.score(f, y)

    # Subject-level accuracy: average signed score per subject, threshold at 0.
    df = meta[["subject", "epilepsy"]].copy().set_index("subject")
    df["score"] = decision
    mean_scores = df.groupby(df.index)["score"].mean()
    epi = df.groupby(df.index)["epilepsy"].first()
    subject_acc = ((mean_scores > 0).astype(bool) == epi.astype(bool)).mean()

    _sec("scores (mirror of Trainer._get_scores)")
    _kv("accuracy_by_window", f"{window_acc:.4f}")
    _kv("accuracy_by_subject", f"{subject_acc:.4f}")
    print("  # synthetic data -> illustrative only, not a real performance number")

    _sec("per-subject mean decision score")
    print((pd.DataFrame({"mean_score": mean_scores, "epilepsy": epi})
           .round(4).to_string()))

    _sec("end of pipeline")
    print("  # Trainer.fit joblib-dumps this pipeline (model.joblib), the feature")
    print("  # extractor, and the scaler; eval.py reloads them to score the test split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
