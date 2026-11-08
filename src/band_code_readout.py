"""Frozen readout of the band-limited code: do hard and soft counts of the learned waveforms classify epilepsy
patients better than band power on the same rows, and do they add to the maintenance drug?

Protocol: ``documentation/band_code_readout_protocol.md``. Run on the HPC from ``code/``; a root holds one readout and
records the sweep root and the readout root that it reads (``readout_roots.json``). ``src/band_code_readout_core.py``
holds the features and the rule. Subcommands, in order:

- ``fit-untrained`` keeps the random start of the adopted dictionary of one fold (one epoch at learning rate 0).
- ``extract-band`` writes the band-power and level features of every patient's fixed segments (no dictionary).
- ``extract`` writes the code features of every patient under one fold's dictionary, with the data of gates 3 to 5.
- ``gates`` checks gates 2 to 6 of section 6 (the launcher runs the tests, gate 1) and writes ``gates.json``.
- ``heads`` fits the head of every arm, only after the gates pass.
- ``read`` applies the rule of section 7 and the reads of section 8 (``band_code_decision.json``).

The 39 test patients stay locked: the readout's cohort refuses them. No label enters a dictionary or a feature.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import rootutils
import torch
from loguru import logger
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.band_code_readout_core import (  # noqa: E402
    BANDS,
    EV_TOLERANCE,
    N_PERMUTATIONS,
    N_REPEAT,
    PERMUTATION_MARGIN,
    REPEAT_TOLERANCE,
    TOP_ATOMS,
    channel_index,
    code_features,
    count_check,
    encode_details,
    level_features,
    readout_outcome,
    recur,
    segment_energy,
)
from src.band_reconstruction import (  # noqa: E402
    CHECKPOINT,
    DOMAIN,
    EXTRACT_BATCH,
    _fixed_view,
    _view,
    check_domain,
    require_deterministic_cublas,
)
from src.band_sweep import _build, _load, _seeded, score_file  # noqa: E402
from src.band_sweep_core import Setting, check_setting, separable_spec  # noqa: E402
from src.covariate_audit_core import (  # noqa: E402
    MISSING_SEX,
    group_read,
    matched_read,
    pair_mask,
    standardize_within_fold,
)
from src.encoder_check_core import alone_interval, drug_gain  # noqa: E402
from src.frozen_readout_core import (  # noqa: E402
    N_FOLDS,
    ArmScores,
    auroc,
    bag_size_link,
    dictionary_split,
    fit_head,
    paired_interval,
    permuted_scores,
    pooled_auroc,
)
from src.reconstruction_baselines import _git, _save_json, _save_npz, _text_array  # noqa: E402
from src.separable_factor_analysis import dominant_frequency  # noqa: E402

ADOPTED = Setting(K=64, L=128, D=8, P=512, R=4)  # the adopted configuration of the sweep (protocol section 3)
DICTIONARIES = {"trained": ADOPTED, "untrained": ADOPTED, "seed2": ADOPTED.with_seed(2)}
# The arms of section 4: (feature file, feature sets). "band" is the file of extract-band.
ARMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "Code": ("trained", ("code",)),
    "Code-untrained": ("untrained", ("code",)),
    "Band-scaled": ("band", ("band_scaled",)),
    "Code-templates": ("trained", ("templates",)),
    "Code-hard": ("trained", ("hard",)),
    "Code-soft": ("trained", ("soft",)),
    "Code-tails": ("trained", ("tails",)),
    "Code-channels": ("trained", ("channels",)),
    "Code-asymmetry": ("trained", ("asymmetry",)),
    "Code+level": ("trained", ("code", "level")),
    "Band-raw": ("band", ("band_raw",)),
    "Code-s2": ("seed2", ("code",)),
    "Code-hydra": ("trained", ("counts",)),
}
SPARSE_ARMS = ("Code-hydra",)  # the HYDRA sparse scaler; every other arm uses StandardScaler (section 10)
SAVED = ("Spectral", "HYDRA", "S4-R1")  # the saved pass-1 scores (descriptive)
MATCHED = ("Code", "Band-scaled", "Code-s2")
PREDICTION = "no_advantage about 60%, beats_band_power 20%, review 10%, code_adds 10% (protocol section 7)"


# ---------------------------------------------------------------------- scalers


class HydraRows:
    """The HYDRA sparse scaler (``hydra_transformer._SparseScaler``) on numpy rows, as pass 1's
    ``frozen_readout.SparseRows`` wraps it: a square root of the values clipped at 0, then a standardization that keeps
    zeros at 0 and adds ``q**4`` to the standard deviation of a feature that is 0 in a share ``q`` of the rows.

    It wraps the class directly, so that this driver does not import the data engine of ``frozen_readout``.
    """

    def fit(self, x: np.ndarray) -> HydraRows:
        """Fit the scaler on the training rows."""
        from src.models.components.hydra_transformer import _SparseScaler

        self.inner = _SparseScaler()
        self.inner.fit(torch.from_numpy(np.asarray(x, dtype=np.float64)))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Scale the rows."""
        return self.inner.transform(torch.from_numpy(np.asarray(x, dtype=np.float64))).numpy()


def scaler_for(arm: str) -> type:
    """The scaler of an arm: ``HydraRows`` for ``SPARSE_ARMS``, ``StandardScaler`` for every other arm."""
    return HydraRows if arm in SPARSE_ARMS else StandardScaler


# ---------------------------------------------------------------------- roots and files


def check_root(root: Path, sweep_root: Path, readout_root: Path) -> None:
    """Record the band domain and the roots this readout reads, or refuse a root of another kind or other roots.

    Raises
    ------
    ValueError
        If the root holds a read of another protocol, or records other roots.
    """
    for name in ("band_decision.json", "sweep.json", "selection.json"):
        if (root / name).is_file():
            raise ValueError(f"{root} holds {name}; the readout needs a root of its own")
    check_domain(root)
    record = {"sweep_root": str(sweep_root.resolve()), "readout_root": str(readout_root.resolve())}
    path = root / "readout_roots.json"
    if path.is_file():
        if json.loads(path.read_text()) != record:
            raise ValueError(f"{root} reads other roots: {path.read_text()}")
        return
    _save_json(path, record)


def feature_file(root: Path, kind: str, fold: int | None = None) -> Path:
    """The features of one dictionary kind and fold, or of the band arms (``kind="band"``)."""
    return root / "features" / ("band.npz" if kind == "band" else f"{kind}_f{fold}.npz")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: run its stage first")
    with np.load(path) as stored:
        return {key: stored[key] for key in stored.files}


def dictionary_folder(root: Path, sweep_root: Path, kind: str, fold: int) -> Path:
    """The dictionary of one kind and fold: the untrained start in this root, the others in the sweep root."""
    if kind == "untrained":
        return root / "dictionaries" / f"untrained_f{fold}"
    return sweep_root / "dictionaries" / f"{DICTIONARIES[kind].name}_f{fold}"


# ---------------------------------------------------------------------- stages


def fit_untrained(args: argparse.Namespace) -> None:
    """The random start of the adopted dictionary of one fold: one epoch at learning rate 0, as in the earlier
    protocols, with the seed and data of the trained dictionary, so it is that dictionary's start."""
    import hydra
    import lightning

    from src.frozen_readout import every_row
    from src.models.components.separable_sae import SeparableSAE
    from src.utils import split_provenance

    root, fold = args.root.resolve(), args.fold
    check_root(root, args.sweep_root, args.readout_root)
    output = dictionary_folder(root, args.sweep_root, "untrained", fold)
    if (output / CHECKPOINT).is_file():
        logger.info(f"{output.name}: finished")
        return
    require_deterministic_cublas()
    context = _seeded(args, ADOPTED, fold)
    plan, cohort, seed = context["plan"], context["cohort"], context["seed"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    train_patients, val_patients = dictionary_split(plan["labels"], plan["folds"], fold, plan["seed"])
    views = {"train": _view(context, every_row(cohort, train_patients)), "val": _fixed_view(context, val_patients)}
    provenance = split_provenance(context["datamodule"], context["cfg"].data)
    provenance["train_subjects"] = train_patients
    provenance["band_code_readout"] = {"fold": fold, "arm": "untrained", "setting": ADOPTED.name, **DOMAIN}
    base = context["datamodule"].train_dataloader()
    loaders = [
        DataLoader(views[split], batch_size=base.batch_size, num_workers=base.num_workers, pin_memory=base.pin_memory)
        for split in ("train", "val")
    ]
    spec = separable_spec(ADOPTED, hydra.utils.instantiate(context["cfg"].feature.spec))
    train_spec = replace(hydra.utils.instantiate(context["cfg"].feature.train_spec), epochs=1, lr=0.0)
    lightning.seed_everything(seed, workers=True)
    model = SeparableSAE(spec=spec, train_spec=train_spec, random_state=seed, device="cuda")
    check_setting(model, ADOPTED)
    output.mkdir(parents=True, exist_ok=True)
    _save_json(
        output / "fit_provenance.json",
        {
            "fold": fold,
            "arm": "untrained",
            "setting": ADOPTED.name,
            "seed": seed,
            "train_patients": len(train_patients),
            "val_patients": len(val_patients),
            "commit": _git("rev-parse", "HEAD"),
            "uncommitted": _git("status", "--porcelain").splitlines(),
            **DOMAIN,
        },
    )
    model.fit_unsupervised(*loaders, provenance=provenance)
    model.save_artifacts(output)


def extract_band(args: argparse.Namespace) -> None:
    """Band power on the scaled rows and on the band-passed segments, and the level, for every patient's segments."""
    from src.models.components.spectral_features import SpectralFeatures

    root = args.root.resolve()
    check_root(root, args.sweep_root, args.readout_root)
    path = feature_file(root, "band")
    if path.is_file():
        logger.info("band features: finished")
        return
    context = _seeded(args, ADOPTED, 0)
    reader = _build(context, ADOPTED)  # rows only; the scaling does not depend on the atoms
    view = _fixed_view(context, sorted(context["plan"]["labels"]))
    spectral = SpectralFeatures(sfreq=context["sfreq"], bands=BANDS, nperseg=512, device="cuda")
    parts: dict[str, list[np.ndarray]] = {"band_scaled": [], "band_raw": [], "level": []}
    with torch.no_grad():
        for x, _ in DataLoader(view, batch_size=EXTRACT_BATCH, shuffle=False, num_workers=args.num_workers):
            rows, _ = reader._full_rows(x)
            parts["band_scaled"].append(spectral(rows).double().cpu().numpy())
            parts["band_raw"].append(spectral(x.to(rows.device)).double().cpu().numpy())
            parts["level"].append(level_features(x))
    subjects = _text_array(view.plan["subject"])
    _save_npz(
        path,
        subject=subjects,
        label=np.array([context["plan"]["labels"][s] for s in subjects]),
        **{name: np.concatenate(chunks) for name, chunks in parts.items()},
    )
    logger.info(f"band features: {len(subjects)} segments")


def extract(args: argparse.Namespace) -> None:
    """The code features of every patient's segments under the dictionary of one fold and kind, with the data of
    gates 3 (explained energy against the sweep), 4 (counts), 5 (repeat run) and 6 (loaded segments)."""
    root, fold, kind = args.root.resolve(), args.fold, args.kind
    sweep_root = args.sweep_root.resolve()
    check_root(root, sweep_root, args.readout_root)
    path = feature_file(root, kind, fold)
    if path.is_file():
        logger.info(f"{path.name}: finished")
        return
    setting = DICTIONARIES[kind]
    folder = dictionary_folder(root, sweep_root, kind, fold)
    if not (folder / CHECKPOINT).is_file():
        raise FileNotFoundError(f"{folder} holds no dictionary: fit it first (fit-untrained for the untrained start)")
    context = _seeded(args, setting, fold)
    model = _load(context, setting, folder)
    plan = context["plan"]
    view = _fixed_view(context, sorted(plan["labels"]))
    subjects = _text_array(view.plan["subject"])
    left, right = channel_index(list(view.target_channels))
    waveforms, patterns = (t.detach() for t in model._split(model._dictionary().detach()))

    def features_of(x: torch.Tensor) -> dict[str, np.ndarray]:
        rows, valid = model._full_rows(x)
        details = encode_details(model, rows, valid)
        minutes = x.shape[-1] / context["sfreq"] / 60.0
        out = code_features(details, patterns, model.spec.pool_len, model.spec.top_r, minutes, left, right)
        residual, signal = segment_energy(rows, details.reconstruction, valid)
        return out | {"residual": residual, "signal": signal}

    parts: dict[str, list[np.ndarray]] = {}
    first: list[torch.Tensor] = []  # the first batches, at least N_REPEAT segments, for gate 5
    failures = 0
    for x, _ in DataLoader(view, batch_size=EXTRACT_BATCH, shuffle=False, num_workers=args.num_workers):
        failures += int((x.flatten(1).abs().amax(dim=1) == 0).sum())
        if sum(len(t) for t in first) < N_REPEAT:
            first.append(x)
        for name, value in features_of(x).items():
            parts.setdefault(name, []).append(value)
    arrays = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    repeat = torch.cat(first)  # the same batches as the pass, so the repeat sees the same batch shapes
    again = {name: np.concatenate(chunks) for name, chunks in _batched(features_of, repeat).items()}
    repeat_gap = max(float(np.abs(again[s] - arrays[s][: len(repeat)]).max()) for s in ("code", "templates"))
    ev_gap = np.nan
    if kind != "untrained":
        held = np.array([plan["folds"][s] == fold for s in subjects])
        stored = _load_npz(score_file(sweep_root, setting, fold))
        if stored["subject"].astype(str).tolist() != subjects[held].tolist():
            raise ValueError(f"the held-out segments of fold {fold} differ from the sweep's score file")
        ev_gap = max(
            float((np.abs(arrays[k][held] - stored[f"{k}_total"]) / np.maximum(stored[f"{k}_total"], 1e-12)).max())
            for k in ("residual", "signal")
        )
    _save_npz(
        path,
        subject=subjects,
        label=np.array([plan["labels"][s] for s in subjects]),
        fold=np.array(fold),
        kind=np.array(kind),
        setting=np.array(setting.name),
        waveforms=waveforms.cpu().numpy(),
        patterns=patterns.cpu().numpy(),
        dominant_hz=dominant_frequency(waveforms.cpu().numpy(), context["sfreq"]),
        repeat_gap=np.array(repeat_gap),
        repeat_segments=np.array(len(repeat)),
        ev_gap=np.array(ev_gap),
        load_failures=np.array(failures),
        **arrays,
    )
    logger.info(f"{path.name}: {len(subjects)} segments, EV gap {ev_gap:.2e}, repeat gap {repeat_gap:.2e}")


def _batched(function: Any, x: torch.Tensor) -> dict[str, list[np.ndarray]]:
    """``function`` on ``x`` in batches of ``EXTRACT_BATCH``, as the feature pass runs it, with the outputs listed."""
    parts: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(x), EXTRACT_BATCH):
        for name, value in function(x[start : start + EXTRACT_BATCH]).items():
            parts.setdefault(name, []).append(value)
    return parts


def _arm_rows(root: Path, arm: str, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subjects, labels and the feature rows of one arm for the head of one fold."""
    kind, sets = ARMS[arm]
    data = _load_npz(feature_file(root, kind, fold))
    blocks = []
    for name in sets:
        if name == "level":
            band = _load_npz(feature_file(root, "band"))
            if band["subject"].astype(str).tolist() != data["subject"].astype(str).tolist():
                raise ValueError("the band and code features hold other segments")
            blocks.append(band["level"])
        else:
            blocks.append(data[name])
    return data["subject"].astype(str), data["label"], np.concatenate(blocks, axis=1).astype(np.float64)


def gates(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Gates 2 to 6 of section 6 on the feature files, with the permutation null at ``C=1``."""
    files = {(k, f): _load_npz(feature_file(root, k, f)) for k in DICTIONARIES for f in range(N_FOLDS)}
    band = _load_npz(feature_file(root, "band"))
    expected = sorted(plan["labels"])
    order = band["subject"].astype(str)
    complete = (
        sorted(set(order.tolist())) == expected
        and all(int((order == s).sum()) == plan["n_segments"] for s in expected)
        and all(d["subject"].astype(str).tolist() == order.tolist() for d in files.values())
    )
    checks = [count_check(d["won_total"], d["positive_total"], int(d["slots"][0])) for d in files.values()]
    ev_gap = max(float(d["ev_gap"]) for (k, _), d in files.items() if k != "untrained")
    repeat_gap = max(float(d["repeat_gap"]) for d in files.values())
    repeat_segments = min(int(d["repeat_segments"]) for d in files.values())
    failures = max(int(d["load_failures"]) for d in files.values())  # every file reads the same segments
    code_rows = {fold: _arm_rows(root, "Code", fold) for fold in range(N_FOLDS)}
    rng = np.random.default_rng(plan["seed"])
    draws = []
    for _ in range(N_PERMUTATIONS):
        values = []
        for fold, (subject, label, x) in code_rows.items():
            held = np.array([plan["folds"][s] == fold for s in subject])
            patients, pooled = permuted_scores(x, label, subject, held, 1.0, StandardScaler, rng)
            values.append(auroc(np.array([plan["labels"][s] for s in patients]), pooled))
        draws.append(float(np.mean(values)))
    permuted_mean = float(np.mean(draws))
    passed = {
        "permutation": abs(permuted_mean - 0.5) <= PERMUTATION_MARGIN,
        "explained_energy_matches_sweep": bool(ev_gap <= EV_TOLERANCE),
        "counts_add_up": all(c["matches_code"] and c["within_slots"] for c in checks),
        "repeat_run": repeat_gap <= REPEAT_TOLERANCE and repeat_segments >= N_REPEAT,
        "complete": complete and failures == 0,
    }
    record = {
        "status": "pass" if all(passed.values()) else "blocked",
        "gates": passed,
        "permuted_mean": permuted_mean,
        "permuted_draws": draws,
        "ev_gap": ev_gap,
        "repeat_gap": repeat_gap,
        "repeat_segments": repeat_segments,
        "load_failures": failures,
        "zero_slot_share": float(np.mean([c["zero_slot_share"] for c in checks])),
    }
    _save_json(root / "gates.json", record)
    logger.info(f"gates: {record['status']} {passed}")
    return record


def heads(root: Path, plan: dict[str, Any]) -> None:
    """The head of every arm on the five folds (section 5); refused until the gates pass. A finished arm is skipped.

    Raises
    ------
    ValueError
        If the gates have not passed.
    """
    path = root / "gates.json"
    if not path.is_file() or json.loads(path.read_text())["status"] != "pass":
        raise ValueError("the gates have not passed: run the gates stage, and fix what it reports")
    (root / "scores").mkdir(parents=True, exist_ok=True)
    (root / "coef").mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        out = root / "scores" / f"{arm}.csv"
        if out.is_file():
            continue
        rows = []
        for fold in range(N_FOLDS):
            subject, label, x = _arm_rows(root, arm, fold)
            held = np.array([plan["folds"][s] == fold for s in subject])
            result = fit_head(x, label, subject, held, scaler_for(arm), seed=plan["seed"] + fold)
            rows += [
                {"arm": arm, "subject": s, "fold": fold, "label": int(y), "score": float(v), "c": result.c}
                for s, y, v in zip(result.subject, result.label, result.score)
            ]
            if arm == "Code":
                np.save(root / "coef" / f"Code_f{fold}.npy", result.coef)
            logger.info(f"head {arm} fold {fold}: C {result.c}, AUROC {auroc(result.label, result.score):.3f}")
        pl.DataFrame(rows).write_csv(out)


# ---------------------------------------------------------------------- read


def _scores(path: Path) -> ArmScores:
    frame = pl.read_csv(path, schema_overrides={"subject": pl.Utf8})
    columns = (frame[name].to_numpy() for name in ("fold", "label", "score"))
    return ArmScores.from_rows(frame["subject"].to_numpy().astype(str), *columns)


def load_cohort(readout_root: Path, audit_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """The patients in sorted order, with fold, label, drug status, age, sex and segments per patient."""
    overrides = {"subject": pl.Utf8}
    patients = pl.read_csv(readout_root / "patients.csv", schema_overrides=overrides).sort("subject")
    subject = np.array(patients["subject"].to_list())
    exposure = pl.read_csv(audit_root / "drug_exposure.csv", schema_overrides=overrides)
    status = dict(zip(exposure["subject"].to_list(), exposure["status"].to_list()))
    covariates = pl.read_csv(audit_root / "covariates.csv", schema_overrides={"subject": pl.Utf8, "sex": pl.Utf8})
    age = dict(zip(covariates["subject"].to_list(), covariates["age"].cast(pl.Float64).to_list()))
    sex = dict(zip(covariates["subject"].to_list(), covariates["sex"].to_list()))
    return {
        "subject": subject,
        "fold": np.array([plan["folds"][s] for s in subject]),
        "label": np.array([plan["labels"][s] for s in subject]),
        "status": np.array([status.get(s, "unknown") for s in subject]),
        "age": np.array([np.nan if age.get(s) is None else float(age[s]) for s in subject]),
        "sex": np.array([sex.get(s) or MISSING_SEX for s in subject]),
        "segments": dict(zip(subject.tolist(), patients["segments"].to_list())),
    }


def _aligned(arm: ArmScores, cohort: dict[str, Any]) -> ArmScores:
    if arm.subject.tolist() != cohort["subject"].tolist() or arm.fold.tolist() != cohort["fold"].tolist():
        raise ValueError("an arm does not hold the patients of the plan in their folds")
    return arm


def _arm_summary(name: str, arm: ArmScores, cohort: dict[str, Any], n_boot: int) -> dict[str, Any]:
    value, low, high = alone_interval(arm, n_boot)
    gain = drug_gain(arm.score, cohort["label"], cohort["fold"], cohort["status"], n_boot)
    z = standardize_within_fold(arm.score, cohort["fold"])
    off = cohort["status"] == "off"
    cell = group_read(z, (cohort["label"] == 1) & off, (cohort["label"] == 0) & off, cohort["fold"], n_boot)
    links = bag_size_link(arm, cohort["segments"])
    return {
        "arm": name,
        "auroc": value,
        "low": low,
        "high": high,
        "pooled_auroc": pooled_auroc(arm),
        "gain": gain[0],
        "gain_low": gain[1],
        "gain_high": gain[2],
        "off_off": cell.value,
        "off_off_low": cell.low,
        "off_off_high": cell.high,
        "off_off_epilepsy": cell.first,
        "off_off_non_epilepsy": cell.second,
        "bag_link_epilepsy": links["epilepsy"],
        "bag_link_non_epilepsy": links["non_epilepsy"],
    }


def _top_atoms(root: Path) -> dict[str, Any]:
    """The atoms with the largest weights of the ``Code`` head per fold, with their dominant frequency, and how many
    recur in every other fold's top atoms (section 8)."""
    tops: dict[int, list[tuple[int, float, tuple[np.ndarray, np.ndarray]]]] = {}
    for fold in range(N_FOLDS):
        coef_path = root / "coef" / f"Code_f{fold}.npy"
        if not coef_path.is_file():
            return {}
        coef = np.load(coef_path)
        k = len(coef) // 2
        weight = np.maximum(np.abs(coef[:k]), np.abs(coef[k:]))
        data = _load_npz(feature_file(root, "trained", fold))
        best = np.argsort(-weight)[:TOP_ATOMS]
        tops[fold] = [
            (int(a), float(data["dominant_hz"][a]), (data["waveforms"][a], data["patterns"][a])) for a in best
        ]
    recurring = {
        fold: sum(all(any(recur(atom[2], other[2]) for other in tops[o]) for o in tops if o != fold) for atom in atoms)
        for fold, atoms in tops.items()
    }
    return {
        "per_fold": {f: [{"atom": a, "dominant_hz": hz} for a, hz, _ in atoms] for f, atoms in tops.items()},
        "recurring_in_every_fold": recurring,
    }


def read(root: Path, readout_root: Path, audit_root: Path, n_boot: int = 10_000) -> dict[str, Any]:
    """The rule of section 7 and the reads of section 8; writes ``band_code_decision.json`` and its table."""
    plan = json.loads((readout_root / "folds.json").read_text())
    gate_record = json.loads((root / "gates.json").read_text())
    cohort = load_cohort(readout_root, audit_root, plan)
    arms = {name: _aligned(_scores(root / "scores" / f"{name}.csv"), cohort) for name in ARMS}
    saved = {name: _aligned(_scores(readout_root / "scores" / f"{name}.csv"), cohort) for name in SAVED}
    learning = paired_interval(arms["Code"], arms["Code-untrained"], n_boot=n_boot)
    band = paired_interval(arms["Code"], arms["Band-scaled"], n_boot=n_boot)
    drug = drug_gain(arms["Code"].score, cohort["label"], cohort["fold"], cohort["status"], n_boot)
    met = {"L": learning[1] > 0, "B": band[1] > 0, "G": drug[1] > 0}
    outcome = readout_outcome(gate_record["status"] == "pass", met["L"], met["B"], met["G"])
    summary = [_arm_summary(n, a, cohort, n_boot) for n, a in {**arms, **saved}.items()]
    pl.DataFrame(summary).write_csv(root / "band_code_summary.csv")
    mask = pair_mask(cohort["label"], cohort["age"], cohort["sex"])
    matched = {}
    for name in MATCHED:
        m = matched_read(arms[name].score, cohort["label"], cohort["fold"], mask, n_boot=n_boot)
        matched[name] = {"value": m.value, "low": m.low, "high": m.high, "pairs": m.pairs}
    replicate = paired_interval(arms["Code-s2"], arms["Code"], n_boot=n_boot)
    scaler_check = paired_interval(arms["Code-hydra"], arms["Code"], n_boot=n_boot)
    record = {
        "protocol": "documentation/band_code_readout_protocol.md, section 7",
        "outcome": outcome,
        "prediction": PREDICTION,
        "gates": gate_record,
        "conditions": met,
        "deciding": {
            "L_Code_minus_Code_untrained": dict(zip(("value", "low", "high"), learning)),
            "B_Code_minus_Band_scaled": dict(zip(("value", "low", "high"), band)),
            "G_gain_over_drug": dict(zip(("value", "low", "high"), drug)),
        },
        "patients": len(cohort["subject"]),
        "known_drug_status": int((cohort["status"] != "unknown").sum()),
        "descriptive": {
            "arms": summary,
            "matched": matched,
            "seed_replicate_Code_s2_minus_Code": dict(zip(("value", "low", "high"), replicate)),
            "seed_rank_correlation": float(spearmanr(arms["Code"].score, arms["Code-s2"].score).statistic),
            "scaler_check_Code_hydra_minus_Code": dict(zip(("value", "low", "high"), scaler_check)),
            "top_atoms": _top_atoms(root),
        },
    }
    _save_json(root / "band_code_decision.json", record)
    logger.info(f"band code readout: {outcome} (conditions {met})")
    return record


# ---------------------------------------------------------------------- entry point


def main() -> None:
    """Command-line entry point for the subcommands of the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("fit-untrained", "the random start of one fold's adopted dictionary"),
        ("extract-band", "the band-power and level features"),
        ("extract", "the code features of one fold's dictionary"),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--sweep-root", type=Path, required=True, help="the sweep root with the dictionaries")
        command.add_argument("--readout-root", type=Path, required=True, help="the root of frozen readout pass 1")
        command.add_argument("--manifests", type=Path, required=True, help="the 40 s offset audit")
        command.add_argument("--data-dir", type=Path, required=True, help="the corpus root")
        command.add_argument("--signal-cache-dir", type=Path, required=True, help="the campaign signal cache")
        command.add_argument("--num-workers", type=int, default=4)
        if name != "extract-band":
            command.add_argument("--fold", type=int, required=True, choices=range(N_FOLDS))
        if name == "extract":
            command.add_argument("--kind", choices=tuple(DICTIONARIES), required=True)
    for name, description in (("gates", "gates 2 to 6"), ("heads", "the heads of every arm"), ("read", "the rule")):
        command = commands.add_parser(name, help=description)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--readout-root", type=Path, required=True)
        if name == "read":
            command.add_argument("--audit-root", type=Path, required=True, help="the covariate audit root")
    args = parser.parse_args()
    if args.command == "fit-untrained":
        fit_untrained(args)
    elif args.command == "extract-band":
        extract_band(args)
    elif args.command == "extract":
        extract(args)
    else:
        root, readout = args.root.resolve(), args.readout_root.resolve()
        plan = json.loads((readout / "folds.json").read_text())
        if args.command == "gates":
            gates(root, plan)
        elif args.command == "heads":
            heads(root, plan)
        else:
            read(root, readout, args.audit_root.resolve())


if __name__ == "__main__":
    main()
