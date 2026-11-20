"""Events corpus protocol, stage 1: the pilot. No dictionary is trained.

Protocol: ``documentation/event_corpus_dictionary_protocol.md``, sections 5 to 10. On the root of the inventory
(``src/event_dictionary.py``), with its ``files.csv`` and ``folds.json``. Subcommands, in order:

- ``cache`` (HPC, CPU): every training recording through the signal chain of the campaign, as ``(20, samples)``
  float32 at 256 Hz under ``cache/train/<patient>/<file>.npy``. The evaluation set is not read.
- ``reader-check`` (HPC; gate 3): the reader of this corpus against the campaign's rows, on recordings of the
  epilepsy corpus.
- ``features`` (GPU): for every kept channel-second, the response of every waveform of the adopted dictionaries of
  the epilepsy corpus (``Transfer``) and of random dictionaries (``Untrained``), and the simple measures, in the
  three bands: as the rank inside the record (``rank.npy``), as the plain value (``plain.npy``), and as the rank at
  a displaced second of the same record (``null.npy``). It also runs the time-alignment check (gate 4).
- ``read``: the choice of the feature on the training folds and its score on the held-out fold, for each class,
  family and read; the gains of ``Transfer`` over ``Untrained`` and over the simple measures; and the gates
  (``pilot.json``, ``pilot_summary.csv``). The pilot is descriptive: no dictionary of this corpus exists yet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import rootutils
import torch
from loguru import logger

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.event_dictionary_core import CLASSES, MONTAGE, N_FOLDS, parse_rec, pieces, units  # noqa: E402
from src.event_response_core import (  # noqa: E402
    BANDS,
    DECIDING,
    LEVEL,
    N_RANDOM,
    SFREQ,
    auroc_columns,
    band_rows,
    choose_and_score,
    class_outcome,
    interval,
    patient_draws,
    random_atoms,
    rank_in_record,
    record_features,
    responses,
    simple_names,
    summarize,
)

PROTOCOL = "documentation/event_corpus_dictionary_protocol.md"
ADOPTED = "K64_L128_D8_P512_R4_E30_s1"  # the adopted dictionaries of the epilepsy corpus, one per fold of that corpus
READS = ("rank", "plain", "null")
FAMILIES = ("transfer", "untrained", "simple")
EVENT_CLASSES = tuple(label for label in CLASSES.values() if label != "bckg")
READER_TOLERANCE = 1e-5  # gate 3
PLANT_AMPLITUDE, ALIGN_PASS, ALIGN_SHIFTED, ALIGN_RECORDS = 6.0, 0.95, 0.60, 200  # gate 4
# Gate 5: at displaced seconds, 0.5 lies in the interval of every deciding class and family, at a level of 1 minus
# 0.01 divided by the number of intervals; and the mean rank of all displaced units is near 0.5.
NULL_ERROR, NULL_MEAN = 0.01, (0.48, 0.52)


def _save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=float) + "\n")


def read_bads(edf: Path) -> tuple[tuple[str, ...], bool]:
    """The bad electrodes of a recording and whether it has too many, from its ``-bads.json`` sidecar; no bad
    electrode without a sidecar."""
    sidecar = Path(str(edf).replace(".edf", "-bads.json"))
    if not sidecar.is_file():
        return (), False
    record = json.loads(sidecar.read_text())
    return tuple(record.get("bad_channels", [])), bool(record.get("too_many_bad_channels", False))


def cache_path(root: Path, group: str, file: str) -> Path:
    """The cached signal of a training file."""
    return root / "cache" / "train" / group / file.replace(".edf", ".npy")


# ---------------------------------------------------------------------- cache and reader check (HPC)


def _cache_one(edf: Path, out: Path) -> str | None:
    from src.data.components.tuev import read_recording

    if out.is_file():
        return None
    try:
        data = read_recording(edf)
    except Exception as error:  # noqa: BLE001 - a recording that cannot be read is counted, and the cache goes on
        return f"{edf}: {error}"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, data)
    return None


def cache(root: Path, events_dir: Path, n_jobs: int = 8) -> None:
    """Write the signal of every training recording through the chain of the campaign. Finished files are kept."""
    from joblib import Parallel, delayed

    from src.data.components.tuev import CHANNELS

    files = pl.read_csv(root / "files.csv", schema_overrides={"group": pl.Utf8})
    files = files.filter((pl.col("set") == "train") & pl.col("header_ok") & pl.col("has_rec"))
    jobs = [
        (events_dir / "edf" / "train" / group / file, cache_path(root, group, file))
        for group, file in files.select("group", "file").iter_rows()
    ]
    errors = Parallel(n_jobs=n_jobs)(delayed(_cache_one)(edf, out) for edf, out in jobs)
    failed = [error for error in errors if error]
    for error in failed:
        logger.error(error)
    sidecars = sum(Path(str(edf).replace(".edf", "-bads.json")).is_file() for edf, _ in jobs)
    _save_json(root / "cache" / "channels.json", list(CHANNELS))
    _save_json(root / "cache" / "cache.json", {"files": len(jobs), "failed": failed, "bad_channel_sidecars": sidecars})
    logger.info(f"cache: {len(jobs) - len(failed)} of {len(jobs)} recordings; {sidecars} have a bad-channel sidecar")


def reader_check(args: argparse.Namespace, n_recordings: int = 6) -> dict[str, Any]:
    """Gate 3: on recordings of the epilepsy corpus, the reader of this corpus gives the rows of the campaign."""
    from src.data.components.tuev import CHANNELS, read_recording
    from src.frozen_readout import load_sources

    _, _, sources, _, _ = load_sources(args)
    source = sources["train"]
    if list(source.target_channels) != list(CHANNELS):
        raise ValueError("the channels of the campaign and of the reader differ")
    seen: set[str] = set()
    rows = []
    for index in range(len(source)):
        description = source.plan["description_row"].iloc[index]
        path, start = str(description["path"]), float(source.plan["start"].iloc[index])
        if path in seen:
            continue
        seen.add(path)
        theirs = source[index][0].numpy().astype(np.float64)
        first = int(round(start * SFREQ))
        mine = read_recording(Path(path))[:, first : first + theirs.shape[1]].astype(np.float64)
        scale = 1.4826 * np.median(np.abs(theirs - np.median(theirs, axis=1, keepdims=True)), axis=1)
        gap = float((np.abs(mine - theirs).max(axis=1) / np.maximum(scale, 1e-30)).max())
        rows.append({"path": path, "start": start, "gap": gap})
        if len(rows) == n_recordings:
            break
    record = {
        "recordings": rows,
        "largest_gap": max(row["gap"] for row in rows),
        "tolerance": READER_TOLERANCE,
        "pass": all(row["gap"] <= READER_TOLERANCE for row in rows),
    }
    _save_json(args.root.resolve() / "reader_check.json", record)
    logger.info(f"reader check: largest gap {record['largest_gap']:.2e} ({'pass' if record['pass'] else 'FAIL'})")
    return record


# ---------------------------------------------------------------------- units


def unit_table(root: Path, events_dir: Path, channels: list[str]) -> tuple[pl.DataFrame, dict[str, int]]:
    """The kept channel-seconds of the training patients that are in a fold: patient, fold, file, record, channel
    (its position in the cached signal), start in the record, and class. Units on a channel with a repaired
    electrode leave, and so do all units of a recording with too many bad channels."""
    position = {name.upper(): i for i, name in enumerate(channels)}
    folds = json.loads((root / "folds.json").read_text())["folds"]
    files = pl.read_csv(root / "files.csv", schema_overrides={"group": pl.Utf8})
    files = files.filter((pl.col("set") == "train") & pl.col("header_ok") & pl.col("has_rec"))
    rows, left = [], {"repaired_channel": 0, "bad_recording": 0, "no_cache": 0}
    for group, file, duration in files.select("group", "file", "duration_s").iter_rows():
        if group not in folds:
            continue
        edf = events_dir / "edf" / "train" / group / file
        kept = [u for u in units(parse_rec(edf.with_suffix(".rec").read_text()), duration) if u["status"] == "kept"]
        bad, too_many = read_bads(edf)
        if too_many:
            left["bad_recording"] += len(kept)
            continue
        if not cache_path(root, group, file).is_file():
            left["no_cache"] += len(kept)
            continue
        bad_upper = {name.upper() for name in bad}
        file_pieces = pieces(duration)
        for unit in kept:
            name = MONTAGE[unit["channel"]]
            if bad_upper & set(name.split("-")):
                left["repaired_channel"] += 1
                continue
            rows.append(
                {
                    "patient": group,
                    "fold": folds[group],
                    "file": file,
                    "piece": unit["piece"],
                    "piece_start": file_pieces[unit["piece"]][0],
                    "piece_stop": file_pieces[unit["piece"]][1],
                    "channel": position[name],
                    "start": unit["start"] - file_pieces[unit["piece"]][0],
                    "label": unit["label"],
                }
            )
    return pl.DataFrame(rows), left


# ---------------------------------------------------------------------- features


def dictionaries(sweep_root: Path | None, n_random: int = N_RANDOM) -> tuple[dict[str, np.ndarray], list[dict]]:
    """The waveforms of each band, stacked, and the description of every feature column in order: for each band,
    the ``Transfer`` dictionaries (1 to 30 Hz only), the random dictionaries, then the simple measures."""
    atoms: dict[str, np.ndarray] = {}
    columns: list[dict[str, Any]] = []
    for band in BANDS:
        parts = []
        if band == "30" and sweep_root is not None:
            for fold in range(N_FOLDS):
                waves = np.load(sweep_root / "dictionaries" / f"{ADOPTED}_f{fold}" / "sae_time_atoms.npy")
                parts.append(waves.astype(np.float32))
                columns += [
                    {"band": band, "family": "transfer", "dictionary": f"f{fold}", "index": k}
                    for k in range(len(waves))
                ]
        for index in range(n_random):
            waves = random_atoms(band, index)
            parts.append(waves)
            columns += [
                {"band": band, "family": "untrained", "dictionary": f"r{index}", "index": k} for k in range(len(waves))
            ]
        atoms[band] = np.concatenate(parts)
        columns += [
            {"band": band, "family": "simple", "dictionary": name, "index": i}
            for i, name in enumerate(simple_names(band))
        ]
    return atoms, columns


def null_starts(starts: np.ndarray, n_seconds: int, rng: np.random.Generator) -> np.ndarray:
    """For each unit, the start of a whole second of the same record that is not next to a marked second; -1 if the
    record has none. The rank at such a second is the null of the read (gate 5)."""
    marked = np.unique(np.floor(starts / SFREQ).astype(int))
    free = np.ones(n_seconds, dtype=bool)
    free[[0, n_seconds - 1]] = False
    for second in marked:
        free[max(second - 1, 0) : second + 3] = False
    candidates = np.flatnonzero(free)
    if not len(candidates):
        return np.full(len(starts), -1)
    return rng.choice(candidates, size=len(starts)) * SFREQ


def record_reads(
    record: np.ndarray, channel: np.ndarray, start: np.ndarray, null: np.ndarray, atoms: dict[str, torch.Tensor]
) -> dict[str, np.ndarray]:
    """The three reads ``(units, columns)`` of the units of one record. ``start`` and ``null`` are in samples."""
    device = next(iter(atoms.values())).device
    distinct = sorted(set(start.tolist()) | {s for s in null.tolist() if s >= 0})
    where = {s: i for i, s in enumerate(distinct)}
    out: dict[str, list[np.ndarray]] = {read: [] for read in READS}
    true_index = torch.tensor([where[s] for s in start.tolist()], device=device)
    null_index = torch.tensor([where.get(s, 0) for s in null.tolist()], device=device)
    channels = torch.as_tensor(np.array(channel), device=device)
    for band in BANDS:
        grid, at = record_features(band_rows(record, band).to(device), atoms[band], band, distinct)
        ranks = torch.stack([rank_in_record(at[i], grid, s) for i, s in enumerate(distinct)])
        out["rank"].append(ranks[true_index, channels].cpu().numpy())
        out["plain"].append(at[true_index, channels].cpu().numpy())
        displaced = ranks[null_index, channels].cpu().numpy()
        displaced[null < 0] = np.nan
        out["null"].append(displaced)
    return {read: np.concatenate(parts, axis=1).astype(np.float32) for read, parts in out.items()}


def alignment_check(root: Path, table: pl.DataFrame, waves: np.ndarray, device: torch.device) -> dict[str, Any]:
    """Gate 4. In records of fold 0, one waveform is planted at one background unit at an amplitude of 6. Its rank
    must separate the planted unit from the other background units of the record, and it must not when every unit
    is read 2 s later."""
    rng = np.random.default_rng(0)
    background = table.filter((pl.col("fold") == 0) & (pl.col("label") == "bckg"))
    planted, shifted, labels, shifted_labels = [], [], [], []
    groups = background.group_by("patient", "file", "piece", maintain_order=True)
    for count, ((patient, file, _), items) in enumerate(groups):
        if len(planted) >= ALIGN_RECORDS:
            break
        signal = np.load(cache_path(root, patient, file), mmap_mode="r")
        first, last = int(items["piece_start"][0] * SFREQ), int(items["piece_stop"][0] * SFREQ)
        rows = band_rows(np.asarray(signal[:, first:last]), "30").to(device)
        channel = items["channel"].to_numpy()
        start = np.round(items["start"].to_numpy() * SFREQ).astype(int)
        target = int(rng.integers(len(start)))
        others = (channel != channel[target]) | (np.abs(start - start[target]) >= 4 * SFREQ)
        if others.sum() < 3:
            continue
        wave = torch.from_numpy(waves[count % len(waves)]).to(device)
        centre = start[target] + SFREQ // 2
        rows[channel[target], centre - 64 : centre + 64] += PLANT_AMPLITUDE * wave / wave.abs().max()
        response = responses(rows, wave.unsqueeze(0))
        n_seconds = rows.shape[-1] // SFREQ
        grid = response[..., : n_seconds * SFREQ].reshape(rows.shape[0], 1, n_seconds, SFREQ).amax(-1)
        use = np.flatnonzero(others | (np.arange(len(start)) == target))
        for shift, values, flags in ((0, planted, labels), (2 * SFREQ, shifted, shifted_labels)):
            for i in use:
                s = int(start[i]) + shift
                if s + SFREQ > rows.shape[-1]:
                    continue
                value = response[..., s : s + SFREQ].amax(-1)
                values.append(float(rank_in_record(value, grid, s)[channel[i], 0]))
                flags.append(i == target)
    score = float(auroc_columns(np.array(planted)[:, None], np.array(labels))[0]) if planted else float("nan")
    moved = float(auroc_columns(np.array(shifted)[:, None], np.array(shifted_labels))[0]) if shifted else float("nan")
    return {
        "records": int(np.sum(labels)),
        "auroc": score,
        "auroc_2s_later": moved,
        "pass": bool(score >= ALIGN_PASS and moved < ALIGN_SHIFTED),
    }


def features(
    root: Path, events_dir: Path, sweep_root: Path | None, device: str = "cuda", n_random: int = N_RANDOM
) -> None:
    """The three reads of every kept unit, the unit table and the time-alignment check."""
    target = torch.device(device)
    channels = json.loads((root / "cache" / "channels.json").read_text())
    table, left = unit_table(root, events_dir, channels)
    waves, columns = dictionaries(sweep_root, n_random)
    atoms = {band: torch.from_numpy(values).to(target) for band, values in waves.items()}
    rng = np.random.default_rng(0)
    parts: dict[str, list[np.ndarray]] = {read: [] for read in READS}
    ordered = []
    with torch.no_grad():
        for (patient, file, _), items in table.group_by("patient", "file", "piece", maintain_order=True):
            signal = np.load(cache_path(root, patient, file), mmap_mode="r")
            first, last = int(items["piece_start"][0] * SFREQ), int(items["piece_stop"][0] * SFREQ)
            record = np.asarray(signal[:, first:last])
            start = np.round(items["start"].to_numpy() * SFREQ).astype(int)
            null = null_starts(start, record.shape[1] // SFREQ, rng)
            reads = record_reads(record, items["channel"].to_numpy(), start, null, atoms)
            for read in READS:
                parts[read].append(reads[read])
            ordered.append(items)
        alignment = alignment_check(root, table, waves["30"], target)
    folder = root / "pilot"
    folder.mkdir(parents=True, exist_ok=True)
    for read in READS:
        np.save(folder / f"{read}.npy", np.concatenate(parts[read]))
    pl.concat(ordered).write_csv(folder / "units.csv")
    _save_json(folder / "columns.json", columns)
    _save_json(folder / "alignment.json", alignment)
    _save_json(folder / "units.json", {"units": table.height, "units_that_leave": left})
    logger.info(f"features: {table.height} units, {len(columns)} columns; alignment {alignment}")


# ---------------------------------------------------------------------- read


def family_columns(columns: list[dict[str, Any]], family: str, fold: int) -> list[int]:
    """The columns that a family can choose in a fold. ``Transfer`` uses one dictionary of the epilepsy corpus per
    fold; the other families use all their columns."""
    return [
        i
        for i, column in enumerate(columns)
        if column["family"] == family and (family != "transfer" or column["dictionary"] == f"f{fold}")
    ]


def read(root: Path, n_boot: int = 10_000) -> dict[str, Any]:
    """The pilot read: scores, gains and gates (protocol sections 6, 7 and 10). Descriptive."""
    folder = root / "pilot"
    table = pl.read_csv(folder / "units.csv", schema_overrides={"patient": pl.Utf8})
    columns = json.loads((folder / "columns.json").read_text())
    label, patient, fold = table["label"].to_numpy(), table["patient"].to_numpy(), table["fold"].to_numpy()
    draws = [
        {"patients": np.unique(patient[fold == f]), "counts": patient_draws(patient[fold == f], n_boot, seed=f)}
        if (fold == f).any()
        else None
        for f in range(N_FOLDS)
    ]
    families = [f for f in FAMILIES if any(column["family"] == f for column in columns)]
    summary: list[dict[str, Any]] = []
    null_boots: list[np.ndarray] = []
    null_mean = float("nan")
    classes: dict[str, Any] = {}
    for name in READS:
        matrix = np.load(folder / f"{name}.npy")
        if name == "null":
            null_mean = float(np.nanmean(matrix))
        for target in EVENT_CLASSES:
            arms = {}
            for family in families:
                chosen = choose_and_score(
                    matrix,
                    [family_columns(columns, family, f) for f in range(N_FOLDS)],
                    label,
                    patient,
                    fold,
                    target,
                    N_FOLDS,
                )
                arms[family] = summarize(chosen, draws) | {
                    "chosen": [columns[e["column"]] if e is not None else None for e in chosen]
                }
            entry: dict[str, Any] = {}
            for family, arm in arms.items():
                known = arm["score"] is not None
                entry[family] = {
                    "score": arm["score"],
                    "interval": interval(arm["boot"]) if known else None,
                    "chosen": arm["chosen"],
                }
                if name == "null" and known and target in DECIDING:
                    null_boots.append(arm["boot"])
                summary.append(
                    {
                        "read": name,
                        "class": target,
                        "family": family,
                        "score": arm["score"],
                        "low": entry[family]["interval"][0] if known else None,
                        "high": entry[family]["interval"][1] if known else None,
                    }
                )
            if "transfer" in arms and arms["transfer"]["score"] is not None:
                gains = {}
                for other in ("untrained", "simple"):
                    difference = arms["transfer"]["boot"] - arms[other]["boot"]
                    gains[other] = {
                        "gain": arms["transfer"]["score"] - arms[other]["score"],
                        "low": interval(difference, LEVEL)[0],
                        "high": interval(difference, LEVEL)[1],
                    }
                entry["gain_over_untrained"], entry["gain_over_simple"] = gains["untrained"], gains["simple"]
                if name == "rank":
                    entry["outcome_if_it_decided"] = class_outcome(
                        arms["transfer"]["score"], gains["untrained"], gains["simple"]
                    )
            classes.setdefault(target, {})[name] = entry
    null_level = 1 - NULL_ERROR / max(len(null_boots), 1)
    null_intervals = [interval(boot, null_level) for boot in null_boots]
    alignment = json.loads((folder / "alignment.json").read_text())
    reader_file = root / "reader_check.json"
    reader = json.loads(reader_file.read_text()) if reader_file.is_file() else None
    gates = {
        "reader_agrees": bool(reader["pass"]) if reader is not None else None,
        "time_alignment": bool(alignment["pass"]),
        "displaced_units_at_chance": bool(null_intervals)
        and all(low <= 0.5 <= high for low, high in null_intervals)
        and NULL_MEAN[0] <= null_mean <= NULL_MEAN[1],
    }
    record = {
        "protocol": f"{PROTOCOL}, section 9 (stage 1, descriptive)",
        "units": table.height,
        "patients": int(len(np.unique(patient))),
        "units_per_class": {name: int((label == name).sum()) for name in CLASSES.values()},
        "units_that_leave": json.loads((folder / "units.json").read_text())["units_that_leave"],
        "gates": gates,
        "status": "pass" if all(value is not False for value in gates.values()) else "blocked",
        "alignment": alignment,
        "displaced_units": {"mean_rank": null_mean, "intervals": len(null_intervals), "level": null_level},
        "reader_check": reader,
        "confound": {
            target: {
                "simple_plain": classes[target]["plain"]["simple"]["score"],
                "simple_rank": classes[target]["rank"]["simple"]["score"],
            }
            for target in EVENT_CLASSES
        },
        "classes": classes,
    }
    pl.DataFrame(summary, infer_schema_length=None).write_csv(root / "pilot_summary.csv")
    _save_json(root / "pilot.json", record)
    logger.info(f"pilot: {record['status']} {gates}")
    return record


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("cache", help="the signal of every training recording (HPC)")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--events-dir", type=Path, required=True)
    command.add_argument("--n-jobs", type=int, default=8)
    command = commands.add_parser("reader-check", help="gate 3, on recordings of the epilepsy corpus (HPC)")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--manifests", type=Path, required=True, help="the 40 s offset audit")
    command.add_argument("--data-dir", type=Path, required=True, help="the root of the epilepsy corpus")
    command.add_argument("--signal-cache-dir", type=Path, required=True, help="the campaign signal cache")
    command.add_argument("--num-workers", type=int, default=4)
    command = commands.add_parser("features", help="the three reads of every unit, and gate 4 (GPU)")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--events-dir", type=Path, required=True)
    command.add_argument("--sweep-root", type=Path, required=True, help="the root with the adopted dictionaries")
    command.add_argument("--device", default="cuda")
    command = commands.add_parser("read", help="scores, gains and gates")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "cache":
        cache(root, args.events_dir.resolve(), args.n_jobs)
    elif args.command == "reader-check":
        reader_check(args)
    elif args.command == "features":
        features(root, args.events_dir.resolve(), args.sweep_root.resolve(), args.device)
    else:
        read(root, args.n_boot)


if __name__ == "__main__":
    main()
