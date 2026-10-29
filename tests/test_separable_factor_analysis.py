"""Checks of the read-only factor analysis ``src/separable_factor_analysis.py`` (no corpus; CPU; seconds).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the
first failure. Covered: the template (both row domains) and the topography against the gate-1
driver, the dominant frequency, the lag coherence and the signed best lag, and the arm summaries
and event-atom rows of written format-5 and separable factors.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402

import src.separable_factor_analysis as fa  # noqa: E402
import src.separable_synthetic as gate  # noqa: E402
from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: E402

CHANNELS = [f"{a}-{c}" for a, c in TUHEEGEpilepsy._TCP_BIPOLAR]


def _write_arm(folder: Path, energy: list[float], focus_atoms: tuple[int, int]) -> None:
    folder.mkdir()
    (folder / "sae_channels.txt").write_text("\n".join(CHANNELS) + "\n")
    with (folder / "sae_atom_stats.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["atom", "energy_share"])
        writer.writerows(enumerate(energy))
    readout = {"amplitude": 6, "T3_atom": focus_atoms[0], "C4_atom": focus_atoms[1]}
    (folder / "synthetic_readout.json").write_text(json.dumps(readout))


def test_formulas_match_the_gate1_driver() -> None:
    assert np.allclose(fa.spike_wave(77), gate.event_template(256.0)[0]), "the template drifted from the driver"
    for pre_emphasis in ("none", "diff"):
        assert np.allclose(fa.row_template(77, pre_emphasis), gate.row_template(256.0, pre_emphasis))
    without_ears = [c for c in CHANNELS if "A1" not in c and "A2" not in c]
    for channels in (CHANNELS, without_ears):
        for focus in fa.FOCI:
            assert np.allclose(fa.focal_topography(channels, focus), gate.focal_topography(channels, focus))


def test_spectrum_coherence_and_lag() -> None:
    t = np.arange(256) / fa.SFREQ
    waveforms = np.stack([np.sin(2 * np.pi * 10.0 * t), np.sin(2 * np.pi * 3.0 * t)])
    assert np.allclose(fa.dominant_frequency(waveforms), [10.0, 3.0], atol=0.2)
    bump = np.r_[np.hanning(50), np.zeros(206)]
    shifted = np.stack([np.roll(bump, shift) for shift in (0, 40)])[:, None, :]
    assert abs(fa.lag_coherence(shifted) - 1.0) < 1e-9, "shifted copies are not fully coherent"
    template = fa.spike_wave(77)
    waveform = np.zeros(256)
    waveform[30:107] = -template
    correlation, offset = fa.best_lag(waveform, template)
    assert abs(correlation + 1.0) < 1e-9 and offset == 30


def test_arm_summaries_and_event_atoms() -> None:
    template, slope = fa.spike_wave(77), np.linspace(-1.0, 1.0, 256)
    waveform = np.zeros(256)
    waveform[93:170] = template
    u_t3, u_c4 = (fa.focal_topography(CHANNELS, focus) for focus in fa.FOCI)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        format5 = root / "1b_f5_a6_s0"
        _write_arm(format5, [0.3, 0.7], focus_atoms=(1, 0))
        flat = fa.unit(np.ones(len(CHANNELS)))
        np.save(format5 / "sae_atoms.npy", np.stack([np.outer(u_c4, waveform), np.outer(flat, slope)]))
        summary = fa.summarize_arm(format5)
        assert abs(summary["median_rank_one_share"] - 1.0) < 1e-9 and summary["median_dominant_hz"] == 0.5
        c4 = next(row for row in fa.event_atoms(format5) if row["focus"] == "C4")
        assert abs(abs(c4["waveform_correlation"]) - 1.0) < 1e-9 and c4["offset"] == 93
        assert abs(c4["best_effective_cosine"] - 1.0) < 1e-9, "the rank-1 signs were not combined"

        separable = root / "1b_s4ext_a6_s0"
        _write_arm(separable, [0.6, 0.4], focus_atoms=(0, 0))
        np.save(separable / "sae_time_atoms.npy", np.stack([fa.unit(waveform), fa.unit(slope)]))
        projections = np.stack([np.stack([u_c4, -u_t3], axis=1), np.stack([flat, -flat], axis=1)])
        np.save(separable / "sae_channel_atoms.npy", projections)
        summary = fa.summarize_arm(separable)
        assert summary["atoms_with_opposite_projections"] == 1 and summary["atoms_with_duplicate_projections"] == 0
        rows = {row["focus"]: row for row in fa.event_atoms(separable)}
        assert abs(rows["C4"]["best_effective_cosine"] - 1.0) < 1e-9
        assert abs(rows["T3"]["worst_effective_cosine"] + 1.0) < 1e-9, "the inverted T3 projection was missed"


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
