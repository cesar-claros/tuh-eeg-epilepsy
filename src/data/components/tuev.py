"""Reader of the TUH EEG Events Corpus: one recording through the signal chain of the campaign.

Protocol: ``documentation/event_corpus_dictionary_protocol.md``, section 3. The chain is that of
``WindowDataset._process`` with the settings of the waveform campaign (``configs/train_sae_waveform.yaml``): the
0.1 Hz high-pass and the 60 Hz notch at the native rate, the repair of the bad channels of the ``-bads.json``
sidecar, the TCP bipolar montage, and the resampling to 256 Hz. It works on the whole recording, which equals the
campaign's filter of a crop with its full context (``tests/test_temporal_filtering.py``). Gate 3 of the protocol
compares its output with the campaign's rows on recordings of the epilepsy corpus.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy
from src.data.components.window_dataset import _read_bads, _resample

TARGET_SFREQ = 256.0
HIGH_PASS_HZ, NOTCH_HZ = 0.1, (60.0,)
# The electrodes of the TCP montage files of the corpus (19, with A1 and A2): the channels that the campaign reads.
MONTAGE_ELECTRODES = frozenset(e.upper() for pair in TUHEEGEpilepsy._TCP_BIPOLAR for e in pair)
# The 20 channels of the campaign, in its order (sorted names; no pair with an ear electrode).
CHANNELS = tuple(sorted(f"{a}-{c}" for a, c in TUHEEGEpilepsy._TCP_BIPOLAR if not {"A1", "A2"} & {a, c}))


def montage_channels(names: list[str]) -> list[str]:
    """The channels of a file that the montage uses, such as ``EEG FP1-REF``."""
    out = []
    for name in names:
        upper = name.upper().strip()
        if upper.startswith("EEG ") and upper[4:].split("-")[0].strip() in MONTAGE_ELECTRODES:
            out.append(name)
    return out


def read_recording(path: Path) -> np.ndarray:
    """One recording as ``(20, samples)`` float32 at 256 Hz, in the order of ``CHANNELS``.

    Raises
    ------
    ValueError
        If the bipolar montage cannot be formed or a channel of ``CHANNELS`` is absent.
    """
    names = TUHEEGEpilepsy._read_raw_edf(path, preload=False).ch_names
    raw = TUHEEGEpilepsy._read_raw_edf(path, include=montage_channels(names), preload=True)
    raw.filter(l_freq=HIGH_PASS_HZ, h_freq=None, verbose="ERROR")
    raw.notch_filter(list(NOTCH_HZ), verbose="ERROR")
    TUHEEGEpilepsy._rename_channels(raw)
    bad, _, too_many = _read_bads(str(path))
    present = [c for c in bad if c in raw.ch_names]
    if present and not too_many:
        TUHEEGEpilepsy._set_montage(raw)
        raw.info["bads"] = present
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    bipolar = TUHEEGEpilepsy._apply_bipolar(raw)
    if bipolar is None:
        raise ValueError(f"{path}: the bipolar montage cannot be formed")
    index = {name: i for i, name in enumerate(bipolar.ch_names)}
    missing = [c for c in CHANNELS if c not in index]
    if missing:
        raise ValueError(f"{path}: no channels {missing}")
    data = bipolar.get_data()[[index[c] for c in CHANNELS]]
    sfreq = float(bipolar.info["sfreq"])
    if not np.isclose(sfreq, TARGET_SFREQ):
        data = _resample(data, sfreq, TARGET_SFREQ)
    return data.astype(np.float32)
