"""Project package. Runs before any ``src`` module, so the settings below cover every entry point."""

import os
import warnings

import mne

# MNE logs an INFO line for every window the loader reads ("Reading 0 ... 31999 = ..."), which
# floods the training logs. Keep MNE's warnings and errors. The MNE_LOGGING_LEVEL environment
# variable still overrides this default, for example MNE_LOGGING_LEVEL=INFO for a diagnostic run.
mne.set_log_level(os.environ.get("MNE_LOGGING_LEVEL", "WARNING"))

# braindecode (models/eegpt.py) builds the "standard_1020" montage when it is imported. MNE 1.13
# deprecated that name in favour of "colin27_1020", which has the same positions. The warning is
# braindecode's to fix; this silences exactly that message, and only when it comes from braindecode.
warnings.filterwarnings(
    "ignore",
    message=r"Montage name 'standard_1020' is deprecated",
    category=FutureWarning,
    module=r"braindecode\.",
)
