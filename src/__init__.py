"""Project package. Runs before any ``src`` module, so one warnings filter covers every entry point."""

import warnings

# braindecode (models/eegpt.py) builds the "standard_1020" montage when it is imported. MNE 1.13
# deprecated that name in favour of "colin27_1020", which has the same positions. The warning is
# braindecode's to fix; this silences exactly that message, and only when it comes from braindecode.
warnings.filterwarnings(
    "ignore",
    message=r"Montage name 'standard_1020' is deprecated",
    category=FutureWarning,
    module=r"braindecode\.",
)
