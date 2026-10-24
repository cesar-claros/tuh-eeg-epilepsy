"""Persistent, lossless cache of deterministic sensor-space window preprocessing."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import BadZipFile

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from src.data.components.window_dataset import WindowDataset


def _stamp(path: str) -> tuple[int, int, int] | None:
    """Cheap source identity; missing files remain distinguishable from present ones."""
    try:
        stat = Path(path).stat()
    except FileNotFoundError:
        return None
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _settings_json(source: WindowDataset) -> str:
    """Freeze preprocessing settings without copying the plan or the cache itself."""
    settings = {key: value for key, value in vars(source).items() if key not in ("plan", "signal_cache")}
    return json.dumps(settings, sort_keys=True, default=str)


class WindowSignalCache:
    """Cache float32 windows without labels, model normalization or learned features.

    Args:
        source: Sensor-space WindowDataset with deterministic preprocessing.
        directory: Shared writable cache root, preferably on fast HPC storage.

    Raises:
        ValueError: Unsupported signal mode or malformed cached/processed arrays.
        RuntimeError: Source files change after the cache is attached.
        OSError: Cache storage cannot be read or written.

    Files are uncompressed NPZ archives with CRC checks, published atomically
    under per-window process locks. Failed loads are never persisted. Identity
    includes ordered metadata, settings, preprocessing code, library versions
    and file size/mtime/ctime. It is not a cryptographic audit of EDF contents.
    """

    def __init__(self, source: WindowDataset, directory: str | Path) -> None:
        if source.mode not in ("raw", "bipolar"):
            raise ValueError("signal caching currently supports raw/bipolar sensor preprocessing only")
        self.settings = _settings_json(source)
        self.shape = (len(source.target_channels), source.target_len)
        self.dependencies: list[tuple[str, ...]] = []
        self.windows: list[tuple[str, str, float, float]] = []
        for row in source.plan.itertuples(index=False):
            description = row.description_row
            # Preserve the loader's sidecar location when the EDF path is a symlink.
            recording = str(Path(str(description["path"])).absolute())
            paths = [recording]
            if source.interpolate_bad_channels or source.apply_aas:
                paths.append(recording.replace(".edf", "-bads.json"))
            self.dependencies.append(tuple(paths))
            self.windows.append(
                (str(description["path"]), str(description["montage"]), float(row.start), float(row.end))
            )
        self.stamps = {path: _stamp(path) for path in sorted({p for paths in self.dependencies for p in paths})}
        # Source code hashes also cover filter-context and repair implementation changes.
        component_dir = Path(__file__).parent
        code = {
            name: hashlib.sha256((component_dir / name).read_bytes()).hexdigest()
            for name in (
                "signal_cache.py",
                "window_dataset.py",
                "tuh_eeg_epilepsy.py",
                "temporal_filtering.py",
                "aas.py",
            )
        }
        identity = {
            "format": 1,
            "settings": self.settings,
            "plan": source.plan.to_json(orient="split", index=False, double_precision=15, default_handler=str),
            "sources": self.stamps,
            "code": code,
            "python": platform.python_version(),
            "packages": {name: version(name) for name in ("numpy", "scipy", "mne", "torch", "braindecode")},
        }
        encoded = json.dumps(identity, sort_keys=True, default=str).encode()
        self.identity = hashlib.sha256(encoded).hexdigest()
        self.directory = Path(directory).expanduser().resolve() / self.identity
        self.directory.mkdir(parents=True, exist_ok=True)
        self.estimated_bytes = len(source) * int(np.prod(self.shape)) * np.dtype(np.float32).itemsize
        existing = sum(1 for _ in self.directory.glob("*.npz"))
        logger.info(
            "Signal cache {}: {}/{} windows present; full float32 payload {:.2f} GiB",
            self.directory,
            existing,
            len(source),
            self.estimated_bytes / 2**30,
        )

    def _check_sources(self, index: int) -> None:
        for path in self.dependencies[index]:
            if _stamp(path) != self.stamps[path]:
                raise RuntimeError(f"signal cache source changed during this run: {path}; start a new campaign")

    def get(self, source: WindowDataset, index: int) -> np.ndarray | None:
        """Return the exact processed window, computing it once on a cache miss.

        Unreadable or malformed cache entries are logged and rebuilt under the
        lock. A loader failure returns None as before, without caching its zero
        placeholder. Source changes and storage-write failures raise.
        """
        row = source.plan.iloc[index]
        description = row["description_row"]
        window = (str(description["path"]), str(description["montage"]), float(row["start"]), float(row["end"]))
        if _settings_json(source) != self.settings or window != self.windows[index]:
            raise RuntimeError("signal cache settings/window changed after attachment; start a new campaign")
        self._check_sources(index)
        path = self.directory / f"{index:08d}.npz"
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                try:
                    with np.load(path, allow_pickle=False) as archive:
                        data = archive["signal"]
                    self._validate(data)
                    self._check_sources(index)
                    return data
                except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile) as exc:
                    logger.warning("Rebuilding unreadable signal cache {}: {}", path, exc)
            data = source._process(row)
            self._check_sources(index)
            if data is None:
                return None
            self._validate(data)
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as handle:
                    temporary = Path(handle.name)
                    np.savez(handle, signal=data)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return data

    def _validate(self, data: np.ndarray) -> None:
        if data.shape != self.shape or data.dtype != np.float32:
            raise ValueError(f"expected float32 signal shape {self.shape}, got {data.dtype} {data.shape}")
