"""Small helpers shared across cavityscope."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_measurement_output_dirs(base_output_dir: str | Path) -> Dict[str, Path]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(base_output_dir) / f"measurement_{timestamp}")
    return {
        "run_dir": run_dir,
        "traces_dir": ensure_dir(run_dir / "trace_plots"),
        "refs_dir": ensure_dir(run_dir / "reference_plots"),
        "raw_dir": ensure_dir(run_dir / "raw_traces"),
        "fit_dir": ensure_dir(run_dir / "fit_plots"),
        "freq_dir": ensure_dir(run_dir / "frequency_plots"),
    }


def make_calibration_output_dir(base_output_dir: str | Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(base_output_dir) / f"calibration_{timestamp}")


def dbm_to_watts(dbm: float) -> float:
    return 1e-3 * (10.0 ** (dbm / 10.0))


def dbm_to_vrms_into_r(dbm: float, r_ohm: float) -> float:
    return math.sqrt(dbm_to_watts(dbm) * r_ohm)


def robust_baseline(y: np.ndarray, percentile: float) -> float:
    if y.size == 0:
        return 0.0
    return float(np.percentile(y, percentile))


def robust_noise_sigma(y: np.ndarray) -> float:
    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med)))
    sigma = 1.4826 * mad
    return max(sigma, 1e-15)


def hz_window_to_half_time(
    width_hz: float, fsr_hz: float, fsr_time_s: float
) -> float:
    return 0.5 * float(width_hz) / float(fsr_hz) * float(fsr_time_s)


class IncrementalCsvWriter:
    """Write rows to a CSV file one at a time, flushing after each row.

    The header is written automatically from the keys of the first row.
    Subsequent rows are appended and flushed immediately, so the file on
    disk always contains every row written so far — even if the process
    crashes before the sweep finishes.

    Usage::

        with IncrementalCsvWriter(path) as w:
            for row in rows:
                w.write_row(row)
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._file = None
        self._writer: Optional[csv.DictWriter] = None
        self._columns: Optional[List[str]] = None

    def write_row(self, row: dict) -> None:
        if self._file is None:
            self._columns = list(row.keys())
            self._file = open(self.path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self._columns,
                extrasaction="ignore",
                restval="",
            )
            self._writer.writeheader()
            self._file.flush()
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
