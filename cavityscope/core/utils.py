"""Small helpers shared across cavityscope."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict

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
