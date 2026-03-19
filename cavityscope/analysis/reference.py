"""Reference-trace (RF-off) analysis: find cavity resonances and pick a carrier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from cavityscope.core.config import SweepConfig
from cavityscope.core.utils import robust_baseline


@dataclass
class ReferenceInfo:
    frequency_hz: float
    fsr_time_s: float
    baseline_v: float
    edge_margin_s: float
    all_carrier_times_s: List[float]
    chosen_carrier_time_s: float
    chosen_carrier_height_v: float


def _smooth_and_baseline(y_v: np.ndarray, cfg: SweepConfig):
    baseline = robust_baseline(y_v, cfg.baseline_percentile)
    y_bs = y_v - baseline
    sigma = cfg.analysis_use_gaussian_sigma_pts
    y_sm = gaussian_filter1d(y_bs, sigma=sigma) if sigma > 0 else y_bs.copy()
    return baseline, y_bs, y_sm


def analyze_reference_trace(
    t: np.ndarray,
    y_v: np.ndarray,
    rf_frequency_hz: float,
    cfg: SweepConfig,
) -> ReferenceInfo:
    """Find cavity resonances in an RF-off trace and select a reference carrier."""
    baseline, _y_bs, y_sm = _smooth_and_baseline(y_v, cfg)

    min_dist_pts = max(
        1, int(cfg.min_peak_distance_fraction_of_trace * y_sm.size)
    )
    prominence = cfg.reference_peak_prominence_fraction * max(
        1e-12, float(y_sm.max() - y_sm.min())
    )
    peaks, _ = find_peaks(y_sm, prominence=prominence, distance=min_dist_pts)
    if peaks.size < cfg.min_reference_peaks_required:
        raise RuntimeError(
            f"Only found {peaks.size} carrier peaks in the RF-off reference trace."
        )

    carrier_times = t[peaks]
    carrier_heights = y_sm[peaks]

    edge_margin_s = cfg.edge_exclusion_fraction_of_trace * float(t[-1] - t[0])
    edge_mask = (carrier_times > t[0] + edge_margin_s) & (
        carrier_times < t[-1] - edge_margin_s
    )
    carrier_times = carrier_times[edge_mask]
    carrier_heights = carrier_heights[edge_mask]
    if carrier_times.size < cfg.min_reference_peaks_required:
        raise RuntimeError(
            "Too many reference peaks are too close to the trace edges."
        )

    if cfg.fsr_time_s is not None:
        fsr_time_s = cfg.fsr_time_s
    else:
        fsr_time_s = float(np.median(np.diff(np.sort(carrier_times))))

    pick_mode = cfg.reference_carrier_pick.lower()
    if pick_mode == "central":
        trace_mid = 0.5 * (t[0] + t[-1])
        idx = int(np.argmin(np.abs(carrier_times - trace_mid)))
    else:
        idx = int(np.argmax(carrier_heights))

    return ReferenceInfo(
        frequency_hz=rf_frequency_hz,
        fsr_time_s=fsr_time_s,
        baseline_v=baseline,
        edge_margin_s=edge_margin_s,
        all_carrier_times_s=carrier_times.tolist(),
        chosen_carrier_time_s=float(carrier_times[idx]),
        chosen_carrier_height_v=float(carrier_heights[idx]),
    )
