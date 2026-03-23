"""S21-like resonance analysis and Vpi overdriving detection.

When an EOM is driven through a scanning Fabry-Perot cavity, the ratio of
sideband power to total measured power traces out a transfer function vs
RF frequency.  This module computes that response and finds resonances.

Metrics
-------
sideband_fraction
    ``(SB- + SB+) / (Carrier + SB- + SB+)``, bounded [0, 1].  Numerically
    stable even at the carrier null, but non-monotonic past Vpi.

s21_like_db
    ``10 * log10(sideband_fraction)``.

absolute_sideband_efficiency
    ``(SB- + SB+) / reference_carrier_area``, approximately ``2 * J1(beta)^2``.
    Requires the RF-off reference carrier area.  Gives the fraction of total
    optical power converted to first-order sidebands, independent of where
    the remaining power ends up.

Overdriving detection
---------------------
When the modulation index exceeds the first J0 zero (beta ~ 2.405) the
carrier vanishes and power migrates to higher-order sidebands that the
first-order integration windows do not capture.  Detection uses *power
depletion*::

    power_depletion = 1 - (Carrier + SB- + SB+) / reference_carrier_area

A depletion above ``cfg.overdriving_depletion_threshold`` flags the point.
The ``sideband_fraction`` metric remains useful for locating resonance
centres (it peaks at the carrier null), but its dB value is unreliable for
quantifying transfer efficiency in overdriven regions.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from cavityscope.core.config import SweepConfig


def _auto_savgol_window(n_points: int) -> int:
    """Pick an odd Savitzky-Golay window length for *n_points* samples."""
    if n_points < 5:
        return max(3, n_points | 1)
    w = min(31, max(5, n_points // 3))
    return w | 1


def _light_smooth(y: np.ndarray, window: int = 7) -> np.ndarray:
    """Apply a narrow Savitzky-Golay filter that preserves narrow peak shapes."""
    n = len(y)
    if n < 5:
        return y.copy()
    w = min(window, n) | 1
    po = min(2, w - 1)
    return savgol_filter(y, window_length=w, polyorder=po, mode="interp")


def _measure_3db_bandwidth(
    freq_mhz: np.ndarray,
    s21_db: np.ndarray,
    peak_idx: int,
) -> Tuple[float, float, float]:
    """Measure the 3 dB bandwidth around a peak.

    Scans left and right from *peak_idx* until the signal drops 3 dB
    below the peak value, using linear interpolation at the crossings.

    Returns ``(fwhm_mhz, left_mhz, right_mhz)``.
    """
    n = len(freq_mhz)
    peak_val = float(s21_db[peak_idx])
    threshold = peak_val - 3.0

    left_mhz = float(freq_mhz[0])
    for i in range(peak_idx - 1, -1, -1):
        if s21_db[i] <= threshold:
            y0, y1 = float(s21_db[i]), float(s21_db[i + 1])
            f0, f1 = float(freq_mhz[i]), float(freq_mhz[i + 1])
            denom = y1 - y0
            if abs(denom) > 1e-12:
                left_mhz = f0 + (threshold - y0) / denom * (f1 - f0)
            else:
                left_mhz = 0.5 * (f0 + f1)
            break

    right_mhz = float(freq_mhz[-1])
    for i in range(peak_idx + 1, n):
        if s21_db[i] <= threshold:
            y0, y1 = float(s21_db[i - 1]), float(s21_db[i])
            f0, f1 = float(freq_mhz[i - 1]), float(freq_mhz[i])
            denom = y0 - y1
            if abs(denom) > 1e-12:
                right_mhz = f0 + (y0 - threshold) / denom * (f1 - f0)
            else:
                right_mhz = 0.5 * (f0 + f1)
            break

    return right_mhz - left_mhz, left_mhz, right_mhz


def compute_s21_columns(
    results_df: pd.DataFrame,
    ref_df: Optional[pd.DataFrame] = None,
    cfg: Optional[SweepConfig] = None,
) -> pd.DataFrame:
    """Add S21-like metric columns and overdriving flags to sweep results.

    New columns: ``rf_frequency_mhz``, ``sideband_fraction``,
    ``s21_like_db``, ``s21_like_db_smooth``, ``absolute_sideband_efficiency``,
    ``power_depletion``, ``overdriven``.

    Parameters
    ----------
    results_df : pd.DataFrame
        Sweep results with ``carrier_area_v_s``, ``sb_minus_area_v_s``,
        ``sb_plus_area_v_s``, and ``rf_frequency_hz`` columns.
    ref_df : pd.DataFrame, optional
        Reference summary containing ``reference_carrier_area_v_s``.
        When absent, absolute efficiency and overdriving columns are
        NaN / False.
    cfg : SweepConfig, optional
        Provides smoothing and threshold settings.
    """
    df = results_df.copy()
    cfg = cfg or SweepConfig()
    eps = np.finfo(float).eps

    carrier = np.clip(df["carrier_area_v_s"].to_numpy(dtype=float), 0, None)
    sbm = np.clip(df["sb_minus_area_v_s"].to_numpy(dtype=float), 0, None)
    sbp = np.clip(df["sb_plus_area_v_s"].to_numpy(dtype=float), 0, None)

    sideband_total = sbm + sbp
    total_signal = carrier + sideband_total

    df["rf_frequency_mhz"] = df["rf_frequency_hz"] / 1e6
    df["sideband_fraction"] = sideband_total / np.maximum(total_signal, eps)
    df["s21_like_db"] = 10.0 * np.log10(
        np.maximum(df["sideband_fraction"], eps)
    )

    # ---- per-power-level Savitzky-Golay smoothing for peak finding ----
    df["s21_like_db_smooth"] = df["s21_like_db"].copy()
    for _, grp in df.groupby("rf_power_dbm"):
        sl = grp.sort_values("rf_frequency_hz")
        y = sl["s21_like_db"].to_numpy()
        n = len(y)
        if n < 3:
            continue
        window = cfg.s21_smoothing_window or _auto_savgol_window(n)
        window = min(window, n) | 1
        polyorder = min(3 if window >= 7 else 2, window - 1)
        df.loc[sl.index, "s21_like_db_smooth"] = savgol_filter(
            y, window_length=window, polyorder=polyorder, mode="interp",
        )

    # ---- absolute efficiency & overdriving ----
    has_ref = (
        ref_df is not None
        and "reference_carrier_area_v_s" in ref_df.columns
    )
    if has_ref:
        ref_area_map = dict(zip(
            ref_df["rf_frequency_hz"].astype(float),
            ref_df["reference_carrier_area_v_s"].astype(float),
        ))
        ref_area = (
            df["rf_frequency_hz"]
            .map(ref_area_map)
            .to_numpy(dtype=float)
        )
        ref_area = np.maximum(ref_area, eps)

        df["absolute_sideband_efficiency"] = sideband_total / ref_area
        df["power_depletion"] = 1.0 - total_signal / ref_area

        if cfg.detect_overdriving:
            df["overdriven"] = (
                df["power_depletion"] > cfg.overdriving_depletion_threshold
            )
        else:
            df["overdriven"] = False
    else:
        df["absolute_sideband_efficiency"] = np.nan
        df["power_depletion"] = np.nan
        df["overdriven"] = False

    return df


def find_resonance_peaks(
    freq_mhz: np.ndarray,
    s21_db_raw: np.ndarray,
    s21_db_smooth: np.ndarray,
    cfg: SweepConfig,
) -> pd.DataFrame:
    """Find resonance peaks and measure their 3 dB bandwidth.

    Peak *positions* are detected on the heavily smoothed trace
    (``s21_db_smooth``).  The **3 dB bandwidth** (FWHM) is then
    measured on a lightly smoothed version of the raw trace so that
    narrow resonances are not artificially broadened by the detection
    filter.

    Parameters
    ----------
    freq_mhz : array
        Frequency axis in MHz (sorted ascending).
    s21_db_raw : array
        Raw S21-like dB values (same length as *freq_mhz*).
    s21_db_smooth : array
        Heavily smoothed S21-like dB used for peak detection.
    cfg : SweepConfig
        Peak-finding thresholds.

    Returns
    -------
    pd.DataFrame
        One row per detected peak with ``center_mhz``,
        ``peak_s21_like_db``, ``peak_sideband_fraction``,
        ``prominence_db``, ``fwhm_mhz``, ``q_estimate``,
        ``left_mhz``, ``right_mhz``.
    """
    cols = [
        "center_mhz", "peak_s21_like_db", "peak_sideband_fraction",
        "prominence_db", "fwhm_mhz", "q_estimate", "left_mhz", "right_mhz",
    ]
    n = len(freq_mhz)
    if n < 3:
        return pd.DataFrame(columns=cols)

    dx = float(np.median(np.diff(freq_mhz))) if n > 1 else 1.0
    distance = max(
        1,
        int(round(cfg.s21_peak_min_separation_mhz / max(abs(dx), 1e-9))),
    )

    # Only look for local maxima (upward resonance peaks).
    # The ``height`` filter on the smoothed trace rejects noise bumps
    # early, before any raw-data refinement.
    peaks, props = find_peaks(
        s21_db_smooth,
        prominence=cfg.s21_peak_prominence_db,
        distance=distance,
        height=cfg.s21_peak_threshold_db,
    )
    if len(peaks) == 0:
        return pd.DataFrame(columns=cols)

    # Light smoothing of raw data for 3 dB bandwidth measurement:
    # narrow enough to preserve sharp resonances, wide enough to
    # suppress single-sample noise dips.
    y_bw = _light_smooth(s21_db_raw, window=7)

    rows = []
    for i, pk in enumerate(peaks):
        # Centre frequency and peak level come from the stable
        # smoothed trace; only FWHM is measured on the raw data.
        center = float(freq_mhz[pk])
        peak_db = float(s21_db_smooth[pk])

        # Refine peak position in lightly smoothed raw data for FWHM
        search_r = max(3, int(round(3.0 / max(abs(dx), 1e-9))))
        lo = max(0, pk - search_r)
        hi = min(n, pk + search_r + 1)
        refined = lo + int(np.argmax(y_bw[lo:hi]))

        fwhm, left, right = _measure_3db_bandwidth(freq_mhz, y_bw, refined)

        rows.append({
            "center_mhz": center,
            "peak_s21_like_db": peak_db,
            "peak_sideband_fraction": 10.0 ** (peak_db / 10.0),
            "prominence_db": float(props["prominences"][i]),
            "fwhm_mhz": fwhm,
            "left_mhz": left,
            "right_mhz": right,
        })

    summary = pd.DataFrame(rows)
    summary["q_estimate"] = (
        summary["center_mhz"] / summary["fwhm_mhz"].clip(lower=1e-6)
    )
    return summary.sort_values("center_mhz").reset_index(drop=True)


def compute_resonance_summary(
    results_df: pd.DataFrame,
    ref_df: Optional[pd.DataFrame],
    cfg: SweepConfig,
) -> Dict[float, pd.DataFrame]:
    """Find resonance peaks for each RF power level.

    Returns
    -------
    dict
        Mapping ``rf_power_dbm`` -> peak summary DataFrame.
    """
    s21_df = compute_s21_columns(results_df, ref_df, cfg)

    summaries: Dict[float, pd.DataFrame] = {}
    for pwr, grp in s21_df.groupby("rf_power_dbm"):
        sl = grp.sort_values("rf_frequency_hz")
        summaries[float(pwr)] = find_resonance_peaks(
            sl["rf_frequency_mhz"].to_numpy(),
            sl["s21_like_db"].to_numpy(),
            sl["s21_like_db_smooth"].to_numpy(),
            cfg,
        )
    return summaries
