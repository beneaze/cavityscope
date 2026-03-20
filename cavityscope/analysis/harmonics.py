"""Harmonic distortion analysis for RF power calibration.

Analyses the output of :meth:`measure_harmonics` from the spectrum
analyzer driver to characterise signal-generator distortion and
quantify how much power is in the fundamental vs. higher harmonics.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_harmonic_metrics(harmonics: List[Dict]) -> Dict:
    """Compute THD and power fractions from per-harmonic measurements.

    Parameters
    ----------
    harmonics : list of dict
        Each dict must have ``harmonic_number`` and ``power_dbm``.
        The entry with ``harmonic_number == 1`` is the fundamental.

    Returns
    -------
    dict with:

    - ``fundamental_power_dbm`` : float
    - ``total_harmonic_power_dbm`` : float — power in harmonics 2, 3, …
    - ``total_power_dbm`` : float — fundamental + harmonics
    - ``thd_percent`` : float — THD as percentage of fundamental *voltage*
    - ``fundamental_power_fraction`` : float — P_f1 / P_total (linear)
    - ``harmonic_powers_dbm`` : list of (harmonic_number, power_dbm)
    - ``harmonic_levels_dbc`` : list of (harmonic_number, dBc_below_fundamental)
    """
    by_k = {h["harmonic_number"]: h["power_dbm"] for h in harmonics}
    if 1 not in by_k:
        raise ValueError("Harmonic list must contain the fundamental (k=1).")

    p1_dbm = by_k[1]
    p1_w = _dbm_to_watts(p1_dbm)

    harm_watts = []
    dbc_list = []
    for k in sorted(by_k):
        pk_dbm = by_k[k]
        dbc = pk_dbm - p1_dbm
        dbc_list.append((k, dbc))
        if k >= 2:
            harm_watts.append(_dbm_to_watts(pk_dbm))

    sum_harm_w = sum(harm_watts) if harm_watts else 0.0
    total_w = p1_w + sum_harm_w

    thd_pct = 100.0 * math.sqrt(sum_harm_w / p1_w) if p1_w > 0 else 0.0

    return {
        "fundamental_power_dbm": p1_dbm,
        "total_harmonic_power_dbm": _watts_to_dbm(sum_harm_w) if sum_harm_w > 0 else float("-inf"),
        "total_power_dbm": _watts_to_dbm(total_w),
        "thd_percent": thd_pct,
        "fundamental_power_fraction": p1_w / total_w if total_w > 0 else 1.0,
        "harmonic_powers_dbm": [(k, by_k[k]) for k in sorted(by_k)],
        "harmonic_levels_dbc": dbc_list,
    }


def build_harmonics_dataframe(
    all_measurements: List[Dict],
) -> pd.DataFrame:
    """Flatten per-point harmonic data into a tidy DataFrame.

    Parameters
    ----------
    all_measurements : list of dict
        Each entry has ``frequency_hz``, ``power_dbm`` (setting),
        ``harmonics`` (list from ``measure_harmonics``), and optionally
        ``metrics`` (from :func:`compute_harmonic_metrics`).

    Returns
    -------
    pd.DataFrame
        One row per (frequency, power, harmonic_number) with columns:
        ``frequency_hz``, ``power_dbm``, ``harmonic_number``,
        ``nominal_freq_hz``, ``measured_freq_hz``, ``harmonic_power_dbm``,
        ``level_dbc``.
    """
    rows = []
    for m in all_measurements:
        freq = m["frequency_hz"]
        pwr = m["power_dbm"]
        fund_dbm = None
        for h in m["harmonics"]:
            if h["harmonic_number"] == 1:
                fund_dbm = h["power_dbm"]
                break

        for h in m["harmonics"]:
            dbc = h["power_dbm"] - fund_dbm if fund_dbm is not None else float("nan")
            rows.append({
                "frequency_hz": freq,
                "power_dbm": pwr,
                "harmonic_number": h["harmonic_number"],
                "nominal_freq_hz": h["nominal_freq_hz"],
                "measured_freq_hz": h["measured_freq_hz"],
                "harmonic_power_dbm": h["power_dbm"],
                "level_dbc": dbc,
            })
    return pd.DataFrame(rows)


def build_thd_dataframe(
    all_measurements: List[Dict],
) -> pd.DataFrame:
    """Build a summary DataFrame with one row per (frequency, power) point.

    Parameters
    ----------
    all_measurements : list of dict
        Each entry has ``frequency_hz``, ``power_dbm`` (setting), and
        ``metrics`` (from :func:`compute_harmonic_metrics`).

    Returns
    -------
    pd.DataFrame with columns ``frequency_hz``, ``power_dbm``,
    ``fundamental_power_dbm``, ``thd_percent``,
    ``fundamental_power_fraction``, ``total_harmonic_power_dbm``.
    """
    rows = []
    for m in all_measurements:
        met = m.get("metrics", {})
        rows.append({
            "frequency_hz": m["frequency_hz"],
            "power_dbm": m["power_dbm"],
            "fundamental_power_dbm": met.get("fundamental_power_dbm", float("nan")),
            "thd_percent": met.get("thd_percent", float("nan")),
            "fundamental_power_fraction": met.get("fundamental_power_fraction", float("nan")),
            "total_harmonic_power_dbm": met.get("total_harmonic_power_dbm", float("nan")),
        })
    return pd.DataFrame(rows)


def _dbm_to_watts(dbm: float) -> float:
    return 1e-3 * (10.0 ** (dbm / 10.0))


def _watts_to_dbm(watts: float) -> float:
    if watts <= 0:
        return float("-inf")
    return 10.0 * math.log10(watts / 1e-3)
