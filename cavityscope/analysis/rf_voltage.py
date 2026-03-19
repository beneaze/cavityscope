"""Extract RF voltage from a scope trace (live calibration channel)."""

from __future__ import annotations

import math

import numpy as np


def extract_vpk_from_trace(
    t: np.ndarray,
    v: np.ndarray,
    rf_frequency_hz: float,
    cycles_for_rms: int = 20,
) -> dict:
    """Measure the RF peak voltage from a scope trace.

    Uses an RMS approach: compute Vrms over an integer number of RF cycles
    near the centre of the trace, then Vpk = sqrt(2) * Vrms.  This is more
    robust than raw min/max against noise and scope quantisation.

    Parameters
    ----------
    t, v : array
        Time (s) and voltage (V) arrays from the scope.
    rf_frequency_hz : float
        Expected RF frequency (used to select the analysis window).
    cycles_for_rms : int
        How many RF cycles to include in the RMS window.

    Returns
    -------
    dict with ``measured_vpk_v``, ``measured_vrms_v``, ``measured_vpp_v``,
    and diagnostic fields.
    """
    dt = float(np.median(np.diff(t)))
    rf_period = 1.0 / rf_frequency_hz
    window_duration = cycles_for_rms * rf_period
    window_samples = max(2, int(round(window_duration / dt)))

    mid = len(v) // 2
    half_win = window_samples // 2
    i0 = max(0, mid - half_win)
    i1 = min(len(v), mid + half_win)
    v_win = v[i0:i1]

    dc_offset = float(np.mean(v_win))
    v_ac = v_win - dc_offset
    vrms = float(np.sqrt(np.mean(v_ac ** 2)))
    vpk = math.sqrt(2.0) * vrms
    vpp = float(np.max(v_win) - np.min(v_win))

    return {
        "measured_vpk_v": vpk,
        "measured_vrms_v": vrms,
        "measured_vpp_v": vpp,
        "measured_dc_offset_v": dc_offset,
        "rms_window_samples": int(i1 - i0),
        "rms_window_duration_s": float((i1 - i0) * dt),
    }
