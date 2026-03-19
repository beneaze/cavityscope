"""Extract RF voltage amplitude from a scope trace via sine fit."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import curve_fit


def _sine_model(t: np.ndarray, amp: float, freq: float, phase: float, offset: float) -> np.ndarray:
    return amp * np.sin(2.0 * np.pi * freq * t + phase) + offset


def extract_vpk_from_trace(
    t: np.ndarray,
    v: np.ndarray,
    rf_frequency_hz: float,
    n_cycles: int = 20,
) -> dict:
    """Measure the RF peak voltage by fitting a sine to the scope trace.

    A windowed section (~*n_cycles* RF periods centred on the trace) is
    selected and ``A sin(2 pi f t + phi) + offset`` is fitted.  The fit
    amplitude gives Vpk directly — more robust than RMS against noise,
    DC offsets, and non-integer cycle counts.

    Parameters
    ----------
    t, v : array
        Time (s) and voltage (V) from the scope.
    rf_frequency_hz : float
        Nominal RF frequency (used as the fit starting value and to
        choose the analysis window width).
    n_cycles : int
        Approximate number of RF cycles to include in the fit window.

    Returns
    -------
    dict with ``measured_vpk_v``, ``measured_vrms_v``, ``measured_vpp_v``,
    ``fit_frequency_hz``, ``fit_phase_rad``, ``fit_dc_offset_v``, and
    diagnostic fields.
    """
    dt = float(np.median(np.diff(t)))
    rf_period = 1.0 / rf_frequency_hz
    window_duration = n_cycles * rf_period
    window_samples = max(4, int(round(window_duration / dt)))

    mid = len(v) // 2
    half_win = window_samples // 2
    i0 = max(0, mid - half_win)
    i1 = min(len(v), mid + half_win)
    t_win = t[i0:i1]
    v_win = v[i0:i1]

    amp_guess = 0.5 * float(np.max(v_win) - np.min(v_win))
    offset_guess = float(np.mean(v_win))

    try:
        popt, _ = curve_fit(
            _sine_model,
            t_win,
            v_win,
            p0=[amp_guess, rf_frequency_hz, 0.0, offset_guess],
            bounds=(
                [0.0, rf_frequency_hz * 0.8, -np.pi, -np.inf],
                [np.inf, rf_frequency_hz * 1.2, np.pi, np.inf],
            ),
            maxfev=5000,
        )
        vpk = abs(float(popt[0]))
        fit_freq = float(popt[1])
        fit_phase = float(popt[2])
        fit_offset = float(popt[3])
        fit_ok = True
    except Exception:
        vpk = amp_guess
        fit_freq = rf_frequency_hz
        fit_phase = float("nan")
        fit_offset = offset_guess
        fit_ok = False

    vrms = vpk / math.sqrt(2.0)
    vpp = 2.0 * vpk

    return {
        "measured_vpk_v": vpk,
        "measured_vrms_v": vrms,
        "measured_vpp_v": vpp,
        "fit_frequency_hz": fit_freq,
        "fit_phase_rad": fit_phase,
        "fit_dc_offset_v": fit_offset,
        "fit_converged": fit_ok,
        "fit_window_samples": int(i1 - i0),
        "fit_window_duration_s": float((i1 - i0) * dt),
    }
