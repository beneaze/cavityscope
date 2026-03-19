"""Plotting helpers for cavity sweep traces and Vpi fits."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from cavityscope.analysis.reference import ReferenceInfo
from cavityscope.core.config import SweepConfig
from cavityscope.core.utils import hz_window_to_half_time, robust_baseline


def _smooth_and_baseline(y_v: np.ndarray, cfg: SweepConfig):
    baseline = robust_baseline(y_v, cfg.baseline_percentile)
    y_bs = y_v - baseline
    sigma = cfg.analysis_use_gaussian_sigma_pts
    y_sm = gaussian_filter1d(y_bs, sigma=sigma) if sigma > 0 else y_bs.copy()
    return baseline, y_bs, y_sm


def plot_trace_with_windows(
    t: np.ndarray,
    y_v: np.ndarray,
    ref: ReferenceInfo,
    rf_frequency_hz: float,
    title: str,
    out_png: Path | str,
    cfg: SweepConfig,
    picked_points: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Plot a single scope trace with carrier and sideband integration windows."""
    _baseline, y_bs, y_sm = _smooth_and_baseline(y_v, cfg)
    fsr_hz = cfg.cavity_fsr_hz
    carrier_hw = hz_window_to_half_time(
        cfg.carrier_window_hz, fsr_hz, ref.fsr_time_s
    )
    sideband_hw = hz_window_to_half_time(
        cfg.sideband_window_hz, fsr_hz, ref.fsr_time_s
    )
    dt_sb = ref.fsr_time_s * (rf_frequency_hz / fsr_hz)
    carrier_t = ref.chosen_carrier_time_s
    sb_m_t = carrier_t - dt_sb
    sb_p_t = carrier_t + dt_sb

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(t * 1e3, y_bs, lw=1.0, label="baseline-subtracted")
    ax.plot(t * 1e3, y_sm, lw=1.0, label="smoothed")

    ax.axvspan(
        (carrier_t - carrier_hw) * 1e3,
        (carrier_t + carrier_hw) * 1e3,
        alpha=0.12,
        label="carrier window",
    )
    ax.axvspan(
        (sb_m_t - sideband_hw) * 1e3,
        (sb_m_t + sideband_hw) * 1e3,
        alpha=0.12,
        label="-1 sb window",
    )
    ax.axvspan(
        (sb_p_t - sideband_hw) * 1e3,
        (sb_p_t + sideband_hw) * 1e3,
        alpha=0.12,
        label="+1 sb window",
    )
    ax.axvline(
        carrier_t * 1e3, ls="--", lw=1.0, alpha=0.8, label="ref carrier"
    )

    if picked_points is not None:
        for tx_key, hy_key, marker, label in [
            ("carrier_times_s", "carrier_heights_v", "o", "carrier"),
            ("sb_minus_times_s", "sb_minus_heights_v", "s", "-1 sb"),
            ("sb_plus_times_s", "sb_plus_heights_v", "^", "+1 sb"),
        ]:
            tx = np.asarray(picked_points.get(tx_key, []), dtype=float)
            hy = np.asarray(picked_points.get(hy_key, []), dtype=float)
            if tx.size:
                ax.scatter(tx * 1e3, hy, marker=marker, s=55, zorder=5, label=label)

    ax.set(xlabel="Time (ms)", ylabel="Voltage above baseline (V)", title=title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_beta_fit(
    dfg: pd.DataFrame,
    fit_row: Dict[str, float],
    out_png: Path | str,
    freq_hz: float,
    cfg: SweepConfig,
) -> None:
    """Plot beta vs Vpk with the linear fit overlaid."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    used = dfg[dfg["used_for_vpi_fit"]].copy()
    rejected = dfg[~dfg["used_for_vpi_fit"]].copy()
    if not rejected.empty:
        ax.scatter(
            rejected["estimated_vpk_at_load"],
            rejected["beta_est"],
            s=55,
            marker="x",
            label="rejected",
        )
    if not used.empty:
        ax.scatter(
            used["estimated_vpk_at_load"],
            used["beta_est"],
            s=65,
            label="used",
        )

    slope = fit_row.get("fit_slope_beta_per_v", float("nan"))
    intercept = fit_row.get("fit_intercept_beta", float("nan"))
    if np.isfinite(slope) and np.isfinite(intercept) and len(used) >= 2:
        xmax = max(float(np.max(dfg["estimated_vpk_at_load"])), 1e-9)
        xfit = np.linspace(0.0, 1.05 * xmax, 200)
        ax.plot(xfit, slope * xfit + intercept, label="fit")

    title = (
        f"Beta vs Vpk, f_RF={freq_hz/1e9:.6f} GHz\n"
        f"Vpi={fit_row.get('fit_vpi_v', float('nan')):.4f} V, "
        f"intercept={fit_row.get('fit_intercept_beta', float('nan')):.4f}, "
        f"R\u00b2={fit_row.get('fit_r2', float('nan')):.4f}, "
        f"N={int(fit_row.get('n_fit_points', 0))}"
    )
    ax.set(
        title=title,
        xlabel="Nominal RF voltage at load, Vpk (V)",
        ylabel="Modulation index beta",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
