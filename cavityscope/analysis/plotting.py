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


def plot_trace_frequency_space(
    t: np.ndarray,
    y_v: np.ndarray,
    ref: ReferenceInfo,
    rf_frequency_hz: float,
    title: str,
    out_png: Path | str,
    cfg: SweepConfig,
    picked_points: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Plot a scope trace converted to frequency space relative to the carrier.

    The time axis is mapped to frequency offset from the chosen carrier via
    ``delta_f = (t - carrier_t) / fsr_time_s * cavity_fsr_hz``.
    """
    _baseline, y_bs, y_sm = _smooth_and_baseline(y_v, cfg)
    fsr_hz = cfg.cavity_fsr_hz
    carrier_t = ref.chosen_carrier_time_s

    f_rel = (t - carrier_t) / ref.fsr_time_s * fsr_hz
    f_rel_ghz = f_rel / 1e9

    carrier_half_hz = 0.5 * cfg.carrier_window_hz
    sideband_half_hz = 0.5 * cfg.sideband_window_hz

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(f_rel_ghz, y_bs, lw=0.8, alpha=0.5, label="baseline-subtracted")
    ax.plot(f_rel_ghz, y_sm, lw=1.0, label="smoothed")

    ax.axvspan(
        -carrier_half_hz / 1e9,
        carrier_half_hz / 1e9,
        color="C0",
        alpha=0.12,
        label="carrier window",
    )
    if rf_frequency_hz > 0:
        sb_ghz = rf_frequency_hz / 1e9
        ax.axvspan(
            -sb_ghz - sideband_half_hz / 1e9,
            -sb_ghz + sideband_half_hz / 1e9,
            color="C1",
            alpha=0.12,
            label="\u22121 sb window",
        )
        ax.axvspan(
            sb_ghz - sideband_half_hz / 1e9,
            sb_ghz + sideband_half_hz / 1e9,
            color="C2",
            alpha=0.12,
            label="+1 sb window",
        )
        ax.axvline(-sb_ghz, ls=":", lw=0.8, color="C1", alpha=0.7)
        ax.axvline(sb_ghz, ls=":", lw=0.8, color="C2", alpha=0.7)

    ax.axvline(0, ls="--", lw=1.0, color="k", alpha=0.4, label="carrier")

    # Mark FSR boundaries for context
    ax.axvline(-fsr_hz / 2 / 1e9, ls="-", lw=0.6, color="gray", alpha=0.35)
    ax.axvline(fsr_hz / 2 / 1e9, ls="-", lw=0.6, color="gray", alpha=0.35, label="FSR/2")

    if picked_points is not None:
        for tx_key, hy_key, marker, lbl in [
            ("carrier_times_s", "carrier_heights_v", "o", "carrier"),
            ("sb_minus_times_s", "sb_minus_heights_v", "s", "\u22121 sb"),
            ("sb_plus_times_s", "sb_plus_heights_v", "^", "+1 sb"),
        ]:
            tx = np.asarray(picked_points.get(tx_key, []), dtype=float)
            hy = np.asarray(picked_points.get(hy_key, []), dtype=float)
            if tx.size:
                fx = (tx - carrier_t) / ref.fsr_time_s * fsr_hz / 1e9
                ax.scatter(fx, hy, marker=marker, s=55, zorder=5, label=lbl)

    ax.set(
        xlabel="Frequency offset from carrier (GHz)",
        ylabel="Voltage above baseline (V)",
        title=title,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
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


def plot_vpi_vs_frequency(
    fit_df: pd.DataFrame,
    out_png: Path | str,
) -> None:
    """Plot Vpi and fit R squared as a function of RF frequency.

    The top panel shows Vpi(f) with a median reference line and the number
    of fit points annotated at each frequency. The bottom panel shows the
    fit R squared so you can spot unreliable fits at a glance.
    """
    if fit_df.empty or "fit_vpi_v" not in fit_df.columns:
        return

    freq_ghz = fit_df["rf_frequency_hz"].to_numpy(dtype=float) / 1e9
    vpi = fit_df["fit_vpi_v"].to_numpy(dtype=float)
    r2 = fit_df["fit_r2"].to_numpy(dtype=float)
    n_pts = fit_df["n_fit_points"].to_numpy(dtype=float)

    valid = np.isfinite(vpi)
    if not valid.any():
        return

    fig, (ax_vpi, ax_r2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax_vpi.plot(
        freq_ghz[valid], vpi[valid], "o-", markersize=7, lw=1.5, label="$V_\\pi$",
    )
    for f, v, n in zip(freq_ghz[valid], vpi[valid], n_pts[valid]):
        ax_vpi.annotate(
            f"N={int(n)}", (f, v),
            textcoords="offset points", xytext=(0, 8),
            fontsize=7, ha="center", alpha=0.7,
        )
    vpi_median = float(np.nanmedian(vpi[valid]))
    ax_vpi.axhline(
        vpi_median, ls="--", lw=0.8, color="gray", alpha=0.5,
        label=f"median = {vpi_median:.3f} V",
    )
    ax_vpi.set_ylabel("$V_\\pi$ (V)")
    ax_vpi.set_title("$V_\\pi$ vs RF frequency")
    ax_vpi.grid(True, alpha=0.25)
    ax_vpi.legend(loc="best")

    bar_width = 0.8 * float(np.min(np.diff(freq_ghz))) if len(freq_ghz) > 1 else 0.1
    ax_r2.bar(freq_ghz[valid], r2[valid], width=bar_width, alpha=0.7)
    ax_r2.set_ylim(0, 1.05)
    ax_r2.axhline(0.99, ls=":", lw=0.8, color="gray", alpha=0.5, label="R\u00b2 = 0.99")
    ax_r2.set(xlabel="RF frequency (GHz)", ylabel="Fit R\u00b2")
    ax_r2.grid(True, alpha=0.25)
    ax_r2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
