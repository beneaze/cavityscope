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
    """Plot beta vs Vpk with the linear fit overlaid.

    When ``beta_unwrapped`` is present the plot shows both the raw
    Branch-1 values (faded) and the unwrapped values (solid).
    """
    has_unwrapped = "beta_unwrapped" in dfg.columns
    beta_col = "beta_unwrapped" if has_unwrapped else "beta_est"

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    used = dfg[dfg["used_for_vpi_fit"]].copy()
    rejected = dfg[~dfg["used_for_vpi_fit"]].copy()

    if has_unwrapped:
        ax.scatter(
            dfg["estimated_vpk_at_load"],
            dfg["beta_est"],
            s=30,
            marker=".",
            color="0.65",
            alpha=0.5,
            label="raw (branch 1)",
            zorder=2,
        )

    if not rejected.empty:
        ax.scatter(
            rejected["estimated_vpk_at_load"],
            rejected[beta_col],
            s=55,
            marker="x",
            label="rejected",
            zorder=3,
        )
    if not used.empty:
        ax.scatter(
            used["estimated_vpk_at_load"],
            used[beta_col],
            s=65,
            label="used" + (" (unwrapped)" if has_unwrapped else ""),
            zorder=4,
        )

    slope = fit_row.get("fit_slope_beta_per_v", float("nan"))
    intercept = fit_row.get("fit_intercept_beta", float("nan"))
    if np.isfinite(slope) and np.isfinite(intercept) and len(used) >= 2:
        xmax = max(float(np.max(dfg["estimated_vpk_at_load"])), 1e-9)
        xfit = np.linspace(0.0, 1.05 * xmax, 200)
        ax.plot(xfit, slope * xfit + intercept, label="fit", zorder=5)

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


def _add_line_labels(ax: plt.Axes, lines, labels, fontsize: float = 7) -> None:
    """Place labels at the right end of each line, outside the plot area."""
    for line, label in zip(lines, labels):
        xdata, ydata = line.get_xdata(), line.get_ydata()
        if len(xdata) == 0:
            continue
        ax.annotate(
            label,
            xy=(xdata[-1], ydata[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=fontsize,
            color=line.get_color(),
            va="center",
            ha="left",
            clip_on=False,
        )


def plot_power_calibration(
    cal_df: pd.DataFrame,
    out_png: Path | str,
) -> None:
    """Plot power calibration (works for both scope-based and SA-based data).

    If ``measured_power_dbm`` is present (SA calibration), shows three panels:
    measured dBm, Vpk vs power setting, and Vpk vs frequency.
    Otherwise shows two panels (Vpk only).

    Frequency labels are placed next to each line on the right side of the
    plot instead of inside a legend, keeping the plot area clean.
    """
    if cal_df.empty:
        return

    col_freq = "frequency_hz" if "frequency_hz" in cal_df.columns else "rf_frequency_hz"
    col_pwr = "power_dbm" if "power_dbm" in cal_df.columns else "rf_power_dbm"
    col_vpk = "vpk_v" if "vpk_v" in cal_df.columns else "measured_vpk_v"

    if col_vpk not in cal_df.columns:
        return

    freqs = sorted(cal_df[col_freq].unique())
    powers = sorted(cal_df[col_pwr].unique())
    has_sa = "measured_power_dbm" in cal_df.columns
    sa_label = "SA" if has_sa else "Scope"
    freq_labels = [f"{f/1e9:.4f}" for f in freqs]

    n_panels = 3 if has_sa else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.8 * n_panels))
    ax_idx = 0

    if has_sa:
        ax_dbm = axes[ax_idx]; ax_idx += 1
        lines = []
        for freq in freqs:
            sub = cal_df[cal_df[col_freq] == freq].sort_values(col_pwr)
            ln, = ax_dbm.plot(
                sub[col_pwr], sub["measured_power_dbm"],
                "o-", markersize=3, lw=1.0,
            )
            lines.append(ln)
        pmin, pmax = min(powers), max(powers)
        ax_dbm.plot([pmin, pmax], [pmin, pmax], "k--", lw=0.8, alpha=0.35)
        ax_dbm.set_xlabel("RF power setting (dBm)", fontsize=8)
        ax_dbm.set_ylabel("Measured power (dBm)", fontsize=8)
        ax_dbm.set_title(f"{sa_label} calibration — measured power vs setting", fontsize=9)
        ax_dbm.grid(True, alpha=0.25)
        _add_line_labels(ax_dbm, lines, freq_labels)
        ax_dbm.margins(x=0.02)
        extra_right = 0.06 + 0.008 * max(len(s) for s in freq_labels)
        ax_dbm.set_xlim(right=pmax + (pmax - pmin) * extra_right)

    ax_pwr = axes[ax_idx]; ax_idx += 1
    lines = []
    for freq in freqs:
        sub = cal_df[cal_df[col_freq] == freq].sort_values(col_pwr)
        ln, = ax_pwr.plot(
            sub[col_pwr], sub[col_vpk],
            "o-", markersize=3, lw=1.0,
        )
        lines.append(ln)
    ax_pwr.set_xlabel("RF power setting (dBm)", fontsize=8)
    ax_pwr.set_ylabel("$V_{pk}$ at load (V)", fontsize=8)
    ax_pwr.set_title(f"{sa_label} calibration — $V_{{pk}}$ vs power setting", fontsize=9)
    ax_pwr.grid(True, alpha=0.25)
    _add_line_labels(ax_pwr, lines, freq_labels)
    pmin, pmax = min(powers), max(powers)
    extra_right = 0.06 + 0.008 * max(len(s) for s in freq_labels)
    ax_pwr.set_xlim(right=pmax + (pmax - pmin) * extra_right)

    ax_freq = axes[ax_idx]
    max_power = max(powers)
    at_max = cal_df[cal_df[col_pwr] == max_power].sort_values(col_freq)
    if not at_max.empty:
        ax_freq.plot(
            at_max[col_freq] / 1e9, at_max[col_vpk],
            "s-", markersize=6, lw=1.5, color="C0",
        )
        ax_freq.set_xlabel("RF frequency (GHz)", fontsize=8)
        ax_freq.set_ylabel("$V_{pk}$ at load (V)", fontsize=8)
        ax_freq.set_title(
            f"$V_{{pk}}$ vs frequency at max power ({max_power:.1f} dBm)",
            fontsize=9,
        )
        ax_freq.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# Keep the old name as an alias
plot_live_calibration = plot_power_calibration


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
    idx_min = int(np.nanargmin(vpi[valid]))
    f_min, v_min = freq_ghz[valid][idx_min], vpi[valid][idx_min]
    ax_vpi.axvline(f_min, ls=":", lw=0.8, color="tab:red", alpha=0.6)
    ax_vpi.annotate(
        f"min $V_\\pi$ = {v_min:.3f} V\n@ {f_min*1e3:.2f} MHz",
        (f_min, v_min),
        textcoords="offset points", xytext=(12, -18),
        fontsize=8, color="tab:red", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.2),
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


def plot_sa_spectrum(
    wideband_freqs: np.ndarray,
    wideband_amps: np.ndarray,
    harmonics: list,
    fundamental_hz: float,
    power_dbm_setting: float,
    metrics: dict,
    out_png: Path | str,
    power_offset_db: float = 0.0,
) -> None:
    """Plot a single wideband spectrum with fundamental and harmonics labelled.

    Shows the full SA trace, marks each measured harmonic with its power
    and dBc level, and annotates the THD.

    Parameters
    ----------
    power_offset_db : float
        Offset added to raw SA readings for display (e.g. +30 for a 30 dB
        external attenuator).  The trace and markers are shifted together
        so relative metrics (dBc, THD) are unaffected.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))

    display_amps = wideband_amps + power_offset_db
    ax.plot(wideband_freqs / 1e9, display_amps,
            color="C0", lw=0.8, alpha=0.85, label="SA trace")

    colours = plt.cm.tab10.colors
    fund_dbm = metrics.get("fundamental_power_dbm", float("nan"))

    for h in harmonics:
        k = h["harmonic_number"]
        f_ghz = h["measured_freq_hz"] / 1e9
        p_dbm = h["power_dbm"] + power_offset_db
        dbc = h["power_dbm"] - fund_dbm if np.isfinite(fund_dbm) else float("nan")

        c = colours[(k - 1) % len(colours)]
        lbl = "f (fund.)" if k == 1 else f"{k}f"
        ax.axvline(f_ghz, ls="--", lw=0.9, color=c, alpha=0.6)
        ax.plot(f_ghz, p_dbm, "o", ms=8, color=c, zorder=5)

        if k == 1:
            txt = f"{lbl}\n{p_dbm:+.1f} dBm"
        else:
            txt = f"{lbl}\n{p_dbm:+.1f} dBm\n({dbc:+.1f} dBc)"
        ax.annotate(txt, (f_ghz, p_dbm),
                    textcoords="offset points", xytext=(6, 8),
                    fontsize=7, color=c, fontweight="bold",
                    ha="left", va="bottom")

    thd = metrics.get("thd_percent", float("nan"))
    frac = metrics.get("fundamental_power_fraction", float("nan"))
    offset_note = f"  [offset {power_offset_db:+.0f} dB]" if power_offset_db != 0 else ""
    ax.set_title(
        f"Spectrum: f₀ = {fundamental_hz/1e9:.4f} GHz, "
        f"P_set = {power_dbm_setting:+.1f} dBm  —  "
        f"THD = {thd:.1f}%,  fundamental carries {frac*100:.1f}% of total power"
        + offset_note,
        fontsize=9,
    )
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Power (dBm)" + (f" [+{power_offset_db:.0f} dB offset]" if power_offset_db != 0 else ""))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_harmonic_waterfall(
    harmonics_df: pd.DataFrame,
    freq_hz: float,
    out_png: Path | str,
) -> None:
    """Bar chart of harmonic power vs. power setting for one frequency.

    Each cluster of bars represents one power setting; bars within the
    cluster show the fundamental and each harmonic.
    """
    sub = harmonics_df[
        np.isclose(harmonics_df["frequency_hz"], freq_hz)
    ].copy()
    if sub.empty:
        return

    powers = sorted(sub["power_dbm"].unique())
    harm_nums = sorted(sub["harmonic_number"].unique())
    n_harms = len(harm_nums)
    bar_w = 0.8 / n_harms

    fig, ax = plt.subplots(figsize=(max(8, len(powers) * 0.6), 5))
    colours = plt.cm.tab10.colors

    for i, k in enumerate(harm_nums):
        hs = sub[sub["harmonic_number"] == k].set_index("power_dbm")
        x = np.arange(len(powers))
        vals = [float(hs.loc[p, "harmonic_power_dbm"]) if p in hs.index else float("nan")
                for p in powers]
        lbl = "fundamental" if k == 1 else f"{k}f"
        ax.bar(x + i * bar_w, vals, width=bar_w,
               color=colours[(k - 1) % len(colours)], label=lbl, alpha=0.85)

    ax.set_xticks(np.arange(len(powers)) + 0.4)
    ax.set_xticklabels([f"{p:+.0f}" for p in powers], fontsize=7)
    ax.set_xlabel("Power setting (dBm)")
    ax.set_ylabel("Measured power (dBm)")
    ax.set_title(f"Harmonic content vs. power — f₀ = {freq_hz/1e9:.4f} GHz")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_thd_summary(
    thd_df: pd.DataFrame,
    out_png: Path | str,
) -> None:
    """Plot THD vs. power setting, one curve per frequency.

    Top: THD (%).  Bottom: fundamental power fraction.
    """
    if thd_df.empty:
        return

    freqs = sorted(thd_df["frequency_hz"].unique())

    fig, (ax_thd, ax_frac) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for freq in freqs:
        s = thd_df[thd_df["frequency_hz"] == freq].sort_values("power_dbm")
        label = f"{freq/1e9:.4f} GHz"
        ax_thd.plot(s["power_dbm"], s["thd_percent"],
                    "o-", ms=4, lw=1.2, label=label)
        ax_frac.plot(s["power_dbm"], s["fundamental_power_fraction"] * 100,
                     "o-", ms=4, lw=1.2, label=label)

    ax_thd.set_ylabel("THD (%)")
    ax_thd.set_title("Total harmonic distortion vs. power setting")
    ax_thd.grid(True, alpha=0.25)
    ax_thd.legend(fontsize=7, ncol=max(1, len(freqs) // 5))

    ax_frac.set_xlabel("Power setting (dBm)")
    ax_frac.set_ylabel("Power in fundamental (%)")
    ax_frac.set_title("Fraction of total power in the fundamental")
    ax_frac.grid(True, alpha=0.25)
    ax_frac.axhline(100, ls=":", lw=0.8, color="gray", alpha=0.4)
    ax_frac.legend(fontsize=7, ncol=max(1, len(freqs) // 5))

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_harmonic_heatmap(
    harmonics_df: pd.DataFrame,
    out_png: Path | str,
) -> None:
    """Heatmap of dBc levels: (power setting) × (harmonic number).

    Averages across frequencies.  Gives a quick overview of which
    harmonics dominate and at what power level.
    """
    sub = harmonics_df[harmonics_df["harmonic_number"] >= 2].copy()
    if sub.empty:
        return

    pivot = sub.pivot_table(
        index="power_dbm", columns="harmonic_number",
        values="level_dbc", aggfunc="mean",
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.2), max(4, len(pivot) * 0.35)))
    im = ax.imshow(
        pivot.values, aspect="auto", cmap="RdYlGn",
        vmin=-80, vmax=0,
        origin="lower",
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(k)}f" for k in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{p:+.0f}" for p in pivot.index], fontsize=7)
    ax.set_xlabel("Harmonic")
    ax.set_ylabel("Power setting (dBm)")
    ax.set_title("Harmonic levels (dBc, averaged across frequencies)")

    for (i, j), val in np.ndenumerate(pivot.values):
        if np.isfinite(val):
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if val > -20 else "black")

    fig.colorbar(im, ax=ax, label="dBc", shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_calibration_fit(
    t: np.ndarray,
    v: np.ndarray,
    rf_frequency_hz: float,
    meas: dict,
    out_png: Path | str,
    n_cycles: int = 20,
) -> None:
    """Plot a calibration sine fit overlaid on the scope trace.

    Shows the analysis window, raw data, fitted sine, and residuals.
    """
    dt = float(np.median(np.diff(t)))
    rf_period = 1.0 / rf_frequency_hz
    window_samples = max(4, int(round(n_cycles * rf_period / dt)))
    mid = len(v) // 2
    half_win = window_samples // 2
    i0 = max(0, mid - half_win)
    i1 = min(len(v), mid + half_win)
    t_win = t[i0:i1]
    v_win = v[i0:i1]

    vpk = meas["measured_vpk_v"]
    f_fit = meas["fit_frequency_hz"]
    phase = meas.get("fit_phase_rad", 0.0)
    offset = meas.get("fit_dc_offset_v", 0.0)
    converged = meas.get("fit_converged", True)
    power_dbm = meas.get("power_dbm", float("nan"))

    if not np.isfinite(phase):
        phase = 0.0
    if not np.isfinite(offset):
        offset = float(np.mean(v_win))

    # Reconstruct fit on the same centred time axis used by the fitter
    t0 = meas.get("fit_t0_s", float(t_win[0] + t_win[-1]) / 2.0)
    t_rel = t_win - t0
    fit_curve = vpk * np.sin(2.0 * np.pi * f_fit * t_rel + phase) + offset
    residual = v_win - fit_curve

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_top.plot(t_win * 1e9, v_win, color="C0", lw=0.6, alpha=0.7, label="data")
    ax_top.plot(t_win * 1e9, fit_curve, color="tab:red", lw=1.2,
                label=f"fit: Vpk={vpk:.4f} V, f={f_fit/1e6:.2f} MHz")
    status = "converged" if converged else "FALLBACK"
    ax_top.set_title(
        f"Cal fit \u2014 f_RF={rf_frequency_hz/1e9:.4f} GHz, "
        f"P_RF={power_dbm:+.1f} dBm  [{status}]",
        fontsize=10,
    )
    ax_top.set_ylabel("Voltage (V)")
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.grid(True, alpha=0.2)

    ax_bot.plot(t_win * 1e9, residual * 1e3, color="C2", lw=0.5)
    ax_bot.axhline(0, ls="-", lw=0.5, color="gray")
    ax_bot.set_xlabel("Time (ns)")
    ax_bot.set_ylabel("Residual (mV)")
    ax_bot.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
