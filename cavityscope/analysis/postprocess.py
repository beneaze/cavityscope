"""Post-hoc (re-)analysis of sweep results: voltage calibration and Vpi fitting.

All functions in this module are pure analysis — they operate on DataFrames
and configuration only, with no hardware dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

from cavityscope.analysis.measurement import add_voltage_columns
from cavityscope.analysis.plotting import plot_beta_fit, plot_vpi_vs_frequency
from cavityscope.analysis.vpi_fitting import fit_beta_vs_vpk
from cavityscope.core.config import SweepConfig
from cavityscope.core.utils import ensure_dir

if TYPE_CHECKING:
    from cavityscope.core.calibration import PowerCalibration


def apply_calibration(
    results_df: pd.DataFrame,
    cfg: SweepConfig,
    calibration: Optional["PowerCalibration"] = None,
) -> pd.DataFrame:
    """Recompute voltage columns on every row using a (new) calibration.

    The raw measurement columns (beta, areas, SNR, …) are preserved.
    Only the ``estimated_*`` voltage columns and ``voltage_source`` are
    replaced.

    Parameters
    ----------
    results_df : pd.DataFrame
        Sweep results (from CSV or ``run_sweep``).
    cfg : SweepConfig
        Sweep configuration.
    calibration : PowerCalibration, optional
        Calibration to apply.  ``None`` → analytical dBm-to-V.
    """
    voltage_cols = [
        "estimated_delivered_dbm",
        "estimated_vrms_at_load",
        "estimated_vpk_at_load",
        "estimated_vpp_at_load",
        "voltage_source",
    ]
    df = results_df.copy()
    df.drop(columns=[c for c in voltage_cols if c in df.columns], inplace=True)

    new_rows = []
    for _, row in df.iterrows():
        r = row.to_dict()
        r = add_voltage_columns(
            r,
            r["rf_power_dbm"],
            cfg,
            calibration=calibration,
            rf_frequency_hz=r["rf_frequency_hz"],
        )
        new_rows.append(r)
    return pd.DataFrame(new_rows)


def compute_vpi_fits(
    results_df: pd.DataFrame,
    cfg: SweepConfig,
    output_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Fit Vpi at each RF frequency and optionally save plots.

    Parameters
    ----------
    results_df : pd.DataFrame
        Sweep results with ``estimated_vpk_at_load`` already populated.
    cfg : SweepConfig
        Sweep configuration.
    output_dir : str or Path, optional
        If given, saves per-frequency fit plots and a Vpi-vs-frequency
        summary plot.

    Returns
    -------
    pd.DataFrame
        One row per RF frequency with fit results.
    """
    if results_df.empty or not cfg.compute_vpi:
        return pd.DataFrame()

    fit_rows: List[Dict] = []
    fit_plot_dir = ensure_dir(Path(output_dir) / "fit_plots") if output_dir else None

    for fhz, dfg in results_df.groupby("rf_frequency_hz"):
        fit_row = {"rf_frequency_hz": float(fhz), **fit_beta_vs_vpk(dfg, cfg)}
        fit_rows.append(fit_row)
        if fit_plot_dir:
            plot_beta_fit(
                dfg,
                fit_row,
                fit_plot_dir / f"vpi_fit_{fhz/1e6:.4f}MHz.png",
                float(fhz),
                cfg,
            )

    fit_df = pd.DataFrame(fit_rows)
    if output_dir:
        plot_vpi_vs_frequency(fit_df, Path(output_dir) / "vpi_vs_frequency.png")
    return fit_df


def reanalyze_with_calibration(
    results_df: pd.DataFrame,
    cfg: SweepConfig,
    calibration: Optional["PowerCalibration"] = None,
    output_dir: Optional[str | Path] = None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Recompute voltage columns and Vpi fits on previously recorded data.

    This lets you apply (or change) a power calibration after the sweep
    has already been recorded.  Only the voltage mapping and the Vpi fits
    are recomputed — the raw beta / area measurements are kept as-is.

    Parameters
    ----------
    results_df : pd.DataFrame
        The ``sweep_results.csv`` loaded as a DataFrame (or ``data["results"]``
        returned by :func:`~cavityscope.sweep.run_sweep`).
    cfg : SweepConfig
        Sweep configuration (controls fit parameters, load impedance, etc.).
    calibration : PowerCalibration, optional
        New calibration to apply.  ``None`` falls back to analytical dBm→V.
        Can also be loaded automatically from ``cfg.power_calibration_csv``.
    output_dir : str or Path, optional
        If given, saves updated CSVs and Vpi fit / summary plots there.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys ``"results"``, ``"fits"``
    """
    if calibration is None and cfg.power_calibration_csv is not None:
        from cavityscope.core.calibration import PowerCalibration as _PC
        calibration = _PC.from_csv(cfg.power_calibration_csv)

    df = apply_calibration(results_df, cfg, calibration=calibration)
    fit_df = compute_vpi_fits(df, cfg, output_dir=output_dir)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "sweep_results.csv", index=False)
        fit_df.to_csv(out / "vpi_fit_summary.csv", index=False)
        if verbose:
            print(f"Saved reanalyzed results to: {out}")

    if verbose:
        src = df["voltage_source"].iloc[0] if "voltage_source" in df.columns else "?"
        print(f"Voltage source: {src}")
        if not fit_df.empty and "fit_vpi_v" in fit_df.columns:
            valid = np.isfinite(fit_df["fit_vpi_v"])
            if valid.any():
                best = fit_df.loc[fit_df["fit_vpi_v"][valid].idxmin()]
                print(
                    f"Best Vpi: {best['fit_vpi_v']:.3f} V "
                    f"@ {best['rf_frequency_hz']/1e9:.6f} GHz"
                )

    return {"results": df, "fits": fit_df}
