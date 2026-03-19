"""Post-hoc (re-)analysis of sweep and calibration data.

All functions in this module are pure analysis — they operate on DataFrames
and configuration only, with no hardware dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from cavityscope.analysis.measurement import add_voltage_columns
from cavityscope.analysis.plotting import (
    plot_beta_fit,
    plot_power_calibration,
    plot_vpi_vs_frequency,
)
from cavityscope.analysis.vpi_fitting import fit_beta_vs_vpk
from cavityscope.core.calibration import PowerCalibration
from cavityscope.core.config import SweepConfig
from cavityscope.core.utils import ensure_dir


# ---------------------------------------------------------------------------
# Power calibration analysis
# ---------------------------------------------------------------------------

def build_calibration(
    cal_df: pd.DataFrame,
    output_dir: Optional[str | Path] = None,
    verbose: bool = True,
) -> PowerCalibration:
    """Build a :class:`PowerCalibration` from a raw calibration DataFrame.

    Optionally saves the CSV, a summary plot, and prints diagnostics.

    Parameters
    ----------
    cal_df : pd.DataFrame
        Must contain ``frequency_hz``, ``power_dbm``, ``vpk_v``
        (as produced by :func:`~cavityscope.sweep.run_power_calibration`).
    output_dir : str or Path, optional
        If given, saves ``power_calibration.csv`` and
        ``power_calibration.png`` there.
    verbose : bool
        Print summary.

    Returns
    -------
    PowerCalibration
    """
    calibration = PowerCalibration(cal_df)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cal_df.to_csv(out / "power_calibration.csv", index=False)
        plot_power_calibration(cal_df, out / "power_calibration.png")
        if verbose:
            print(f"  Saved: {out / 'power_calibration.csv'}")
            print(f"  Saved: {out / 'power_calibration.png'}")

    if verbose:
        print(f"  {calibration}")

    return calibration


def load_calibration_run(
    run_dir: str | Path,
    verbose: bool = True,
) -> Dict:
    """Load a previously saved calibration run from disk.

    Parameters
    ----------
    run_dir : str or Path
        Path to a ``calibration_YYYYMMDD_HHMMSS/`` folder.
    verbose : bool
        Print summary.

    Returns
    -------
    dict with keys ``"cal_df"``, ``"calibration"``, ``"config"`` (if present)

    Example
    -------
    >>> data = load_calibration_run("vpi_sweep_output/calibration_20260319_120000")
    >>> cal  = data["calibration"]   # PowerCalibration ready to use
    >>> df   = data["cal_df"]        # raw DataFrame for inspection / replotting
    """
    run_dir = Path(run_dir)
    csv_path = run_dir / "power_calibration.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No power_calibration.csv in {run_dir}")

    cal_df = pd.read_csv(csv_path)
    calibration = PowerCalibration(cal_df)

    cfg_dict = None
    cfg_path = run_dir / "config_used.json"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg_dict = json.load(f)

    if verbose:
        print(f"Loaded calibration from: {run_dir}")
        print(f"  {calibration}")
        if cfg_dict:
            freqs = cfg_dict.get("rf_frequencies_hz", [])
            powers = cfg_dict.get("rf_powers_dbm", [])
            print(f"  Grid: {len(freqs)} frequencies x {len(powers)} powers")

    return {"cal_df": cal_df, "calibration": calibration, "config": cfg_dict}


# ---------------------------------------------------------------------------
# Sweep reanalysis
# ---------------------------------------------------------------------------

def apply_calibration(
    results_df: pd.DataFrame,
    cfg: SweepConfig,
    calibration: Optional[PowerCalibration] = None,
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
    calibration: Optional[PowerCalibration] = None,
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
        calibration = PowerCalibration.from_csv(cfg.power_calibration_csv)

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
