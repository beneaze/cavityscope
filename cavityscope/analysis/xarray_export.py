"""xarray / NetCDF export and import for sweep results.

Converts the flat CSV-style DataFrames into a structured xarray Dataset
where the natural sweep dimensions (rf_frequency_hz × rf_power_dbm) are
explicit coordinates.  Reference and fit data live on the frequency axis
only, and the full SweepConfig is stored as attributes.

Requires optional dependencies: ``pip install xarray netCDF4``
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import xarray as xr

    from cavityscope.core.config import SweepConfig


def _import_xarray():
    try:
        import xarray as xr

        return xr
    except ImportError:
        raise ImportError(
            "xarray is required for NetCDF export.  "
            "Install it with:  pip install xarray netCDF4"
        ) from None


# ---------------------------------------------------------------------------
# DataFrame → xarray Dataset
# ---------------------------------------------------------------------------

_STRING_COLS = frozenset({"sideband_mode_used", "voltage_source"})
_INDEX_COLS = frozenset({"rf_frequency_hz", "rf_power_dbm"})


def sweep_to_xarray(
    results_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    cfg: "SweepConfig",
    *,
    s21_df: Optional[pd.DataFrame] = None,
) -> "xr.Dataset":
    """Build an xarray Dataset from the DataFrames returned by ``run_sweep``.

    Parameters
    ----------
    results_df : DataFrame
        Main sweep results (one row per frequency × power point).
    ref_df : DataFrame
        Reference summary (one row per frequency).
    fit_df : DataFrame
        Vpi fit results (one row per frequency).
    cfg : SweepConfig
        Sweep configuration — stored as Dataset attributes.
    s21_df : DataFrame, optional
        S21 analysis results (same grid as *results_df*).

    Returns
    -------
    xr.Dataset
    """
    xr = _import_xarray()

    freqs = np.array(sorted(results_df["rf_frequency_hz"].unique()))
    powers = np.array(sorted(results_df["rf_power_dbm"].unique()))

    ds = xr.Dataset(
        coords={
            "rf_frequency_hz": freqs,
            "rf_power_dbm": powers,
        }
    )

    sweep_var_names: list[str] = []
    ref_var_names: list[str] = []
    fit_var_names: list[str] = []

    # -- 2D sweep variables (frequency × power) ----------------------------
    for col in results_df.columns:
        if col in _INDEX_COLS or col in _STRING_COLS:
            continue
        try:
            pivot = (
                results_df.pivot(
                    index="rf_frequency_hz",
                    columns="rf_power_dbm",
                    values=col,
                )
                .reindex(index=freqs, columns=powers)
            )
            ds[col] = (
                ["rf_frequency_hz", "rf_power_dbm"],
                pivot.values.astype(float),
            )
            sweep_var_names.append(col)
        except (ValueError, TypeError):
            continue

    for col in _STRING_COLS:
        if col in results_df.columns:
            vals = results_df[col].unique()
            if len(vals) == 1:
                ds.attrs[col] = str(vals[0])

    # -- S21 variables (same 2D grid) --------------------------------------
    s21_var_names: list[str] = []
    if s21_df is not None and not s21_df.empty:
        for col in s21_df.columns:
            if col in _INDEX_COLS or col in ds.data_vars:
                continue
            try:
                pivot = (
                    s21_df.pivot(
                        index="rf_frequency_hz",
                        columns="rf_power_dbm",
                        values=col,
                    )
                    .reindex(index=freqs, columns=powers)
                )
                ds[col] = (
                    ["rf_frequency_hz", "rf_power_dbm"],
                    pivot.values.astype(float),
                )
                s21_var_names.append(col)
            except (ValueError, TypeError):
                continue

    # -- 1D reference variables (frequency only) ---------------------------
    #    Prefixed with "ref_" to avoid collisions with 2D variables.
    if ref_df is not None and not ref_df.empty:
        ref_sorted = ref_df.set_index("rf_frequency_hz").reindex(freqs)
        for col in ref_sorted.columns:
            var_name = f"ref_{col}"
            try:
                ds[var_name] = (
                    "rf_frequency_hz",
                    ref_sorted[col].values.astype(float),
                )
                ref_var_names.append(var_name)
            except (ValueError, TypeError):
                continue

    # -- 1D fit variables (frequency only) ---------------------------------
    if fit_df is not None and not fit_df.empty:
        fit_sorted = fit_df.set_index("rf_frequency_hz").reindex(freqs)
        for col in fit_sorted.columns:
            try:
                ds[col] = (
                    "rf_frequency_hz",
                    fit_sorted[col].values.astype(float),
                )
                fit_var_names.append(col)
            except (ValueError, TypeError):
                continue

    # -- Group membership (makes the loader trivial) -----------------------
    ds.attrs["_sweep_vars"] = ",".join(sweep_var_names)
    ds.attrs["_ref_vars"] = ",".join(ref_var_names)
    ds.attrs["_fit_vars"] = ",".join(fit_var_names)
    ds.attrs["_s21_vars"] = ",".join(s21_var_names)

    # -- Config as attributes ----------------------------------------------
    _store_config_attrs(ds, cfg)

    return ds


def _store_config_attrs(ds: "xr.Dataset", cfg: "SweepConfig") -> None:
    """Serialize SweepConfig into Dataset attributes."""
    for k, v in cfg.to_dict().items():
        key = f"cfg_{k}"
        if isinstance(v, (list, np.ndarray)):
            try:
                ds.attrs[key] = np.asarray(v, dtype=float)
            except (ValueError, TypeError):
                ds.attrs[key] = str(v)
        elif v is None:
            ds.attrs[key] = "None"
        else:
            ds.attrs[key] = v


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_sweep_netcdf(
    results_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    cfg: "SweepConfig",
    path: str | Path,
    *,
    s21_df: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> None:
    """Convert sweep DataFrames to xarray and write a NetCDF file."""
    ds = sweep_to_xarray(results_df, ref_df, fit_df, cfg, s21_df=s21_df)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, engine="netcdf4")
    if verbose:
        print(f"  Saved NetCDF: {path}")


def load_sweep_netcdf(
    path: str | Path,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Load a sweep NetCDF file back into DataFrames.

    Returns a dict with the same keys as ``run_sweep``:
    ``"results"``, ``"references"``, ``"fits"``, and (if present)
    ``"dataset"`` (the raw xr.Dataset for direct xarray usage).

    Parameters
    ----------
    path : str or Path
        Path to a ``.nc`` file produced by :func:`save_sweep_netcdf`.
    verbose : bool
        Print summary.
    """
    xr = _import_xarray()
    ds = xr.open_dataset(path)

    freqs = ds.coords["rf_frequency_hz"].values
    powers = ds.coords["rf_power_dbm"].values

    def _split_attr(name: str) -> list[str]:
        raw = str(ds.attrs.get(name, ""))
        return [s for s in raw.split(",") if s]

    sweep_vars = _split_attr("_sweep_vars")
    ref_vars = _split_attr("_ref_vars")
    fit_vars = _split_attr("_fit_vars")

    # -- Reconstruct 2D results DataFrame ----------------------------------
    rows: list[dict] = []
    for fi, f in enumerate(freqs):
        for pi, p in enumerate(powers):
            row: dict = {
                "rf_frequency_hz": float(f),
                "rf_power_dbm": float(p),
            }
            for var in sweep_vars:
                if var in ds:
                    row[var] = float(ds[var].values[fi, pi])
            rows.append(row)
    results_df = pd.DataFrame(rows)

    for attr in _STRING_COLS:
        if attr in ds.attrs:
            results_df[attr] = ds.attrs[attr]

    # -- Reconstruct 1D reference DataFrame --------------------------------
    ref_data: dict = {"rf_frequency_hz": freqs}
    for var in ref_vars:
        if var in ds:
            original = var[4:] if var.startswith("ref_") else var
            ref_data[original] = ds[var].values
    ref_df = pd.DataFrame(ref_data) if len(ref_data) > 1 else pd.DataFrame()

    # -- Reconstruct 1D fit DataFrame --------------------------------------
    fit_data: dict = {"rf_frequency_hz": freqs}
    for var in fit_vars:
        if var in ds:
            fit_data[var] = ds[var].values
    fit_df = pd.DataFrame(fit_data) if len(fit_data) > 1 else pd.DataFrame()

    if verbose:
        n = len(results_df)
        nf = len(freqs)
        np_ = len(powers)
        print(f"Loaded {path}: {n} points ({nf} freqs × {np_} powers), "
              f"{len(ref_df)} references, {len(fit_df)} fits")

    return {
        "results": results_df,
        "references": ref_df,
        "fits": fit_df,
        "dataset": ds,
    }
