"""Orchestration logic for an RF Vpi sweep.

This module is hardware-agnostic: it accepts any objects satisfying
``ScopeInterface`` and ``RFSourceInterface`` from ``cavityscope.core``.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from cavityscope.analysis.measurement import (
    add_voltage_columns,
    measure_trace_against_reference,
)
from cavityscope.analysis.plotting import (
    plot_beta_fit,
    plot_trace_frequency_space,
    plot_trace_with_windows,
)
from cavityscope.analysis.reference import analyze_reference_trace
from cavityscope.analysis.vpi_fitting import fit_beta_vs_vpk
from cavityscope.core.config import SweepConfig
from cavityscope.core.instruments import RFSourceInterface, ScopeInterface
from cavityscope.core.utils import make_measurement_output_dirs


def run_sweep(
    scope: ScopeInterface,
    rf_source: RFSourceInterface,
    cfg: SweepConfig,
    scope_channel: int = 1,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Execute a full RF Vpi sweep and return result DataFrames.

    Parameters
    ----------
    scope : ScopeInterface
        An already-opened oscilloscope.
    rf_source : RFSourceInterface
        An already-opened RF signal generator.
    cfg : SweepConfig
        Sweep configuration.
    scope_channel : int
        Which scope channel to read (1-4).
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    dict with keys ``"results"``, ``"references"``, ``"fits"``
        Each value is a :class:`pandas.DataFrame`.
    """
    dirs = make_measurement_output_dirs(cfg.output_dir)
    run_dir = dirs["run_dir"]

    with open(run_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    if verbose:
        print("Measurement folder:", run_dir)
        print("Connected scope:", scope.idn())

    rf_source.set_output(False)
    time.sleep(0.2)

    results: List[Dict] = []
    reference_rows: List[Dict] = []
    fit_rows: List[Dict] = []

    for freq_hz in cfg.rf_frequencies_hz:
        freq_hz = float(freq_hz)
        if verbose:
            print(f"\n=== RF frequency: {freq_hz/1e9:.6f} GHz ===")

        rf_source.apply(freq_hz=freq_hz, enabled=False)
        time.sleep(cfg.settle_after_rf_change_s)

        scope.acquire_single_and_wait(
            timeout_s=cfg.trigger_timeout_s,
            settle_s=cfg.settle_after_scope_single_s,
        )
        t_ref, y_ref, _ = scope.read_waveform(scope_channel)
        ref = analyze_reference_trace(t_ref, y_ref, freq_hz, cfg)

        reference_rows.append(
            {
                "rf_frequency_hz": freq_hz,
                "reference_fsr_time_s": ref.fsr_time_s,
                "reference_baseline_v": ref.baseline_v,
                "n_reference_carriers": len(ref.all_carrier_times_s),
                "chosen_carrier_time_s": ref.chosen_carrier_time_s,
                "chosen_carrier_height_v": ref.chosen_carrier_height_v,
                "carrier_window_hz": cfg.carrier_window_hz,
                "sideband_window_hz": cfg.sideband_window_hz,
            }
        )

        ref_picked = {
            "carrier_times_s": np.asarray([ref.chosen_carrier_time_s]),
            "carrier_heights_v": np.asarray([ref.chosen_carrier_height_v]),
            "sb_minus_times_s": np.asarray([]),
            "sb_minus_heights_v": np.asarray([]),
            "sb_plus_times_s": np.asarray([]),
            "sb_plus_heights_v": np.asarray([]),
        }

        if cfg.save_reference_plots:
            plot_trace_with_windows(
                t_ref,
                y_ref,
                ref,
                rf_frequency_hz=freq_hz,
                title=f"Reference trace, RF off, f_RF={freq_hz/1e9:.6f} GHz",
                out_png=dirs["refs_dir"] / f"reference_{freq_hz/1e6:.0f}MHz.png",
                cfg=cfg,
                picked_points=ref_picked,
            )

        if cfg.save_frequency_plots:
            plot_trace_frequency_space(
                t_ref,
                y_ref,
                ref,
                rf_frequency_hz=freq_hz,
                title=f"Reference (freq. space), RF off, f_RF={freq_hz/1e9:.6f} GHz",
                out_png=dirs["freq_dir"] / f"freq_reference_{freq_hz/1e6:.0f}MHz.png",
                cfg=cfg,
                picked_points=ref_picked,
            )

        if cfg.save_raw_traces_csv:
            pd.DataFrame({"t_s": t_ref, "v": y_ref}).to_csv(
                dirs["raw_dir"] / f"reference_{freq_hz/1e6:.0f}MHz.csv",
                index=False,
            )

        for power_dbm in cfg.rf_powers_dbm:
            power_dbm = float(power_dbm)
            if verbose:
                print(f"  power = {power_dbm:7.3f} dBm")

            rf_source.apply(freq_hz=freq_hz, power_dbm=power_dbm, enabled=True)
            time.sleep(cfg.settle_after_rf_change_s)

            scope.acquire_single_and_wait(
                timeout_s=cfg.trigger_timeout_s,
                settle_s=cfg.settle_after_scope_single_s,
            )
            t, y, _ = scope.read_waveform(scope_channel)

            meas, picked = measure_trace_against_reference(
                t=t, y_v=y, ref=ref, rf_frequency_hz=freq_hz, cfg=cfg
            )
            row = {
                "rf_frequency_hz": freq_hz,
                "rf_power_dbm": power_dbm,
                "reference_fsr_time_s": ref.fsr_time_s,
                **meas,
            }
            row = add_voltage_columns(row, power_dbm, cfg)
            results.append(row)

            if cfg.save_trace_plots:
                mode_label = cfg.sideband_mode.lower()
                trace_label = f"f_RF={freq_hz/1e9:.6f} GHz, P_RF={power_dbm:.2f} dBm, mode={mode_label}"
                trace_stem = f"trace_{freq_hz/1e6:.0f}MHz_{power_dbm:+06.2f}dBm"
                plot_trace_with_windows(
                    t,
                    y,
                    ref,
                    rf_frequency_hz=freq_hz,
                    title=trace_label,
                    out_png=dirs["traces_dir"] / f"{trace_stem}.png",
                    cfg=cfg,
                    picked_points=picked,
                )

            if cfg.save_frequency_plots:
                freq_label = (
                    f"Freq. space: f_RF={freq_hz/1e9:.6f} GHz, "
                    f"P_RF={power_dbm:.2f} dBm"
                )
                freq_stem = f"freq_{freq_hz/1e6:.0f}MHz_{power_dbm:+06.2f}dBm"
                plot_trace_frequency_space(
                    t,
                    y,
                    ref,
                    rf_frequency_hz=freq_hz,
                    title=freq_label,
                    out_png=dirs["freq_dir"] / f"{freq_stem}.png",
                    cfg=cfg,
                    picked_points=picked,
                )

            if cfg.save_raw_traces_csv:
                pd.DataFrame({"t_s": t, "v": y}).to_csv(
                    dirs["raw_dir"]
                    / f"trace_{freq_hz/1e6:.0f}MHz_{power_dbm:+06.2f}dBm.csv",
                    index=False,
                )

    df = pd.DataFrame(results)
    ref_df = pd.DataFrame(reference_rows)
    fit_df = pd.DataFrame()

    if not df.empty and cfg.compute_vpi:
        for fhz, dfg in df.groupby("rf_frequency_hz"):
            fit_row = {"rf_frequency_hz": float(fhz), **fit_beta_vs_vpk(dfg, cfg)}
            fit_rows.append(fit_row)
            plot_beta_fit(
                dfg,
                fit_row,
                dirs["fit_dir"] / f"vpi_fit_{fhz/1e6:.0f}MHz.png",
                float(fhz),
                cfg,
            )
        fit_df = pd.DataFrame(fit_rows)

    df.to_csv(run_dir / "sweep_results.csv", index=False)
    ref_df.to_csv(run_dir / "reference_summary.csv", index=False)
    fit_df.to_csv(run_dir / "vpi_fit_summary.csv", index=False)

    if verbose:
        print("\nSaved to:", run_dir)

    return {"results": df, "references": ref_df, "fits": fit_df}
