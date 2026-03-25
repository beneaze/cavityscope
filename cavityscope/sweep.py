"""Hardware orchestration for an RF Vpi sweep.

This module drives oscilloscopes and RF sources via the interfaces in
``cavityscope.core``.  All pure-analysis logic (voltage calibration,
Vpi fitting, reanalysis) lives in ``cavityscope.analysis``.
"""

from __future__ import annotations

import gc
import json
import time
from typing import Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from cavityscope.analysis.measurement import (
    add_voltage_columns,
    compute_carrier_area,
    measure_trace_against_reference,
    preprocess_trace,
)
from cavityscope.analysis.plotting import (
    plot_calibration_fit,
    plot_trace_frequency_space,
    plot_trace_with_windows,
)
from cavityscope.analysis.postprocess import (
    build_calibration,
    compute_s21_analysis,
    compute_vpi_fits,
)
from cavityscope.analysis.reference import analyze_reference_trace
from cavityscope.analysis.rf_voltage import extract_vpk_from_trace
from cavityscope.core.calibration import PowerCalibration
from cavityscope.core.config import SweepConfig
from cavityscope.core.instruments import (
    RFSourceInterface,
    ScopeInterface,
    SpectrumAnalyzerInterface,
)
from cavityscope.core.utils import (
    IncrementalCsvWriter,
    dbm_to_vrms_into_r,
    ensure_dir,
    make_calibration_output_dir,
    make_measurement_output_dirs,
)


def _save_trace(
    raw_dir, stem: str, t: np.ndarray, y: np.ndarray, fmt: str,
) -> None:
    """Save a raw scope trace in the requested format."""
    if fmt == "npy":
        np.save(raw_dir / f"{stem}.npy", np.column_stack([t, y]))
    elif fmt == "csv":
        pd.DataFrame({"t_s": t, "v": y}).to_csv(
            raw_dir / f"{stem}.csv", index=False,
        )
    elif fmt == "npz":
        np.savez_compressed(raw_dir / f"{stem}.npz", t_s=t, v=y)


def _acquire_with_retry(
    scope: ScopeInterface,
    channel: int,
    timeout_s: float,
    settle_s: float,
    max_retries: int = 3,
    verbose: bool = False,
):
    """Acquire a single trace, retrying on empty data or read errors."""
    has_fast = hasattr(scope, "read_waveform_fast")
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            scope.acquire_single_and_wait(timeout_s=timeout_s, settle_s=settle_s)
            if has_fast:
                t, y, info = scope.read_waveform_fast()
            else:
                t, y, info = scope.read_waveform(channel)
            if t.size > 0:
                return t, y, info
            reason = "empty waveform"
        except Exception as exc:
            last_error = exc
            reason = str(exc)
        if verbose:
            print(f"    [retry {attempt}/{max_retries}] {reason}, re-acquiring...")
        if hasattr(scope, "flush"):
            scope.flush()
        time.sleep(0.5)
    msg = f"Scope read failed after {max_retries} retries on channel {channel}."
    if last_error is not None:
        raise RuntimeError(msg) from last_error
    raise RuntimeError(msg)


def run_power_calibration(
    scope: ScopeInterface,
    rf_source: RFSourceInterface,
    cfg: SweepConfig,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> PowerCalibration:
    """Run a separate scope-based power calibration before the main sweep.

    For each (frequency, power) in the sweep grid the scope timebase is set
    so that ~``cfg.cal_cycles_to_capture`` RF cycles fill the screen, then
    a single acquisition is taken on ``cfg.cal_scope_channel`` and the Vpk
    is extracted via a sine fit.

    The original scope timebase is restored afterwards.

    Parameters
    ----------
    scope : ScopeInterface
        An already-opened oscilloscope.
    rf_source : RFSourceInterface
        An already-opened RF signal generator.
    cfg : SweepConfig
        Sweep configuration (uses the same frequency/power grid).
    output_dir : str, optional
        Where to save ``power_calibration.csv`` and the plot.
        Defaults to ``cfg.output_dir``.
    verbose : bool
        Print progress.

    Returns
    -------
    PowerCalibration
        Ready to pass into :func:`run_sweep` as the *calibration* argument.
    """
    if cfg.cal_scope_channel is None:
        raise ValueError("cfg.cal_scope_channel must be set to run power calibration.")

    cal_ch = cfg.cal_scope_channel
    n_cycles = cfg.cal_cycles_to_capture
    out = make_calibration_output_dir(output_dir or cfg.output_dir)

    with open(out / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    original_timebase = scope.get_timebase()
    if verbose:
        print(f"Power calibration: scope ch{cal_ch}, {n_cycles} cycles/acquisition")
        print(f"  Original timebase: {original_timebase:.3E} s/div")

    rf_source.set_output(False)
    time.sleep(0.2)

    cal_csv = IncrementalCsvWriter(out / "power_calibration.csv")
    fit_plot_dir = ensure_dir(out / "fit_plots")

    manual_timebase = cfg.cal_timebase_s_per_div
    min_visible_cycles = max(n_cycles, 5)

    n_total = len(cfg.rf_frequencies_hz) * len(cfg.rf_powers_dbm)
    pbar = tqdm(total=n_total, desc="Scope cal", unit="pt",
                disable=not verbose, leave=True)

    _plot_fig = Figure()
    FigureCanvasAgg(_plot_fig)

    try:
        for freq_hz in cfg.rf_frequencies_hz:
            freq_hz = float(freq_hz)
            rf_period = 1.0 / freq_hz

            rf_source.set_output(False)

            if manual_timebase is not None:
                scale = manual_timebase
            else:
                desired_window = min_visible_cycles * rf_period
                scale = desired_window / 10.0
            scope.set_timebase(scale)
            time.sleep(0.1)

            actual_scale = scope.get_timebase()
            actual_window = actual_scale * 10.0
            actual_cycles = actual_window / rf_period

            if verbose:
                mode = "manual" if manual_timebase is not None else "auto"
                pbar.write(
                    f"\n  f = {freq_hz/1e9:.4f} GHz  "
                    f"({mode}: {actual_scale:.3E} s/div, "
                    f"~{actual_cycles:.0f} cycles visible)"
                )

            rf_source.apply(freq_hz=freq_hz, power_dbm=float(cfg.rf_powers_dbm[0]),
                            enabled=True)
            time.sleep(cfg.cal_settle_s)

            for power_dbm in cfg.rf_powers_dbm:
                power_dbm = float(power_dbm)
                pbar.set_postfix_str(
                    f"{freq_hz/1e9:.4f} GHz, {power_dbm:+.1f} dBm"
                )
                rf_source.apply(freq_hz=freq_hz, power_dbm=power_dbm)
                time.sleep(cfg.cal_settle_s)

                t_rf, v_rf, _ = _acquire_with_retry(
                    scope, cal_ch,
                    timeout_s=cfg.trigger_timeout_s, settle_s=0.02,
                    max_retries=cfg.scope_read_max_retries, verbose=verbose,
                )

                fit_cycles = max(5, int(actual_cycles))
                meas = extract_vpk_from_trace(
                    t_rf, v_rf,
                    rf_frequency_hz=freq_hz,
                    n_cycles=fit_cycles,
                )
                meas["power_dbm"] = power_dbm
                row = {
                    "frequency_hz": freq_hz,
                    "power_dbm": power_dbm,
                    "vpk_v": meas["measured_vpk_v"],
                    **{k: v for k, v in meas.items()
                       if k not in ("measured_vpk_v", "power_dbm")},
                }
                cal_csv.write_row(row)

                if verbose:
                    ok = "ok" if meas["fit_converged"] else "FALLBACK"
                    pbar.write(f"    {power_dbm:+7.2f} dBm → Vpk = {meas['measured_vpk_v']:.4f} V  [{ok}]")

                try:
                    plot_calibration_fit(
                        t_rf, v_rf,
                        rf_frequency_hz=freq_hz,
                        meas=meas,
                        out_png=fit_plot_dir
                        / f"cal_fit_{freq_hz/1e6:.4f}MHz_{power_dbm:+06.2f}dBm.png",
                        n_cycles=fit_cycles,
                        reuse_fig=_plot_fig,
                    )
                except Exception as exc:
                    if verbose:
                        pbar.write(f"    [warning] fit plot failed: {exc}")

                del t_rf, v_rf, meas
                plt.close("all")
                gc.collect()

                pbar.update(1)

        rf_source.set_output(False)
    finally:
        pbar.close()
        cal_csv.close()
        _plot_fig.clf()
        del _plot_fig
        scope.set_timebase(original_timebase)
        if verbose:
            print(f"\n  Restored timebase: {original_timebase:.3E} s/div")

    cal_path = out / "power_calibration.csv"
    cal_df = pd.read_csv(cal_path) if cal_path.exists() else pd.DataFrame()
    calibration = build_calibration(cal_df, output_dir=out, verbose=verbose)

    if verbose:
        print(f"  Calibration folder: {out}")

    return calibration


def run_sa_power_calibration(
    sa: SpectrumAnalyzerInterface,
    rf_source: RFSourceInterface,
    cfg: SweepConfig,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> PowerCalibration:
    """Run power calibration using a spectrum analyzer with harmonic analysis.

    For each (frequency, power) in the sweep grid the spectrum analyzer
    measures the fundamental tone and its harmonics (2f, 3f, …).  The
    fundamental power is converted to Vpk; the harmonic data is saved
    alongside the calibration for diagnostics.

    Unlike the scope-based calibration, this approach is immune to
    harmonics from the signal generator — the SA resolves the fundamental
    cleanly, and the harmonic analysis quantifies how much distortion is
    present so you can judge amplifier linearity.

    Parameters
    ----------
    sa : SpectrumAnalyzerInterface
        An already-opened spectrum analyzer.  Must support
        ``measure_harmonics(...)`` for the full diagnostic, or at
        minimum ``measure_power_at_frequency(...)`` for the basic
        fundamental-only path.
    rf_source : RFSourceInterface
        An already-opened RF signal generator.
    cfg : SweepConfig
        Sweep configuration (uses the same frequency/power grid).
    output_dir : str, optional
        Where to save calibration files.  Defaults to ``cfg.output_dir``.
    verbose : bool
        Print progress.

    Returns
    -------
    PowerCalibration
        Ready to pass into :func:`run_sweep` as the *calibration* argument.
    """
    import math

    from cavityscope.analysis.harmonics import compute_harmonic_metrics
    from cavityscope.analysis.plotting import (
        plot_harmonic_heatmap,
        plot_harmonic_waterfall,
        plot_sa_spectrum,
        plot_thd_summary,
    )

    out = make_calibration_output_dir(output_dir or cfg.output_dir)

    with open(out / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    has_harmonics = hasattr(sa, "measure_harmonics")
    n_harmonics = cfg.cal_sa_n_harmonics if has_harmonics else 0

    if verbose:
        sa_id = sa.idn() if hasattr(sa, "idn") else "unknown"
        print(f"SA power calibration: {sa_id}")
        print(f"  Span: {cfg.cal_sa_span_hz/1e3:.0f} kHz, "
              f"RBW: {cfg.cal_sa_rbw_hz or 'auto'}, "
              f"Ref level: {cfg.cal_sa_ref_level_dbm} dBm")
        if cfg.cal_sa_power_offset_db != 0.0:
            print(f"  Power offset: {cfg.cal_sa_power_offset_db:+.1f} dB "
                  f"(added to raw SA reading before dBm→Vpk)")
        if n_harmonics > 1:
            print(f"  Harmonic analysis: up to {n_harmonics} harmonics")

    rf_source.set_output(False)
    time.sleep(0.2)

    cal_csv = IncrementalCsvWriter(out / "power_calibration.csv")
    harmonics_csv_writer = IncrementalCsvWriter(out / "harmonics.csv")
    thd_csv_writer = IncrementalCsvWriter(out / "thd_summary.csv")
    has_harmonic_data = False
    spectrum_dir = ensure_dir(out / "spectra") if cfg.cal_sa_save_spectra else None

    n_total = len(cfg.rf_frequencies_hz) * len(cfg.rf_powers_dbm)
    pbar = tqdm(total=n_total, desc="SA cal", unit="pt",
                disable=not verbose, leave=True)

    _plot_fig = Figure()
    FigureCanvasAgg(_plot_fig)

    try:
        for freq_hz in cfg.rf_frequencies_hz:
            freq_hz = float(freq_hz)

            rf_source.set_output(False)
            time.sleep(0.1)

            if verbose:
                pbar.write(f"\n  f = {freq_hz/1e9:.4f} GHz")

            rf_source.apply(freq_hz=freq_hz, power_dbm=float(cfg.rf_powers_dbm[0]),
                            enabled=True)
            time.sleep(cfg.cal_sa_settle_s)

            for power_dbm in cfg.rf_powers_dbm:
                power_dbm = float(power_dbm)
                pbar.set_postfix_str(
                    f"{freq_hz/1e9:.4f} GHz, {power_dbm:+.1f} dBm"
                )
                rf_source.apply(freq_hz=freq_hz, power_dbm=power_dbm)
                time.sleep(cfg.cal_sa_settle_s)

                if has_harmonics and n_harmonics >= 2:
                    hdata = sa.measure_harmonics(
                        fundamental_hz=freq_hz,
                        n_harmonics=n_harmonics,
                        per_tone_span_hz=cfg.cal_sa_span_hz,
                        rbw_hz=cfg.cal_sa_rbw_hz,
                        ref_level_dbm=cfg.cal_sa_ref_level_dbm,
                        settle_s=cfg.cal_sa_settle_s,
                    )
                    harmonics_list = hdata["harmonics"]
                    wb_freqs, wb_amps = hdata["wideband_trace"]

                    fund = next(h for h in harmonics_list if h["harmonic_number"] == 1)
                    measured_dbm = fund["power_dbm"]
                    measured_freq_hz = fund["measured_freq_hz"]

                    metrics = compute_harmonic_metrics(harmonics_list)

                    has_harmonic_data = True
                    fund_dbm = fund["power_dbm"]
                    for h in harmonics_list:
                        dbc = h["power_dbm"] - fund_dbm
                        harmonics_csv_writer.write_row({
                            "frequency_hz": freq_hz,
                            "power_dbm": power_dbm,
                            "harmonic_number": h["harmonic_number"],
                            "nominal_freq_hz": h["nominal_freq_hz"],
                            "measured_freq_hz": h["measured_freq_hz"],
                            "harmonic_power_dbm": h["power_dbm"],
                            "level_dbc": dbc,
                        })
                    thd_csv_writer.write_row({
                        "frequency_hz": freq_hz,
                        "power_dbm": power_dbm,
                        "fundamental_power_dbm": metrics["fundamental_power_dbm"],
                        "thd_percent": metrics["thd_percent"],
                        "fundamental_power_fraction": metrics["fundamental_power_fraction"],
                        "total_harmonic_power_dbm": metrics["total_harmonic_power_dbm"],
                    })

                    if spectrum_dir is not None:
                        try:
                            plot_sa_spectrum(
                                wb_freqs, wb_amps,
                                harmonics_list,
                                fundamental_hz=freq_hz,
                                power_dbm_setting=power_dbm,
                                metrics=metrics,
                                power_offset_db=cfg.cal_sa_power_offset_db,
                                out_png=spectrum_dir
                                / f"spectrum_{freq_hz/1e6:.4f}MHz_{power_dbm:+06.2f}dBm.png",
                                reuse_fig=_plot_fig,
                            )
                        except Exception as exc:
                            if verbose:
                                pbar.write(f"    [warning] spectrum plot failed: {exc}")

                    if verbose:
                        thd = metrics["thd_percent"]
                        frac = metrics["fundamental_power_fraction"]
                        corrected = measured_dbm + cfg.cal_sa_power_offset_db
                        pbar.write(f"    {power_dbm:+7.2f} dBm → "
                              f"{measured_dbm:+7.2f} dBm (SA raw)"
                              + (f" → {corrected:+7.2f} dBm (corrected)"
                                 if cfg.cal_sa_power_offset_db != 0.0 else "")
                              + f" → THD = {thd:.1f}%, "
                              f"fund = {frac*100:.1f}% of total")

                    del wb_freqs, wb_amps, hdata
                else:
                    measured_freq_hz, measured_dbm = sa.measure_power_at_frequency(
                        freq_hz=freq_hz,
                        span_hz=cfg.cal_sa_span_hz,
                        rbw_hz=cfg.cal_sa_rbw_hz,
                        ref_level_dbm=cfg.cal_sa_ref_level_dbm,
                        settle_s=cfg.cal_sa_settle_s,
                    )
                    harmonics_list = None
                    metrics = None
                    if verbose:
                        corrected = measured_dbm + cfg.cal_sa_power_offset_db
                        pbar.write(f"    {power_dbm:+7.2f} dBm → "
                              f"{measured_dbm:+7.2f} dBm (SA raw)"
                              + (f" → {corrected:+7.2f} dBm (corrected)"
                                 if cfg.cal_sa_power_offset_db != 0.0 else ""))

                corrected_dbm = measured_dbm + cfg.cal_sa_power_offset_db
                vrms = dbm_to_vrms_into_r(corrected_dbm, cfg.assumed_load_ohm)
                vpk = vrms * math.sqrt(2.0)

                row = {
                    "frequency_hz": freq_hz,
                    "power_dbm": power_dbm,
                    "vpk_v": vpk,
                    "measured_power_dbm": measured_dbm,
                    "measured_frequency_hz": measured_freq_hz,
                }
                if harmonics_list is not None and metrics is not None:
                    row["thd_percent"] = metrics["thd_percent"]
                    row["fundamental_power_fraction"] = metrics["fundamental_power_fraction"]
                    for h in harmonics_list:
                        k = h["harmonic_number"]
                        row[f"h{k}_power_dbm"] = h["power_dbm"]
                        if k >= 2:
                            row[f"h{k}_dbc"] = h["power_dbm"] - measured_dbm

                cal_csv.write_row(row)

                del row, harmonics_list, metrics
                plt.close("all")
                gc.collect()

                pbar.update(1)

        rf_source.set_output(False)
    finally:
        pbar.close()
        cal_csv.close()
        harmonics_csv_writer.close()
        thd_csv_writer.close()
        _plot_fig.clf()
        del _plot_fig

    cal_path = out / "power_calibration.csv"
    cal_df = pd.read_csv(cal_path) if cal_path.exists() else pd.DataFrame()
    calibration = build_calibration(cal_df, output_dir=out, verbose=verbose)

    if has_harmonic_data:
        harmonics_df = pd.read_csv(out / "harmonics.csv")
        thd_df = pd.read_csv(out / "thd_summary.csv")

        try:
            plot_thd_summary(thd_df, out / "thd_summary.png")
        except Exception as exc:
            if verbose:
                print(f"  [warning] THD summary plot failed: {exc}")

        try:
            plot_harmonic_heatmap(harmonics_df, out / "harmonic_heatmap.png")
        except Exception as exc:
            if verbose:
                print(f"  [warning] harmonic heatmap plot failed: {exc}")

        for freq_hz in cfg.rf_frequencies_hz:
            try:
                plot_harmonic_waterfall(
                    harmonics_df, float(freq_hz),
                    out / f"harmonic_waterfall_{float(freq_hz)/1e6:.4f}MHz.png",
                )
            except Exception as exc:
                if verbose:
                    print(f"  [warning] waterfall plot failed for {freq_hz}: {exc}")

        if verbose:
            max_thd = thd_df["thd_percent"].max()
            mean_thd = thd_df["thd_percent"].mean()
            print(f"\n  Harmonic analysis: mean THD = {mean_thd:.1f}%, max THD = {max_thd:.1f}%")
            print(f"  Saved: harmonics.csv, thd_summary.csv, plots")

    if verbose:
        print(f"  Calibration folder: {out}")

    return calibration


def run_sweep(
    scope: ScopeInterface,
    rf_source: RFSourceInterface,
    cfg: SweepConfig,
    scope_channel: int = 1,
    calibration: Optional[PowerCalibration] = None,
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
    calibration : PowerCalibration, optional
        If provided, maps (power_dBm, frequency_hz) to actual Vpk.
        Overrides the analytical dBm→V conversion.  Can also be loaded
        automatically from ``cfg.power_calibration_csv``.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    dict with keys ``"results"``, ``"references"``, ``"fits"``
        Each value is a :class:`pandas.DataFrame`.
    """
    if calibration is None and cfg.power_calibration_csv is not None:
        calibration = PowerCalibration.from_csv(cfg.power_calibration_csv)

    dirs = make_measurement_output_dirs(cfg.output_dir)
    run_dir = dirs["run_dir"]

    with open(run_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    if verbose:
        print("Measurement folder:", run_dir)
        print("Connected scope:", scope.idn())
        if calibration is not None:
            print("Power calibration:", calibration)

    rf_source.set_output(False)
    time.sleep(0.2)

    original_mdepth = None
    if cfg.scope_memory_depth is not None and hasattr(scope, "set_memory_depth"):
        original_mdepth = scope.get_memory_depth()
        scope.set_memory_depth(cfg.scope_memory_depth)
        if verbose:
            print(f"Scope memory depth: {original_mdepth} -> {cfg.scope_memory_depth}")

    if hasattr(scope, "prepare_for_readout"):
        scope.prepare_for_readout(scope_channel)

    sweep_csv = IncrementalCsvWriter(run_dir / "sweep_results.csv")
    ref_csv = IncrementalCsvWriter(run_dir / "reference_summary.csv")

    n_freqs = len(cfg.rf_frequencies_hz)
    n_powers = len(cfg.rf_powers_dbm)
    n_total = n_freqs * (1 + n_powers)  # 1 reference + n_powers per frequency
    pbar = tqdm(total=n_total, desc="Vpi sweep", unit="pt",
                disable=not verbose, leave=True)
    point_counter = 0

    _plot_fig = Figure()
    FigureCanvasAgg(_plot_fig)

    try:
        for freq_hz in cfg.rf_frequencies_hz:
            freq_hz = float(freq_hz)
            if verbose:
                pbar.write(f"\n=== RF frequency: {freq_hz/1e9:.6f} GHz ===")

            pbar.set_postfix_str(f"{freq_hz/1e9:.4f} GHz, ref")
            rf_source.apply(freq_hz=freq_hz, enabled=False)
            time.sleep(cfg.settle_after_rf_change_s)

            t_ref, y_ref, _ = _acquire_with_retry(
                scope, scope_channel,
                timeout_s=cfg.trigger_timeout_s,
                settle_s=cfg.settle_after_scope_single_s,
                max_retries=cfg.scope_read_max_retries, verbose=verbose,
            )
            ref = analyze_reference_trace(t_ref, y_ref, freq_hz, cfg)
            ref_carrier_area = compute_carrier_area(t_ref, y_ref, ref, cfg)

            ref_row = {
                "rf_frequency_hz": freq_hz,
                "reference_fsr_time_s": ref.fsr_time_s,
                "reference_baseline_v": ref.baseline_v,
                "n_reference_carriers": len(ref.all_carrier_times_s),
                "chosen_carrier_time_s": ref.chosen_carrier_time_s,
                "chosen_carrier_height_v": ref.chosen_carrier_height_v,
                "integration_window_hz": cfg.integration_window_hz,
                "reference_carrier_area_v_s": ref_carrier_area,
            }
            ref_csv.write_row(ref_row)

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
                    out_png=dirs["refs_dir"] / f"reference_{freq_hz/1e6:.4f}MHz.png",
                    cfg=cfg,
                    picked_points=ref_picked,
                    reuse_fig=_plot_fig,
                )
                gc.collect()

            if cfg.save_frequency_plots:
                plot_trace_frequency_space(
                    t_ref,
                    y_ref,
                    ref,
                    rf_frequency_hz=freq_hz,
                    title=f"Reference (freq. space), RF off, f_RF={freq_hz/1e9:.6f} GHz",
                    out_png=dirs["freq_dir"] / f"freq_reference_{freq_hz/1e6:.4f}MHz.png",
                    cfg=cfg,
                    picked_points=ref_picked,
                    reuse_fig=_plot_fig,
                )

            if cfg.save_raw_traces_csv:
                _save_trace(
                    dirs["raw_dir"], f"reference_{freq_hz/1e6:.4f}MHz",
                    t_ref, y_ref, cfg.raw_trace_format,
                )

            del t_ref, y_ref, ref_picked
            plt.close("all")
            gc.collect()

            pbar.update(1)

            rf_source.apply(freq_hz=freq_hz, power_dbm=float(cfg.rf_powers_dbm[0]),
                            enabled=True)
            time.sleep(cfg.settle_after_rf_change_s)

            for power_dbm in cfg.rf_powers_dbm:
                power_dbm = float(power_dbm)
                point_counter += 1
                pbar.set_postfix_str(
                    f"{freq_hz/1e9:.4f} GHz, {power_dbm:+.1f} dBm"
                )
                if verbose:
                    pbar.write(f"  power = {power_dbm:7.3f} dBm")

                rf_source.apply(freq_hz=freq_hz, power_dbm=power_dbm)
                time.sleep(cfg.settle_after_power_change_s)

                t, y, _ = _acquire_with_retry(
                    scope, scope_channel,
                    timeout_s=cfg.trigger_timeout_s,
                    settle_s=cfg.settle_after_scope_single_s,
                    max_retries=cfg.scope_read_max_retries, verbose=verbose,
                )

                pp = preprocess_trace(y, cfg)

                meas, picked = measure_trace_against_reference(
                    t=t, y_v=y, ref=ref, rf_frequency_hz=freq_hz, cfg=cfg,
                    preprocessed=pp,
                )
                row = {
                    "rf_frequency_hz": freq_hz,
                    "rf_power_dbm": power_dbm,
                    "reference_fsr_time_s": ref.fsr_time_s,
                    **meas,
                }
                row = add_voltage_columns(
                    row, power_dbm, cfg,
                    calibration=calibration,
                    rf_frequency_hz=freq_hz,
                )
                sweep_csv.write_row(row)

                do_plots = (
                    cfg.plot_every_n_points <= 1
                    or point_counter % cfg.plot_every_n_points == 1
                )

                if cfg.save_trace_plots and do_plots:
                    mode_label = cfg.sideband_mode.lower()
                    trace_label = f"f_RF={freq_hz/1e9:.6f} GHz, P_RF={power_dbm:.2f} dBm, mode={mode_label}"
                    trace_stem = f"trace_{freq_hz/1e6:.4f}MHz_{power_dbm:+06.2f}dBm"
                    plot_trace_with_windows(
                        t,
                        y,
                        ref,
                        rf_frequency_hz=freq_hz,
                        title=trace_label,
                        out_png=dirs["traces_dir"] / f"{trace_stem}.png",
                        cfg=cfg,
                        picked_points=picked,
                        preprocessed=pp,
                        reuse_fig=_plot_fig,
                    )
                    gc.collect()

                if cfg.save_frequency_plots and do_plots:
                    freq_label = (
                        f"Freq. space: f_RF={freq_hz/1e9:.6f} GHz, "
                        f"P_RF={power_dbm:.2f} dBm"
                    )
                    freq_stem = f"freq_{freq_hz/1e6:.4f}MHz_{power_dbm:+06.2f}dBm"
                    plot_trace_frequency_space(
                        t,
                        y,
                        ref,
                        rf_frequency_hz=freq_hz,
                        title=freq_label,
                        out_png=dirs["freq_dir"] / f"{freq_stem}.png",
                        cfg=cfg,
                        picked_points=picked,
                        preprocessed=pp,
                        reuse_fig=_plot_fig,
                    )

                if cfg.save_raw_traces_csv:
                    _save_trace(
                        dirs["raw_dir"],
                        f"trace_{freq_hz/1e6:.4f}MHz_{power_dbm:+06.2f}dBm",
                        t, y, cfg.raw_trace_format,
                    )

                del t, y, meas, picked, pp, row
                plt.close("all")
                gc.collect()

                pbar.update(1)

    finally:
        pbar.close()
        sweep_csv.close()
        ref_csv.close()
        _plot_fig.clf()
        del _plot_fig
        if original_mdepth is not None and hasattr(scope, "set_memory_depth"):
            scope.set_memory_depth(original_mdepth)

    sweep_path = run_dir / "sweep_results.csv"
    ref_path = run_dir / "reference_summary.csv"
    df = pd.read_csv(sweep_path) if sweep_path.exists() else pd.DataFrame()
    ref_df = pd.read_csv(ref_path) if ref_path.exists() else pd.DataFrame()
    fit_df = compute_vpi_fits(df, cfg, output_dir=run_dir)

    fit_df.to_csv(run_dir / "vpi_fit_summary.csv", index=False)

    s21_data: Dict[str, pd.DataFrame | Dict] = {}
    if cfg.plot_s21_response:
        s21_data = compute_s21_analysis(
            df, ref_df, cfg, output_dir=run_dir, verbose=verbose,
        )

    try:
        from cavityscope.analysis.xarray_export import save_sweep_netcdf

        save_sweep_netcdf(
            df, ref_df, fit_df, cfg,
            run_dir / "sweep_results.nc",
            s21_df=s21_data.get("s21_results"),
            verbose=verbose,
        )
    except ImportError:
        if verbose:
            print("  (xarray/netCDF4 not installed — skipping .nc export)")
    except Exception as exc:
        if verbose:
            print(f"  [warning] NetCDF export failed: {exc}")

    if verbose:
        print("\nSaved to:", run_dir)

    return {"results": df, "references": ref_df, "fits": fit_df, **s21_data}
