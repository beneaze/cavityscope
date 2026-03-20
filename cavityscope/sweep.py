"""Hardware orchestration for an RF Vpi sweep.

This module drives oscilloscopes and RF sources via the interfaces in
``cavityscope.core``.  All pure-analysis logic (voltage calibration,
Vpi fitting, reanalysis) lives in ``cavityscope.analysis``.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from cavityscope.analysis.measurement import (
    add_voltage_columns,
    measure_trace_against_reference,
)
from cavityscope.analysis.plotting import (
    plot_calibration_fit,
    plot_trace_frequency_space,
    plot_trace_with_windows,
)
from cavityscope.analysis.postprocess import build_calibration, compute_vpi_fits
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
    dbm_to_vrms_into_r,
    ensure_dir,
    make_calibration_output_dir,
    make_measurement_output_dirs,
)


def _acquire_with_retry(
    scope: ScopeInterface,
    channel: int,
    timeout_s: float,
    settle_s: float,
    max_retries: int = 3,
    verbose: bool = False,
):
    """Acquire a single trace, retrying on empty data or read errors."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            scope.acquire_single_and_wait(timeout_s=timeout_s, settle_s=settle_s)
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

    rows: List[Dict] = []
    fit_plot_dir = ensure_dir(out / "fit_plots")

    manual_timebase = cfg.cal_timebase_s_per_div
    min_visible_cycles = max(n_cycles, 5)

    try:
        for freq_hz in cfg.rf_frequencies_hz:
            freq_hz = float(freq_hz)
            rf_period = 1.0 / freq_hz

            # Turn RF off while we reconfigure the scope timebase so
            # the scope never sees a turn-on transient.
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
                print(
                    f"\n  f = {freq_hz/1e9:.4f} GHz  "
                    f"({mode}: {actual_scale:.3E} s/div, "
                    f"~{actual_cycles:.0f} cycles visible)"
                )

            # Enable RF output once per frequency; subsequent power
            # changes only adjust amplitude without re-toggling enable,
            # avoiding turn-on transients that can false-trigger the scope.
            rf_source.apply(freq_hz=freq_hz, power_dbm=float(cfg.rf_powers_dbm[0]),
                            enabled=True)
            time.sleep(cfg.cal_settle_s)

            for power_dbm in cfg.rf_powers_dbm:
                power_dbm = float(power_dbm)
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
                rows.append({
                    "frequency_hz": freq_hz,
                    "power_dbm": power_dbm,
                    "vpk_v": meas["measured_vpk_v"],
                    **{k: v for k, v in meas.items()
                       if k not in ("measured_vpk_v", "power_dbm")},
                })

                if verbose:
                    ok = "ok" if meas["fit_converged"] else "FALLBACK"
                    print(f"    {power_dbm:+7.2f} dBm → Vpk = {meas['measured_vpk_v']:.4f} V  [{ok}]")

                try:
                    plot_calibration_fit(
                        t_rf, v_rf,
                        rf_frequency_hz=freq_hz,
                        meas=meas,
                        out_png=fit_plot_dir
                        / f"cal_fit_{freq_hz/1e6:.4f}MHz_{power_dbm:+06.2f}dBm.png",
                        n_cycles=fit_cycles,
                    )
                except Exception as exc:
                    if verbose:
                        print(f"    [warning] fit plot failed: {exc}")

        rf_source.set_output(False)
    finally:
        scope.set_timebase(original_timebase)
        if verbose:
            print(f"\n  Restored timebase: {original_timebase:.3E} s/div")

    cal_df = pd.DataFrame(rows)
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

    from cavityscope.analysis.harmonics import (
        build_harmonics_dataframe,
        build_thd_dataframe,
        compute_harmonic_metrics,
    )
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

    rows: List[Dict] = []
    harmonic_measurements: List[Dict] = []
    spectrum_dir = ensure_dir(out / "spectra") if cfg.cal_sa_save_spectra else None

    try:
        for freq_hz in cfg.rf_frequencies_hz:
            freq_hz = float(freq_hz)

            rf_source.set_output(False)
            time.sleep(0.1)

            if verbose:
                print(f"\n  f = {freq_hz/1e9:.4f} GHz")

            rf_source.apply(freq_hz=freq_hz, power_dbm=float(cfg.rf_powers_dbm[0]),
                            enabled=True)
            time.sleep(cfg.cal_sa_settle_s)

            for power_dbm in cfg.rf_powers_dbm:
                power_dbm = float(power_dbm)
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

                    harmonic_measurements.append({
                        "frequency_hz": freq_hz,
                        "power_dbm": power_dbm,
                        "harmonics": harmonics_list,
                        "metrics": metrics,
                        "wideband_trace": (wb_freqs, wb_amps),
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
                            )
                        except Exception as exc:
                            if verbose:
                                print(f"    [warning] spectrum plot failed: {exc}")

                    if verbose:
                        thd = metrics["thd_percent"]
                        frac = metrics["fundamental_power_fraction"]
                        corrected = measured_dbm + cfg.cal_sa_power_offset_db
                        print(f"    {power_dbm:+7.2f} dBm → "
                              f"{measured_dbm:+7.2f} dBm (SA raw)"
                              + (f" → {corrected:+7.2f} dBm (corrected)"
                                 if cfg.cal_sa_power_offset_db != 0.0 else "")
                              + f" → THD = {thd:.1f}%, "
                              f"fund = {frac*100:.1f}% of total")
                else:
                    measured_freq_hz, measured_dbm = sa.measure_power_at_frequency(
                        freq_hz=freq_hz,
                        span_hz=cfg.cal_sa_span_hz,
                        rbw_hz=cfg.cal_sa_rbw_hz,
                        ref_level_dbm=cfg.cal_sa_ref_level_dbm,
                        settle_s=cfg.cal_sa_settle_s,
                    )
                    if verbose:
                        corrected = measured_dbm + cfg.cal_sa_power_offset_db
                        print(f"    {power_dbm:+7.2f} dBm → "
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
                if harmonic_measurements and harmonic_measurements[-1]["power_dbm"] == power_dbm:
                    m = harmonic_measurements[-1]["metrics"]
                    row["thd_percent"] = m["thd_percent"]
                    row["fundamental_power_fraction"] = m["fundamental_power_fraction"]
                    for h in harmonic_measurements[-1]["harmonics"]:
                        k = h["harmonic_number"]
                        row[f"h{k}_power_dbm"] = h["power_dbm"]
                        if k >= 2:
                            row[f"h{k}_dbc"] = h["power_dbm"] - measured_dbm

                rows.append(row)

        rf_source.set_output(False)
    finally:
        pass

    cal_df = pd.DataFrame(rows)
    calibration = build_calibration(cal_df, output_dir=out, verbose=verbose)

    if harmonic_measurements:
        harmonics_df = build_harmonics_dataframe(harmonic_measurements)
        thd_df = build_thd_dataframe(harmonic_measurements)

        harmonics_df.to_csv(out / "harmonics.csv", index=False)
        thd_df.to_csv(out / "thd_summary.csv", index=False)

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

    results: List[Dict] = []
    reference_rows: List[Dict] = []

    for freq_hz in cfg.rf_frequencies_hz:
        freq_hz = float(freq_hz)
        if verbose:
            print(f"\n=== RF frequency: {freq_hz/1e9:.6f} GHz ===")

        rf_source.apply(freq_hz=freq_hz, enabled=False)
        time.sleep(cfg.settle_after_rf_change_s)

        t_ref, y_ref, _ = _acquire_with_retry(
            scope, scope_channel,
            timeout_s=cfg.trigger_timeout_s,
            settle_s=cfg.settle_after_scope_single_s,
            max_retries=cfg.scope_read_max_retries, verbose=verbose,
        )
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
                out_png=dirs["refs_dir"] / f"reference_{freq_hz/1e6:.4f}MHz.png",
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
                out_png=dirs["freq_dir"] / f"freq_reference_{freq_hz/1e6:.4f}MHz.png",
                cfg=cfg,
                picked_points=ref_picked,
            )

        if cfg.save_raw_traces_csv:
            pd.DataFrame({"t_s": t_ref, "v": y_ref}).to_csv(
                dirs["raw_dir"] / f"reference_{freq_hz/1e6:.4f}MHz.csv",
                index=False,
            )

        # Enable RF once before the power sweep; subsequent power
        # changes only adjust amplitude, avoiding turn-on transients.
        rf_source.apply(freq_hz=freq_hz, power_dbm=float(cfg.rf_powers_dbm[0]),
                        enabled=True)
        time.sleep(cfg.settle_after_rf_change_s)

        for power_dbm in cfg.rf_powers_dbm:
            power_dbm = float(power_dbm)
            if verbose:
                print(f"  power = {power_dbm:7.3f} dBm")

            rf_source.apply(freq_hz=freq_hz, power_dbm=power_dbm)
            time.sleep(cfg.settle_after_rf_change_s)

            t, y, _ = _acquire_with_retry(
                scope, scope_channel,
                timeout_s=cfg.trigger_timeout_s,
                settle_s=cfg.settle_after_scope_single_s,
                max_retries=cfg.scope_read_max_retries, verbose=verbose,
            )

            meas, picked = measure_trace_against_reference(
                t=t, y_v=y, ref=ref, rf_frequency_hz=freq_hz, cfg=cfg
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
            results.append(row)

            if cfg.save_trace_plots:
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
                )

            if cfg.save_frequency_plots:
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
                )

            if cfg.save_raw_traces_csv:
                pd.DataFrame({"t_s": t, "v": y}).to_csv(
                    dirs["raw_dir"]
                    / f"trace_{freq_hz/1e6:.4f}MHz_{power_dbm:+06.2f}dBm.csv",
                    index=False,
                )

    df = pd.DataFrame(results)
    ref_df = pd.DataFrame(reference_rows)
    fit_df = compute_vpi_fits(df, cfg, output_dir=run_dir)

    df.to_csv(run_dir / "sweep_results.csv", index=False)
    ref_df.to_csv(run_dir / "reference_summary.csv", index=False)
    fit_df.to_csv(run_dir / "vpi_fit_summary.csv", index=False)

    if verbose:
        print("\nSaved to:", run_dir)

    return {"results": df, "references": ref_df, "fits": fit_df}
