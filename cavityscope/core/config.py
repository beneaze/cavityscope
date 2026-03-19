"""Sweep configuration container with sensible defaults."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np


@dataclass
class SweepConfig:
    """All tuneable parameters for a cavity Vpi sweep.

    Construct with overrides::

        cfg = SweepConfig(
            rf_frequencies_hz=[1e9, 2e9],
            rf_powers_dbm=list(np.arange(-20, 10, 2)),
        )
    """

    # -- Sweep grid ------------------------------------------------------------
    rf_frequencies_hz: List[float] = field(
        default_factory=lambda: np.arange(0.4e9, 5.1e9, 0.5e9).tolist()
    )
    rf_powers_dbm: List[float] = field(
        default_factory=lambda: np.arange(-20.0, 10.1, 2.0).tolist()
    )
    settle_after_rf_change_s: float = 0.15
    settle_after_scope_single_s: float = 0.05
    trigger_timeout_s: float = 4.0
    scope_read_max_retries: int = 3

    # -- Cavity / reference extraction -----------------------------------------
    cavity_fsr_hz: float = 1.5e9
    fsr_time_s: Optional[float] = None
    analysis_use_gaussian_sigma_pts: float = 2.0
    reference_peak_prominence_fraction: float = 0.08
    min_peak_distance_fraction_of_trace: float = 0.01
    baseline_percentile: float = 10.0
    edge_exclusion_fraction_of_trace: float = 0.05
    min_reference_peaks_required: int = 2
    reference_carrier_pick: str = "highest"

    # -- Integration windows (in frequency domain) -----------------------------
    carrier_window_hz: float = 120e6
    sideband_window_hz: float = 80e6

    # -- Sideband selection ----------------------------------------------------
    sideband_mode: str = "both"

    # -- Beta extraction / fit filtering ---------------------------------------
    min_sideband_area_snr: float = 3.0
    min_ratio_for_beta: float = 1e-6
    max_ratio_for_beta: float = 10.0
    fit_include_intercept: bool = True
    min_points_for_vpi_fit: int = 3

    # -- Optional Vpi estimation -----------------------------------------------
    compute_vpi: bool = True
    net_power_offset_db: float = 0.0
    assumed_load_ohm: float = 50.0

    # -- Power calibration (optional) ------------------------------------------
    # Path to a CSV with columns: power_dbm, vpk_v (and optionally frequency_hz).
    # When set, calibrated voltages replace the analytical dBm-to-V conversion.
    power_calibration_csv: Optional[str] = None

    # -- Scope-based power calibration (optional, runs as a separate phase) ----
    # Channel where the RF output is monitored (high-Z probe before modulator).
    # Set to None to skip.  When set, run_power_calibration() sweeps the same
    # frequency/power grid, sets the scope timebase for the RF frequency,
    # measures Vpk via sine fit, and returns a PowerCalibration object.
    cal_scope_channel: Optional[int] = None
    cal_cycles_to_capture: int = 20
    cal_settle_s: float = 0.15

    # -- Output ----------------------------------------------------------------
    output_dir: str = "vpi_sweep_output"
    save_trace_plots: bool = True
    save_reference_plots: bool = True
    save_raw_traces_csv: bool = True
    save_frequency_plots: bool = True

    def to_dict(self) -> dict:
        return asdict(self)
