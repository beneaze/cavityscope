"""Per-trace sideband / carrier measurement against a reference."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from cavityscope.analysis.reference import ReferenceInfo
from cavityscope.analysis.vpi_fitting import solve_beta_from_ratio
from cavityscope.core.config import SweepConfig
from cavityscope.core.utils import (
    dbm_to_vrms_into_r,
    hz_window_to_half_time,
    robust_baseline,
    robust_noise_sigma,
)

if TYPE_CHECKING:
    from cavityscope.core.calibration import PowerCalibration


def _smooth_and_baseline(y_v: np.ndarray, cfg: SweepConfig):
    baseline = robust_baseline(y_v, cfg.baseline_percentile)
    y_bs = y_v - baseline
    sigma = cfg.analysis_use_gaussian_sigma_pts
    y_sm = gaussian_filter1d(y_bs, sigma=sigma) if sigma > 0 else y_bs.copy()
    return baseline, y_bs, y_sm


def _integrate_window(
    t: np.ndarray,
    y: np.ndarray,
    center_t: float,
    half_window_s: float,
) -> Tuple[float, float, int]:
    mask = (t >= center_t - half_window_s) & (t <= center_t + half_window_s)
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        raise ValueError("Integration window outside trace or too small.")
    _trapz = getattr(np, "trapezoid", np.trapz)
    area = float(_trapz(y[idx], t[idx]))
    center_height = float(np.interp(center_t, t, y))
    return area, center_height, int(idx.size)


def measure_trace_against_reference(
    t: np.ndarray,
    y_v: np.ndarray,
    ref: ReferenceInfo,
    rf_frequency_hz: float,
    cfg: SweepConfig,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Measure carrier and sideband areas for one RF-on trace."""
    baseline, y_bs, y_sm = _smooth_and_baseline(y_v, cfg)
    fsr_hz = cfg.cavity_fsr_hz

    if rf_frequency_hz >= 0.5 * fsr_hz:
        raise ValueError(
            f"rf_frequency_hz={rf_frequency_hz:g} exceeds FSR/2={0.5*fsr_hz:g}."
        )

    carrier_t = ref.chosen_carrier_time_s
    dt_sb = ref.fsr_time_s * (rf_frequency_hz / fsr_hz)
    sb_minus_t = carrier_t - dt_sb
    sb_plus_t = carrier_t + dt_sb

    carrier_hw = hz_window_to_half_time(
        cfg.carrier_window_hz, fsr_hz, ref.fsr_time_s
    )
    sideband_hw = hz_window_to_half_time(
        cfg.sideband_window_hz, fsr_hz, ref.fsr_time_s
    )

    carrier_area, carrier_h, carrier_n = _integrate_window(
        t, y_sm, carrier_t, carrier_hw
    )
    sb_m_area, sb_m_h, sb_m_n = _integrate_window(
        t, y_sm, sb_minus_t, sideband_hw
    )
    sb_p_area, sb_p_h, sb_p_n = _integrate_window(
        t, y_sm, sb_plus_t, sideband_hw
    )

    noise_sigma = robust_noise_sigma(y_bs)
    dt_mean = float(np.median(np.diff(t)))
    carrier_noise = noise_sigma * math.sqrt(max(carrier_n, 1)) * dt_mean
    sb_m_noise = noise_sigma * math.sqrt(max(sb_m_n, 1)) * dt_mean
    sb_p_noise = noise_sigma * math.sqrt(max(sb_p_n, 1)) * dt_mean

    mode = cfg.sideband_mode.lower()
    if mode == "plus":
        sb_area, sb_noise = sb_p_area, sb_p_noise
    elif mode == "minus":
        sb_area, sb_noise = sb_m_area, sb_m_noise
    else:
        sb_area = 0.5 * (sb_p_area + sb_m_area)
        sb_noise = 0.5 * math.sqrt(sb_p_noise**2 + sb_m_noise**2)

    ratio = sb_area / max(carrier_area, 1e-30)
    beta = solve_beta_from_ratio(ratio)
    sb_snr = sb_area / max(sb_noise, 1e-30)

    used = (
        np.isfinite(beta)
        and ratio >= cfg.min_ratio_for_beta
        and ratio <= cfg.max_ratio_for_beta
        and sb_snr >= cfg.min_sideband_area_snr
    )

    row: Dict[str, float] = {
        "baseline_v": baseline,
        "carrier_time_s": carrier_t,
        "sb_minus_time_s": sb_minus_t,
        "sb_plus_time_s": sb_plus_t,
        "dt_sideband_s": dt_sb,
        "carrier_area_v_s": carrier_area,
        "sb_minus_area_v_s": sb_m_area,
        "sb_plus_area_v_s": sb_p_area,
        "selected_sideband_area_v_s": sb_area,
        "carrier_center_height_v": carrier_h,
        "sb_minus_center_height_v": sb_m_h,
        "sb_plus_center_height_v": sb_p_h,
        "carrier_noise_area_est_v_s": carrier_noise,
        "sb_minus_noise_area_est_v_s": sb_m_noise,
        "sb_plus_noise_area_est_v_s": sb_p_noise,
        "selected_sideband_area_snr": sb_snr,
        "sideband_mode_used": mode,
        "sb_to_carrier_ratio": ratio,
        "beta_est": beta,
        "carrier_window_hz": float(cfg.carrier_window_hz),
        "sideband_window_hz": float(cfg.sideband_window_hz),
        "carrier_window_s": 2.0 * carrier_hw,
        "sideband_window_s": 2.0 * sideband_hw,
        "used_for_vpi_fit": bool(used),
    }

    points = {
        "carrier_times_s": np.asarray([carrier_t]),
        "carrier_heights_v": np.asarray([carrier_h]),
        "sb_minus_times_s": np.asarray([sb_minus_t]),
        "sb_minus_heights_v": np.asarray([sb_m_h]),
        "sb_plus_times_s": np.asarray([sb_plus_t]),
        "sb_plus_heights_v": np.asarray([sb_p_h]),
    }
    return row, points


def add_voltage_columns(
    row: Dict[str, float],
    rf_power_dbm: float,
    cfg: SweepConfig,
    calibration: Optional["PowerCalibration"] = None,
    rf_frequency_hz: Optional[float] = None,
) -> Dict[str, float]:
    """Append voltage columns — calibrated when available, analytical otherwise.

    When *calibration* is provided the Vpk comes from the interpolated
    calibration table.  Otherwise the nominal ``dBm → Vrms → Vpk`` conversion
    is used (subject to ``net_power_offset_db`` and ``assumed_load_ohm``).
    """
    delivered_dbm = rf_power_dbm + cfg.net_power_offset_db

    if calibration is not None:
        vpk = calibration.vpk(rf_power_dbm, frequency_hz=rf_frequency_hz)
        vrms = vpk / math.sqrt(2.0)
        row["voltage_source"] = "calibration"
    else:
        vrms = dbm_to_vrms_into_r(delivered_dbm, cfg.assumed_load_ohm)
        vpk = math.sqrt(2.0) * vrms
        row["voltage_source"] = "analytical"

    vpp = 2.0 * vpk
    row["estimated_delivered_dbm"] = delivered_dbm
    row["estimated_vrms_at_load"] = vrms
    row["estimated_vpk_at_load"] = vpk
    row["estimated_vpp_at_load"] = vpp
    return row
