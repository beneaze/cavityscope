"""Beta extraction from Bessel ratios and linear Vpi fitting."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import jv

from cavityscope.core.config import SweepConfig


def solve_beta_from_ratio(ratio: float) -> float:
    """Solve J1(beta)^2 / J0(beta)^2 = ratio on the first branch (0 < beta < ~2.405)."""
    if not np.isfinite(ratio) or ratio < 0:
        return float("nan")

    def f(beta: float) -> float:
        return (jv(1, beta) ** 2) / (jv(0, beta) ** 2) - ratio

    lo = 1e-9
    hi = 2.4048255577 - 1e-4
    try:
        if np.sign(f(lo)) == np.sign(f(hi)):
            return float("nan")
        return float(brentq(f, lo, hi, maxiter=200))
    except Exception:
        return float("nan")


def fit_beta_vs_vpk(dfg: pd.DataFrame, cfg: SweepConfig) -> Dict[str, float]:
    """Fit beta = slope * Vpk (+ intercept) and derive Vpi = pi / slope."""
    dff = dfg[dfg["used_for_vpi_fit"]].copy()
    out: Dict[str, float] = {
        "n_fit_points": len(dff),
        "fit_include_intercept": float(cfg.fit_include_intercept),
        "fit_slope_beta_per_v": float("nan"),
        "fit_intercept_beta": float("nan"),
        "fit_r2": float("nan"),
        "fit_vpi_v": float("nan"),
    }
    if len(dff) < cfg.min_points_for_vpi_fit:
        return out

    x = dff["estimated_vpk_at_load"].to_numpy(dtype=float)
    y = dff["beta_est"].to_numpy(dtype=float)

    if cfg.fit_include_intercept:
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
    else:
        slope = float(np.dot(x, y) / max(np.dot(x, x), 1e-30))
        intercept = 0.0
        yhat = slope * x

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = (
        float(np.sum((y - np.mean(y)) ** 2))
        if cfg.fit_include_intercept
        else float(np.sum(y**2))
    )
    r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    vpi = float("nan") if slope <= 0 else math.pi / slope

    out.update(
        {
            "n_fit_points": len(dff),
            "fit_slope_beta_per_v": float(slope),
            "fit_intercept_beta": float(intercept),
            "fit_r2": r2,
            "fit_vpi_v": vpi,
        }
    )
    return out
