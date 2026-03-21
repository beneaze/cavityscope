"""Beta extraction from Bessel ratios, branch unwrapping, and linear Vpi fitting."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import jv, jn_zeros

from cavityscope.core.config import SweepConfig

_J0_ZEROS = jn_zeros(0, 5)
_J1_ZEROS = jn_zeros(1, 5)
_BRANCH_BOUNDARIES = sorted([0.0] + _J0_ZEROS.tolist() + _J1_ZEROS.tolist())


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


def solve_beta_all_branches(ratio: float, max_beta: float = 8.0) -> List[float]:
    """Return all beta values satisfying J1(b)^2/J0(b)^2 = ratio up to *max_beta*.

    The ratio diverges at every zero of J0 and vanishes at every zero of J1,
    creating alternating monotonic branches.  This function searches each
    branch with ``brentq`` and returns every solution found, sorted ascending.
    """
    if not np.isfinite(ratio) or ratio < 0:
        return []

    eps = 1e-6
    solutions: List[float] = []
    for i in range(len(_BRANCH_BOUNDARIES) - 1):
        lo = _BRANCH_BOUNDARIES[i] + eps
        hi = _BRANCH_BOUNDARIES[i + 1] - eps
        if lo >= max_beta:
            break
        hi = min(hi, max_beta)
        if hi <= lo:
            continue

        def f(beta: float, _r=ratio) -> float:
            return (jv(1, beta) ** 2) / (jv(0, beta) ** 2) - _r

        try:
            fa, fb = f(lo), f(hi)
            if np.sign(fa) != np.sign(fb):
                sol = brentq(f, lo, hi, maxiter=200)
                solutions.append(float(sol))
        except Exception:
            pass
    return solutions


# ---------------------------------------------------------------------------
# Beta unwrapping
# ---------------------------------------------------------------------------

def _build_safe_zone(
    v_sorted: np.ndarray,
    b1_sorted: np.ndarray,
    mono_tol: float = 0.05,
    fit_n_sigma: float = 3.0,
    min_fit_pts: int = 3,
) -> List[int]:
    """Identify the initial voltage-sorted segment that is unambiguously on
    Branch 1, using monotonicity *and* running-fit consistency.

    Returns indices into the sorted arrays.
    """
    safe_idx: List[int] = []
    safe_v: List[float] = []
    safe_b: List[float] = []
    running_max = -np.inf

    for i, (v, b) in enumerate(zip(v_sorted, b1_sorted)):
        if not np.isfinite(b):
            continue

        if b < running_max - mono_tol:
            break

        if len(safe_v) >= min_fit_pts:
            coeffs = np.polyfit(safe_v, safe_b, 1)
            predicted = coeffs[0] * v + coeffs[1]
            residuals = np.array(safe_b) - (
                coeffs[0] * np.array(safe_v) + coeffs[1]
            )
            sigma = max(float(np.std(residuals)), 0.05)
            if abs(b - predicted) > fit_n_sigma * sigma:
                break

        safe_idx.append(i)
        safe_v.append(float(v))
        safe_b.append(float(b))
        running_max = max(running_max, b)

    return safe_idx


def unwrap_beta(
    voltages: np.ndarray,
    ratios: np.ndarray,
    max_beta: float = 8.0,
    mono_tol: float = 0.05,
    fit_n_sigma: float = 3.0,
) -> np.ndarray:
    """Resolve the Bessel branch ambiguity in *ratios* using voltage ordering.

    1. Compute all-branch candidate betas for every ratio.
    2. Build a safe zone of low-voltage points unambiguously on Branch 1
       (via monotonicity + running-fit consistency).
    3. Preliminary linear fit on the safe zone.
    4. For every point, pick the candidate closest to the preliminary
       prediction.

    Returns an array of unwrapped beta values (same length as *voltages*).
    """
    n = len(voltages)
    all_cands = [solve_beta_all_branches(r, max_beta) for r in ratios]
    beta_b1 = np.array(
        [c[0] if c else np.nan for c in all_cands], dtype=float
    )

    order = np.argsort(voltages)
    v_sorted = voltages[order]
    b1_sorted = beta_b1[order]

    safe_sorted = _build_safe_zone(
        v_sorted, b1_sorted, mono_tol, fit_n_sigma
    )
    safe_orig = [int(order[i]) for i in safe_sorted]

    if len(safe_orig) < 2:
        return beta_b1.copy()

    safe_v = voltages[safe_orig]
    safe_b = beta_b1[safe_orig]
    coeffs = np.polyfit(safe_v, safe_b, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    if slope <= 0:
        return beta_b1.copy()

    unwrapped = np.empty(n, dtype=float)
    for j in range(n):
        predicted = slope * voltages[j] + intercept
        cands = all_cands[j]
        if cands:
            unwrapped[j] = min(cands, key=lambda c: abs(c - predicted))
        else:
            unwrapped[j] = np.nan
    return unwrapped


# ---------------------------------------------------------------------------
# Linear Vpi fit (with optional sigma-clipping)
# ---------------------------------------------------------------------------

def _linear_fit(x, y, include_intercept):
    """Return (slope, intercept, yhat)."""
    if include_intercept:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope = float(np.dot(x, y) / max(np.dot(x, x), 1e-30))
        intercept = 0.0
    yhat = slope * x + intercept
    return float(slope), float(intercept), yhat


def fit_beta_vs_vpk(dfg: pd.DataFrame, cfg: SweepConfig) -> Dict[str, float]:
    """Fit beta = slope * Vpk (+ intercept) and derive Vpi = pi / slope."""
    beta_col = (
        "beta_unwrapped"
        if "beta_unwrapped" in dfg.columns
        else "beta_est"
    )

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
    y = dff[beta_col].to_numpy(dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < cfg.min_points_for_vpi_fit:
        return out

    slope, intercept, yhat = _linear_fit(x, y, cfg.fit_include_intercept)

    # --- Iterative sigma-clipping ---
    clip = cfg.fit_sigma_clip
    if clip > 0 and len(x) > cfg.min_points_for_vpi_fit:
        for _ in range(cfg.fit_sigma_clip_max_iter):
            residuals = y - yhat
            sigma = max(float(np.std(residuals)), 1e-12)
            keep = np.abs(residuals) <= clip * sigma
            if keep.all() or keep.sum() < cfg.min_points_for_vpi_fit:
                break
            x, y = x[keep], y[keep]
            slope, intercept, yhat = _linear_fit(
                x, y, cfg.fit_include_intercept
            )

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
            "n_fit_points": int(len(x)),
            "fit_slope_beta_per_v": slope,
            "fit_intercept_beta": intercept,
            "fit_r2": r2,
            "fit_vpi_v": vpi,
        }
    )
    return out
