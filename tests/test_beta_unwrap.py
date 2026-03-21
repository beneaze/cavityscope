"""Synthetic tests for multi-branch beta solving and unwrapping."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import jv

from cavityscope.analysis.vpi_fitting import (
    solve_beta_all_branches,
    solve_beta_from_ratio,
    unwrap_beta,
    fit_beta_vs_vpk,
)
from cavityscope.core.config import SweepConfig


# ---------------------------------------------------------------------------
# solve_beta_all_branches
# ---------------------------------------------------------------------------

class TestSolveBetaAllBranches:
    @pytest.mark.parametrize("beta_true", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    def test_round_trip(self, beta_true):
        ratio = (jv(1, beta_true) ** 2) / (jv(0, beta_true) ** 2)
        candidates = solve_beta_all_branches(ratio, max_beta=8.0)
        assert any(
            abs(c - beta_true) < 1e-6 for c in candidates
        ), f"True beta {beta_true} not among candidates {candidates}"

    def test_branch1_matches_original(self):
        for ratio in [0.01, 0.1, 0.5, 1.0, 5.0]:
            original = solve_beta_from_ratio(ratio)
            candidates = solve_beta_all_branches(ratio)
            if np.isfinite(original):
                assert len(candidates) >= 1
                assert abs(candidates[0] - original) < 1e-6

    def test_invalid_inputs(self):
        assert solve_beta_all_branches(-1.0) == []
        assert solve_beta_all_branches(float("nan")) == []
        assert solve_beta_all_branches(float("inf")) == []

    def test_multiple_branches(self):
        ratio = 0.5
        candidates = solve_beta_all_branches(ratio, max_beta=8.0)
        assert len(candidates) >= 3, "Ratio 0.5 should have solutions on 3+ branches"
        assert all(candidates[i] < candidates[i + 1] for i in range(len(candidates) - 1))


# ---------------------------------------------------------------------------
# unwrap_beta
# ---------------------------------------------------------------------------

def _make_sweep(Vpi, voltages, noise_frac=0.0, seed=42):
    """Simulate a voltage sweep returning (voltages, ratios, betas_true)."""
    rng = np.random.RandomState(seed)
    betas_true = math.pi * voltages / Vpi
    if noise_frac > 0:
        carrier = jv(0, betas_true) ** 2 * (1 + noise_frac * rng.randn(len(voltages)))
        sideband = jv(1, betas_true) ** 2 * (1 + noise_frac * rng.randn(len(voltages)))
        ratios = np.abs(sideband / np.maximum(np.abs(carrier), 1e-30))
    else:
        ratios = (jv(1, betas_true) ** 2) / (jv(0, betas_true) ** 2)
    mask = (ratios <= 10.0) & np.isfinite(ratios)
    return voltages[mask], ratios[mask], betas_true[mask]


class TestUnwrapBeta:
    def test_no_folding(self):
        """All points on Branch 1 — unwrapping should be a no-op."""
        voltages = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
        v, r, bt = _make_sweep(Vpi=20.0, voltages=voltages)
        unwrapped = unwrap_beta(v, r)
        np.testing.assert_allclose(unwrapped, bt, atol=1e-4)

    def test_one_crossing(self):
        """Typical case: sweep crosses one branch boundary."""
        voltages = np.array([0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.8, 2.3, 2.8, 4.0, 4.5, 4.8])
        v, r, bt = _make_sweep(Vpi=5.0, voltages=voltages)
        unwrapped = unwrap_beta(v, r)
        np.testing.assert_allclose(unwrapped, bt, atol=1e-3)

    def test_large_gap(self):
        """Points near the carrier null are rejected, leaving a gap."""
        voltages = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 5.5, 6.0])
        v, r, bt = _make_sweep(Vpi=5.0, voltages=voltages)
        unwrapped = unwrap_beta(v, r)
        np.testing.assert_allclose(unwrapped, bt, atol=1e-3)

    def test_small_vpi(self):
        """Most points past Branch 1."""
        voltages = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 2.5, 3.0])
        v, r, bt = _make_sweep(Vpi=2.0, voltages=voltages)
        unwrapped = unwrap_beta(v, r)
        np.testing.assert_allclose(unwrapped, bt, atol=1e-3)

    def test_deep_crossing(self):
        """Sweep crosses 3+ branch boundaries."""
        voltages = np.linspace(0.3, 8.0, 20)
        v, r, bt = _make_sweep(Vpi=3.0, voltages=voltages)
        unwrapped = unwrap_beta(v, r)
        np.testing.assert_allclose(unwrapped, bt, atol=1e-3)

    def test_dense_near_pole(self):
        """Dense sampling through the branch boundary region."""
        voltages = np.arange(0.2, 6.0, 0.2)
        v, r, bt = _make_sweep(Vpi=5.0, voltages=voltages)
        unwrapped = unwrap_beta(v, r)
        np.testing.assert_allclose(unwrapped, bt, atol=1e-3)

    def test_noisy_10pct(self):
        """10% measurement noise — Vpi estimate should be within 5%."""
        voltages = np.array([0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.8, 2.3, 2.8, 4.0, 4.5, 4.8])
        v, r, bt = _make_sweep(Vpi=5.0, voltages=voltages, noise_frac=0.10)
        unwrapped = unwrap_beta(v, r)
        slope = np.polyfit(v, unwrapped, 1)[0]
        Vpi_est = math.pi / slope
        assert abs(Vpi_est - 5.0) / 5.0 < 0.05, f"Vpi estimate {Vpi_est:.3f} too far from 5.0"


# ---------------------------------------------------------------------------
# fit_beta_vs_vpk with unwrapped data
# ---------------------------------------------------------------------------

class TestFitWithUnwrap:
    def test_unwrapped_column_used(self):
        """fit_beta_vs_vpk should prefer beta_unwrapped when present."""
        Vpi = 5.0
        voltages = np.linspace(0.5, 4.8, 10)
        betas_true = math.pi * voltages / Vpi

        df = pd.DataFrame({
            "estimated_vpk_at_load": voltages,
            "beta_est": np.ones_like(voltages),  # garbage
            "beta_unwrapped": betas_true,
            "used_for_vpi_fit": np.ones(len(voltages), dtype=bool),
        })
        cfg = SweepConfig(fit_include_intercept=False)
        result = fit_beta_vs_vpk(df, cfg)
        assert abs(result["fit_vpi_v"] - Vpi) < 0.01

    def test_sigma_clipping(self):
        """Sigma-clipping should reject outliers."""
        Vpi = 5.0
        voltages = np.linspace(0.5, 4.0, 12)
        betas = math.pi * voltages / Vpi
        betas[10] = 0.1  # outlier

        df = pd.DataFrame({
            "estimated_vpk_at_load": voltages,
            "beta_est": betas,
            "used_for_vpi_fit": np.ones(len(voltages), dtype=bool),
        })

        cfg_no_clip = SweepConfig(fit_include_intercept=False, fit_sigma_clip=0.0)
        cfg_clip = SweepConfig(fit_include_intercept=False, fit_sigma_clip=2.5)

        r_no = fit_beta_vs_vpk(df, cfg_no_clip)
        r_yes = fit_beta_vs_vpk(df, cfg_clip)

        assert abs(r_yes["fit_vpi_v"] - Vpi) < abs(r_no["fit_vpi_v"] - Vpi)
        assert abs(r_yes["fit_vpi_v"] - Vpi) < 0.1
