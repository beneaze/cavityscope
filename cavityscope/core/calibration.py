"""RF power calibration: map (power_setting_dBm, frequency_hz) → actual Vpk.

Calibration CSV format
----------------------
**Single-frequency** (no ``frequency_hz`` column)::

    power_dbm,vpk_v
    -20.0,0.0032
    -10.0,0.0100
      0.0,0.0316
     10.0,0.1000

**Frequency-dependent** (with ``frequency_hz`` column)::

    frequency_hz,power_dbm,vpk_v
    1.0e9,-20.0,0.0030
    1.0e9,-10.0,0.0098
    1.0e9,  0.0,0.0310
    2.0e9,-20.0,0.0035
    2.0e9,-10.0,0.0105
    2.0e9,  0.0,0.0330

Values between calibration points are linearly interpolated.
Values outside the calibrated range are linearly extrapolated (with a warning).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


class PowerCalibration:
    """Interpolating lookup from (power_dBm [, frequency_hz]) → Vpk.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``power_dbm`` and ``vpk_v`` columns.
        Optionally contains ``frequency_hz`` for per-frequency curves.
    """

    def __init__(self, df: pd.DataFrame):
        required = {"power_dbm", "vpk_v"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Calibration table missing required columns: {missing}. "
                f"Got: {list(df.columns)}"
            )

        self._freq_dependent = "frequency_hz" in df.columns
        self._curves: Dict[float, tuple] = {}

        if self._freq_dependent:
            for freq, grp in df.groupby("frequency_hz"):
                grp_sorted = grp.sort_values("power_dbm")
                self._curves[float(freq)] = (
                    grp_sorted["power_dbm"].to_numpy(dtype=float),
                    grp_sorted["vpk_v"].to_numpy(dtype=float),
                )
            self._freq_keys = np.array(sorted(self._curves.keys()))
        else:
            df_sorted = df.sort_values("power_dbm")
            self._curves[None] = (
                df_sorted["power_dbm"].to_numpy(dtype=float),
                df_sorted["vpk_v"].to_numpy(dtype=float),
            )
            self._freq_keys = None

    # -- Construction helpers --------------------------------------------------

    @classmethod
    def from_csv(cls, path: str | Path) -> "PowerCalibration":
        """Load calibration from a CSV file."""
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        return cls(df)

    @classmethod
    def from_arrays(
        cls,
        power_dbm: np.ndarray,
        vpk_v: np.ndarray,
        frequency_hz: Optional[np.ndarray] = None,
    ) -> "PowerCalibration":
        """Build calibration from numpy arrays (handy in notebooks)."""
        data = {"power_dbm": np.asarray(power_dbm), "vpk_v": np.asarray(vpk_v)}
        if frequency_hz is not None:
            data["frequency_hz"] = np.asarray(frequency_hz)
        return cls(pd.DataFrame(data))

    # -- Lookup ----------------------------------------------------------------

    @staticmethod
    def _interp1d(x_cal: np.ndarray, y_cal: np.ndarray, x_query: float) -> float:
        if x_query < x_cal[0] or x_query > x_cal[-1]:
            warnings.warn(
                f"Power {x_query:.2f} dBm is outside calibration range "
                f"[{x_cal[0]:.2f}, {x_cal[-1]:.2f}] — extrapolating.",
                stacklevel=3,
            )
        return float(np.interp(x_query, x_cal, y_cal))

    def vpk(self, power_dbm: float, frequency_hz: Optional[float] = None) -> float:
        """Return calibrated Vpk for a given power setting (and frequency)."""
        if not self._freq_dependent:
            x, y = self._curves[None]
            return self._interp1d(x, y, power_dbm)

        if frequency_hz is None:
            raise ValueError(
                "Calibration is frequency-dependent but no frequency_hz was given."
            )

        fk = self._freq_keys
        if frequency_hz <= fk[0]:
            x, y = self._curves[fk[0]]
            return self._interp1d(x, y, power_dbm)
        if frequency_hz >= fk[-1]:
            x, y = self._curves[fk[-1]]
            return self._interp1d(x, y, power_dbm)

        idx_hi = int(np.searchsorted(fk, frequency_hz))
        idx_lo = idx_hi - 1
        f_lo, f_hi = fk[idx_lo], fk[idx_hi]

        v_lo = self._interp1d(*self._curves[f_lo], power_dbm)
        v_hi = self._interp1d(*self._curves[f_hi], power_dbm)

        alpha = (frequency_hz - f_lo) / (f_hi - f_lo)
        return float(v_lo + alpha * (v_hi - v_lo))

    def __repr__(self) -> str:
        n = sum(len(v[0]) for v in self._curves.values())
        nf = len(self._curves)
        if self._freq_dependent:
            return f"PowerCalibration({n} points across {nf} frequencies)"
        return f"PowerCalibration({n} points, single frequency)"
