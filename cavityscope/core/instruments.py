"""Generic instrument interfaces used throughout cavityscope.

These are thin protocol-style wrappers so that the analysis and sweep logic
never depends on a specific hardware driver.  Concrete drivers live in the
separate *hardwarelib* package; you wire them in at notebook / script level.
"""

from __future__ import annotations

from typing import Dict, Optional, Protocol, Tuple

import numpy as np


class ScopeInterface(Protocol):
    """Anything that can acquire and return a waveform."""

    def open(self) -> None: ...
    def close(self) -> None: ...
    def idn(self) -> str: ...

    def acquire_single_and_wait(
        self, timeout_s: float, settle_s: float = 0.0
    ) -> None: ...

    def read_waveform(
        self, channel: int
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]: ...

    def get_timebase(self) -> float: ...
    def set_timebase(self, scale_s_per_div: float) -> None: ...
    def set_trigger_mode(self, mode: str = "AUTO") -> None: ...


class RFSourceInterface(Protocol):
    """Anything that can output an RF tone at a given frequency and power."""

    def open(self) -> None: ...
    def close(self) -> None: ...
    def set_output(self, enabled: bool) -> None: ...

    def apply(
        self,
        freq_hz: float,
        power_dbm: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> None: ...
