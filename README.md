# cavityscope

Toolkit for scanning Fabry-Perot cavity measurements — RF Vpi sweeps, sideband extraction, and modulation-index fitting.

## Structure

```
cavityscope/
├── core/
│   ├── instruments.py      # Generic ScopeInterface / RFSourceInterface protocols
│   ├── config.py           # SweepConfig dataclass with all tuneable parameters
│   └── utils.py            # Shared helpers (dBm conversion, windowing, I/O)
├── analysis/
│   ├── reference.py        # RF-off reference trace analysis (FSR, carrier picking)
│   ├── measurement.py      # Per-trace sideband / carrier integration
│   ├── vpi_fitting.py      # Beta extraction from Bessel ratios, linear Vpi fit
│   └── plotting.py         # Trace + fit plotting
└── sweep.py                # Top-level sweep orchestration (hardware-agnostic)

notebooks/
└── rf_vpi_sweep.ipynb      # Ready-to-run notebook — just plug in your hardware
```

## Design principles

- **Hardware is generic in the library.** The sweep and analysis code depend only on `ScopeInterface` and `RFSourceInterface` (Python `Protocol` classes). Concrete drivers live in the separate [`hardwarelib`](../hardwarelib) package.
- **Specific devices are wired in at notebook level.** The Jupyter notebook imports the concrete driver (e.g. `RigolDHO4000`, `WindfreakSynthHD`) and passes it to the sweep runner. Swapping to a different scope or signal generator requires changing only the import and one constructor call.
- **Configuration is a dataclass.** All parameters live in `SweepConfig`, which has sensible defaults and serialises to JSON.

## Installation

```bash
# Install hardwarelib first (editable)
pip install -e ../hardwarelib

# Then install cavityscope
pip install -e .
```

## Quick start

Open `notebooks/rf_vpi_sweep.ipynb`, set your instrument addresses, and run all cells.

Or from a script:

```python
from hardwarelib.oscilloscopes.rigol import RigolDHO4000
from hardwarelib.signal_generators.windfreak import WindfreakSynthHD
from cavityscope.core import SweepConfig
from cavityscope.sweep import run_sweep

scope = RigolDHO4000("TCPIP0::192.168.1.50::INSTR")
rf = WindfreakSynthHD("COM12", channel=0)

scope.open()
rf.open()
try:
    data = run_sweep(scope, rf, SweepConfig())
finally:
    rf.set_output(False)
    rf.close()
    scope.close()
```
