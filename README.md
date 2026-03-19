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

## How Vpi is extracted from the modulation-index fit

The half-wave voltage \(V_\pi\) is the voltage at which the electro-optic
phase modulator imparts a phase shift of \(\pi\) radians. `cavityscope`
extracts it from cavity transmission traces in four stages.

### 1. Reference trace (RF off)

With the RF drive off, a scope trace of the scanning Fabry–Pérot
transmission is acquired. Peak-finding on the smoothed, baseline-subtracted
signal locates the cavity resonances (carriers). The free spectral range
(FSR) in time, \(\Delta t_\text{FSR}\), is taken as the median spacing
between adjacent peaks.

### 2. Sideband and carrier integration (RF on)

For each RF power/frequency setting the modulator is driven and a new scope
trace is captured. The first-order sidebands appear at time offsets

\[
\delta t = \Delta t_\text{FSR}\;\frac{f_\text{RF}}{f_\text{FSR}}
\]

from the carrier. Integration windows (configurable in Hz, mapped to time
via the FSR) are placed around the carrier and the \(\pm 1\) sidebands.
The integrated area of each peak is computed with the trapezoidal rule on
the smoothed trace after baseline subtraction.

### 3. Modulation index (beta) from the Bessel ratio

For a pure phase modulator driven at modulation index \(\beta\), the
electric field acquires sidebands whose amplitudes are Bessel functions of
the first kind. The carrier power scales as \(J_0(\beta)^2\) and each
first-order sideband as \(J_1(\beta)^2\). The measured ratio of sideband
area to carrier area therefore satisfies

\[
R \;=\; \frac{A_\text{sideband}}{A_\text{carrier}}
     \;=\; \frac{J_1(\beta)^2}{J_0(\beta)^2}.
\]

This equation is numerically inverted for \(\beta\) on the first monotonic
branch \(0 < \beta < 2.4048\) (the first zero of \(J_0\)) using Brent's
root-finding method (`solve_beta_from_ratio` in `vpi_fitting.py`).

Points are excluded from the subsequent fit when:

- the ratio falls outside `[min_ratio_for_beta, max_ratio_for_beta]`,
- the sideband signal-to-noise ratio is below `min_sideband_area_snr`, or
- the root-finder returns no finite solution.

### 4. Linear fit and Vpi extraction

By definition of the half-wave voltage,

\[
\beta \;=\; \frac{\pi\, V_\text{pk}}{V_\pi},
\]

so \(\beta\) is linear in the peak drive voltage \(V_\text{pk}\) with slope
\(m = \pi / V_\pi\). A linear fit (with or without intercept, controlled by
`fit_include_intercept`) of the surviving \((\hat V_\text{pk},\,\hat\beta)\)
points yields the slope, from which

\[
V_\pi \;=\; \frac{\pi}{m}.
\]

The fit quality is reported as \(R^2\), and the result is stored per RF
frequency in the `vpi_fit_summary.csv` output.

> **Voltage estimation.** \(V_\text{pk}\) at the modulator is either
> calculated analytically from the set-point dBm
> (\(V_\text{rms} = \sqrt{P \cdot R}\), \(V_\text{pk} = \sqrt{2}\,V_\text{rms}\))
> or looked up from an optional power-calibration table that maps
> `(power_dbm, frequency_hz)` to measured \(V_\text{pk}\).

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
