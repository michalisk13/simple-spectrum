# Pluto Spectrum Analyzer

## Overview
Pluto Spectrum Analyzer is a real-time FFT spectrum viewer for the ADALM-Pluto SDR. It provides an instrument-style UI for live spectrum, spectrogram, markers, and trace analysis while preserving calibration and state between runs. The application currently targets the Pluto SDR and its pyadi-iio interface.

## Features
- Live spectrum display with center/span controls
- RBW derived from FFT size and window ENBW
- VBW smoothing using an EMA in linear power
- Markers, peak table, and presets
- Spectrogram modes and LUT controls
- State persistence for UI and settings
- Calibration support (dBFS to dBm)

## Requirements
- Python 3.9+
- Packages: pyadi-iio, numpy, pyqtgraph, PyQt5/PyQt6
- OS: Windows and Linux are supported (Linux recommended for USB gadget mode)

## Install
1. Create a virtual environment (recommended).
2. Install dependencies:
   ```bash
   python -m pip install pyadi-iio numpy pyqtgraph PyQt6
   ```
3. Ensure the Pluto SDR is reachable via USB or Ethernet gadget mode.

## How to run
```bash
python spectrum-monitor.py
```

## Usage guide
### Typical workflows
- **Fast view**: use the Fast View preset for responsive updates while tuning.
- **Wide scan**: switch to Wide Scan to see more bandwidth at once.
- **Measure**: use the Measure preset and calibration for repeatable readings.

### RBW selection
RBW is derived from FFT size and the selected window. The RBW dropdown either auto-selects the FFT size or lets you pick a specific RBW that maps to the nearest FFT size for the current span.

### Update rate
The update rate controls UI refresh and worker pacing. Higher rates increase CPU usage and can reduce sweep stability; lower rates improve stability at the cost of responsiveness.

### Spectrogram interpretation and settings
The spectrogram shows time on the Y axis and frequency across X. Adjust the speed (rows/sec) and depth (seconds) to trade between history and responsiveness. LUT and scale options control contrast and dynamic range.

## SDR connection configuration
### Setting the URI
Open **File → SDR Settings…** to enter a URI (for example, `ip:192.168.2.1`). The last used URI and a short history are saved in state.

### Unreachable device
If the Pluto is not reachable, the test button will report the failure. The application continues running so you can correct the URI and reconnect without restarting.

## Troubleshooting
- **No device found**: confirm USB/Ethernet connectivity and that the Pluto responds to `ip:192.168.2.1` (or your custom URI).
- **Aliasing or unexpected span**: the visible span equals the sample rate; verify the span and RF bandwidth settings.
- **DC spike**: enable DC removal and adjust DC blanking to reduce LO leakage.
- **Gaps in trace**: reduce overlap or FFT size if the worker cannot keep up.
- **Performance tips**: lower FFT size, lower update rate, and disable the spectrogram for best performance.

## Roadmap
- Waterfall rendering improvements
- Multi-receiver support
- IQ recording
- Calibration assistant
