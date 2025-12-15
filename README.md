# Simple Spectrum
A lightweight real time spectrum viewer for the ADALM Pluto SDR using Python, pyadi iio, and pyqtgraph.

This is an FFT based spectrum display (not a swept analyzer). The visible span is determined by the Pluto sample rate and the frequency resolution is determined by FFT size.

## Features
- Real time spectrum plot with center frequency control
- Span control (changes SDR sample rate and RF bandwidth)
- Gain control mode selection (manual, fast attack, slow attack, hybrid)
- Manual gain control when in manual mode
- Auto Y scaling and manual Y scaling (Y min, Y max)
- Hover tooltip showing frequency and magnitude at the nearest FFT bin
- Optional auto span behavior (keeps X range aligned to LO ± SR/2)

## How it works
- **Center frequency** sets the Pluto LO (rx_lo).
- **Span** is the sampled bandwidth and equals the **sample rate**.
  - Visible frequency range is approximately: `LO ± sample_rate / 2`
- **Resolution bandwidth (RBW)** in an FFT analyzer is approximately:
  - `RBW ≈ sample_rate / FFT_size`
- The displayed values are a PSD style magnitude in dB (relative, not calibrated dBm).

## Requirements
- Python 3.9 or newer recommended
- ADALM Pluto SDR connected via USB or Ethernet gadget mode
- Pluto reachable at `ip:192.168.2.1` (default gadget mode)

### Linux packages you may need
On Ubuntu or Debian:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv libusb-1.0-0
