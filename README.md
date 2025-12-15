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
sudo apt install -y python3 python3-pip python3-venv libusb-1.0-0 "
```

## Tips

### GSM and narrow signals
GSM channels are narrow (200 kHz) and bursty. For better visibility:

- Use a smaller span (for example **1 to 5 MHz**)
- Use **fast_attack** gain mode to detect activity quickly, then switch to **manual** to avoid saturation

Also note that **GSM900 uplink and downlink are different bands**.  
Uplink signals may only appear when a phone nearby is actively transmitting.

---

### Performance
High FFT sizes update a lot of points per frame. If the GUI feels slow:

- Use a smaller FFT size (when that feature is added)
- Increase the update interval
- Use downsampling and clip-to-view  
  (already enabled in the refactored code)

---

## Troubleshooting

### I do not see any signal
Confirm that you can receive samples from the Pluto:

```bash
python -c "import adi; s=adi.Pluto('ip:192.168.2.1'); s.rx_enabled_channels=[0]; s.rx_buffer_size=4096; s.rx_destroy_buffer(); print(len(s.rx()))"
```

## TODO

- Add FFT size selector to control **RBW (Resolution Bandwidth)**
- Display computed **RBW** in the status line
- Add averaging and **VBW-style smoothing** (EMA over frames)
- Add **peak hold** and **max hold** traces
- Add **marker support** (click to place marker, delta markers)
- Add **waterfall view**
- Add presets for common bands  
  - WiFi 2.4  
  - GSM900 uplink  
  - GSM900 downlink  
  - LTE bands
- Add calibration option  
  - Relative dB to dBFS  
  - Approximate dBm using external calibration
- Add **CSV export** of current trace
- Add **headless mode** for logging



