# ADALM-Pluto WiFi 2.4 GHz Spectrum Analyzer

A lightweight Python-based spectrum analyzer for the ADALM-Pluto SDR, focused on real-time analysis of the 2.4 GHz ISM band (WiFi, Bluetooth, and related signals).

The project provides a simple, hackable GUI for visualizing RF activity using a properly normalized FFT / power spectral density (PSD) pipeline built on top of the Analog Devices IIO stack.

This tool is intended for SDR experimentation, RF inspection, and learning purposes rather than calibrated laboratory measurements.

---

## Features

- Real-time spectrum visualization using ADALM-Pluto  
- Focused on the 2.4 GHz ISM band (WiFi channels 1–13)  
- Single-channel RX configuration compatible with stock Pluto firmware  
- Windowed FFT with correct PSD normalization  
- Live frequency axis referenced to RF center frequency  
- Adjustable center frequency and gain  
- Fast PyQt / pyqtgraph GUI  
- Clean and readable DSP pipeline suitable for extension  

---

## Architecture Overview

**Signal chain:**

1. ADALM-Pluto RX (complex baseband samples)  
2. Optional DC offset removal  
3. Windowing (Hann window) to reduce spectral leakage  
4. FFT and frequency shift  
5. Power Spectral Density (PSD) normalization  
6. Logarithmic magnitude display (dB scale)  

Magnitude values are **relative**, not absolute power (dBm).

---

## Requirements

- Python 3.9+
- ADALM-Pluto SDR
- USB or Ethernet connection to Pluto

### Python dependencies

```bash
pip install numpy pyqtgraph pyadi-iio
