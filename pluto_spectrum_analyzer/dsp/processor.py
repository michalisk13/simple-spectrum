"""DSP primitives for spectrum analysis.

Provides FFT windowing, RBW/ENBW calculations, and spectrogram slice creation.
This module must not import UI or SDR classes; it is purely numerical.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class SpectrumProcessor:
    """
    Handles DSP for the spectrum.
    Provides RBW/ENBW and power spectrum calculation.
    """

    WINDOW_COEFFS = {
        "Hann": ("hann", None),
        "Blackman Harris": ("blackmanharris", None),
        "Flat top": ("flattop", None),
    }

    def __init__(self, fft_size: int, window_name: str):
        self.fft_size = int(fft_size)
        self.window_name = window_name
        self.window = self._make_window(self.fft_size, self.window_name)
        self._update_window_stats()

    def _make_window(self, n: int, name: str) -> np.ndarray:
        # Build window with fixed coefficients to avoid extra dependencies.
        if name == "Hann":
            return np.hanning(n).astype(np.float32)
        if name == "Blackman Harris":
            # Coefficients per Harris 1978, used for low sidelobes.
            a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
            idx = np.arange(n)
            w = (
                a0
                - a1 * np.cos(2.0 * np.pi * idx / (n - 1))
                + a2 * np.cos(4.0 * np.pi * idx / (n - 1))
                - a3 * np.cos(6.0 * np.pi * idx / (n - 1))
            )
            return w.astype(np.float32)
        if name == "Flat top":
            # Flattop coefficients favor amplitude accuracy over sidelobes.
            a0, a1, a2, a3, a4 = 1.0, 1.93, 1.29, 0.388, 0.028
            idx = np.arange(n)
            w = (
                a0
                - a1 * np.cos(2.0 * np.pi * idx / (n - 1))
                + a2 * np.cos(4.0 * np.pi * idx / (n - 1))
                - a3 * np.cos(6.0 * np.pi * idx / (n - 1))
                + a4 * np.cos(8.0 * np.pi * idx / (n - 1))
            )
            return w.astype(np.float32)
        return np.hanning(n).astype(np.float32)

    def _update_window_stats(self) -> None:
        win = self.window
        # Coherent gain for amplitude correction, ENBW for RBW scaling.
        self.coherent_gain = float(np.sum(win) / len(win))
        self.enbw_bins = float(len(win) * np.sum(win**2) / (np.sum(win) ** 2))

    def update_fft_size(self, n: int, window_name: Optional[str] = None) -> None:
        self.fft_size = int(n)
        if window_name is not None:
            self.window_name = window_name
        self.window = self._make_window(self.fft_size, self.window_name)
        self._update_window_stats()

    def rbw_hz(self, fs_hz: float) -> float:
        # RBW is bin width scaled by window ENBW (in bins).
        return (float(fs_hz) / float(self.fft_size)) * self.enbw_bins

    def process_buffer(
        self,
        x: np.ndarray,
        fs_hz: float,
        overlap: float,
        detector: str,
        dc_remove: bool,
        dc_blank_bins: int,
    ) -> Tuple[np.ndarray, int]:
        n = self.fft_size
        if len(self.window) != n:
            # Safety net to prevent crashes if window length drifts from FFT size.
            self.update_fft_size(n, self.window_name)
        if len(x) < n:
            # Pad short buffers to allow FFT without errors.
            pad = np.zeros(n, dtype=np.complex64)
            pad[: len(x)] = x
            x = pad

        if dc_remove:
            # Subtract mean to reduce DC spike before windowing.
            x = x - np.mean(x)

        # Overlap controls how many FFTs are computed per update.
        step = max(int(n * (1.0 - overlap)), 1)
        starts = range(0, len(x) - n + 1, step)

        detector = detector.lower()
        power_accum = None
        count = 0

        for start in starts:
            segment = x[start : start + n]
            if len(segment) != len(self.window):
                # Skip mismatched segments rather than crashing mid-loop.
                continue
            windowed = segment * self.window
            spectrum = np.fft.fftshift(np.fft.fft(windowed))
            # Normalize by coherent gain to keep dBFS stable across windows.
            power = (np.abs(spectrum) / (n * self.coherent_gain)) ** 2

            if dc_blank_bins > 0:
                # Blank bins around DC to suppress LO leakage.
                mid = n // 2
                lo = max(mid - dc_blank_bins, 0)
                hi = min(mid + dc_blank_bins + 1, n)
                power[lo:hi] = np.percentile(power, 10)

            if detector == "sample":
                power_accum = power
                count = 1
                break
            if detector == "peak":
                if power_accum is None:
                    power_accum = power
                else:
                    power_accum = np.maximum(power_accum, power)
                count += 1
                continue
            if detector == "rms":
                if power_accum is None:
                    power_accum = power
                else:
                    power_accum = power_accum + power
                count += 1
                continue

        if power_accum is None:
            power_accum = np.zeros(n, dtype=np.float32)

        if detector == "rms" and count > 0:
            power_accum = power_accum / float(count)

        return power_accum.astype(np.float32), max(count, 1)

    def compute_spectrogram_slice(
        self,
        x: np.ndarray,
        fs_hz: float,
        overlap: float,
        dc_remove: bool,
    ) -> np.ndarray:
        """
        Returns ONE PSD slice (dB) suitable for the spectrogram.
        Uses Welch-style averaging over the buffer.
        """
        n = self.fft_size
        if len(self.window) != n:
            self.update_fft_size(n, self.window_name)
        if len(x) < n:
            pad = np.zeros(n, dtype=np.complex64)
            pad[: len(x)] = x
            x = pad

        if dc_remove:
            x = x - np.mean(x)

        step = max(int(n * (1.0 - overlap)), 1)
        starts = range(0, len(x) - n + 1, step)

        power_accum = None
        count = 0

        for start in starts:
            segment = x[start : start + n]
            if len(segment) != len(self.window):
                continue
            windowed = segment * self.window
            spectrum = np.fft.fftshift(np.fft.fft(windowed))
            power = (np.abs(spectrum) / (n * self.coherent_gain)) ** 2

            if power_accum is None:
                power_accum = power
            else:
                power_accum = power_accum + power
            count += 1

        if power_accum is None:
            power_accum = np.zeros(n, dtype=np.float32)
        elif count > 0:
            power_accum = power_accum / float(count)

        return 10.0 * np.log10(np.maximum(power_accum, 1e-20)).astype(np.float32)

    def compute_spectrogram_peak(
        self,
        x: np.ndarray,
        fs_hz: float,
        overlap: float,
        dc_remove: bool,
    ) -> np.ndarray:
        n = self.fft_size
        if len(self.window) != n:
            self.update_fft_size(n, self.window_name)
        if len(x) < n:
            pad = np.zeros(n, dtype=np.complex64)
            pad[: len(x)] = x
            x = pad

        if dc_remove:
            x = x - np.mean(x)

        step = max(int(n * (1.0 - overlap)), 1)
        starts = range(0, len(x) - n + 1, step)

        power_accum = None
        for start in starts:
            segment = x[start : start + n]
            if len(segment) != len(self.window):
                continue
            windowed = segment * self.window
            spectrum = np.fft.fftshift(np.fft.fft(windowed))
            power = (np.abs(spectrum) / (n * self.coherent_gain)) ** 2
            if power_accum is None:
                power_accum = power
            else:
                power_accum = np.maximum(power_accum, power)

        if power_accum is None:
            power_accum = np.zeros(n, dtype=np.float32)
        return 10.0 * np.log10(np.maximum(power_accum, 1e-20)).astype(np.float32)

    def freqs_hz(self, fs_hz: float, lo_hz: float) -> np.ndarray:
        n = self.fft_size
        f = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / float(fs_hz)))
        return f + float(lo_hz)
