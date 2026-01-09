"""Pluto SDR wrapper and connection helpers.

Encapsulates pyadi-iio Pluto interactions and tuning logic. This module must not
import any UI classes to keep SDR operations headless and testable.
"""

from __future__ import annotations

from typing import Optional

import adi
import numpy as np

from pluto_spectrum_analyzer.config import SpectrumConfig


class PlutoSdr:
    """
    Small wrapper around pyadi iio Pluto.

    This isolates SDR specific calls from UI logic.
    """

    def __init__(self, cfg: SpectrumConfig):
        self.cfg = cfg
        self.dev = adi.Pluto(uri=cfg.uri)

        self.dev.rx_enabled_channels = [0]
        self.dev.rx_buffer_size = cfg.fft_size
        self.dev.rx_destroy_buffer()

        self.apply_common()
        self.set_gain_mode(cfg.gain_mode)
        self.set_gain_db(cfg.gain_db)
        self.set_center_hz(cfg.center_hz)

    def close(self) -> None:
        # Explicitly release the device handle when reinitializing.
        self.dev = None

    def apply_common(self) -> None:
        # Apply base sample rate and RF bandwidth settings.
        self.dev.sample_rate = int(self.cfg.sample_rate_hz)
        self.dev.rx_rf_bandwidth = int(self.cfg.rf_bw_hz)

    def set_center_hz(self, hz: int) -> None:
        hz = int(hz)

        # Clamp to Pluto tuning range (325 MHz .. 3.8 GHz).
        if hz < 325_000_000:
            hz = 325_000_000
        if hz > 3_800_000_000:
            hz = 3_800_000_000

        self.dev.rx_lo = hz
        self.cfg.center_hz = hz

    def set_span_hz(self, span_hz: int) -> None:
        span_hz = int(span_hz)

        # Use span as sample rate, keep RF BW within span.
        self.cfg.sample_rate_hz = span_hz
        self.cfg.rf_bw_hz = min(int(span_hz), int(self.cfg.rf_bw_hz))

        self.dev.sample_rate = span_hz
        self.dev.rx_rf_bandwidth = int(self.cfg.rf_bw_hz)

        self.dev.rx_destroy_buffer()

    def set_fft_size(self, n: int, buffer_factor: int) -> None:
        n = int(n)
        self.cfg.fft_size = n
        self.cfg.buffer_factor = int(buffer_factor)
        # RX buffer holds multiple FFTs for overlap/detectors.
        self.dev.rx_buffer_size = n * int(buffer_factor)
        self.dev.rx_destroy_buffer()

    def set_gain_mode(self, mode: str) -> None:
        self.cfg.gain_mode = mode
        self.dev.gain_control_mode_chan0 = mode

    def set_gain_db(self, gain_db: int) -> None:
        gain_db = int(gain_db)
        # Clamp to Pluto-supported gain range (0..70 dB).
        if gain_db < 0:
            gain_db = 0
        if gain_db > 70:
            gain_db = 70

        self.cfg.gain_db = gain_db
        self.dev.rx_hardwaregain_chan0 = gain_db

    def set_rf_bw(self, hz: int) -> None:
        hz = int(hz)
        self.cfg.rf_bw_hz = hz
        self.dev.rx_rf_bandwidth = hz

    def read_rx(self):
        x = self.dev.rx()
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x.astype("complex64")

    @property
    def sample_rate(self) -> float:
        return float(self.dev.sample_rate)

    @property
    def lo(self) -> float:
        return float(self.dev.rx_lo)

    @property
    def rf_bw(self) -> float:
        return float(self.dev.rx_rf_bandwidth)

    @property
    def gain_db(self) -> int:
        return int(self.dev.rx_hardwaregain_chan0)


class NullSdr:
    """Fallback SDR that returns silence when hardware is unavailable."""

    def __init__(self, cfg: SpectrumConfig):
        self.cfg = cfg

    def close(self) -> None:
        return None

    def apply_common(self) -> None:
        return None

    def set_center_hz(self, hz: int) -> None:
        hz = int(hz)
        if hz < 325_000_000:
            hz = 325_000_000
        if hz > 3_800_000_000:
            hz = 3_800_000_000
        self.cfg.center_hz = hz

    def set_span_hz(self, span_hz: int) -> None:
        span_hz = max(1, int(span_hz))
        self.cfg.sample_rate_hz = span_hz
        self.cfg.rf_bw_hz = min(int(span_hz), int(self.cfg.rf_bw_hz))

    def set_fft_size(self, n: int, buffer_factor: int) -> None:
        self.cfg.fft_size = int(n)
        self.cfg.buffer_factor = int(buffer_factor)

    def set_gain_mode(self, mode: str) -> None:
        self.cfg.gain_mode = mode

    def set_gain_db(self, gain_db: int) -> None:
        gain_db = int(gain_db)
        if gain_db < 0:
            gain_db = 0
        if gain_db > 70:
            gain_db = 70
        self.cfg.gain_db = gain_db

    def set_rf_bw(self, hz: int) -> None:
        self.cfg.rf_bw_hz = int(hz)

    def read_rx(self):
        n = max(1, int(self.cfg.fft_size) * int(self.cfg.buffer_factor))
        return np.zeros(n, dtype=np.complex64)

    @property
    def sample_rate(self) -> float:
        return float(max(1, int(self.cfg.sample_rate_hz)))

    @property
    def lo(self) -> float:
        return float(self.cfg.center_hz)

    @property
    def rf_bw(self) -> float:
        return float(self.cfg.rf_bw_hz)

    @property
    def gain_db(self) -> int:
        return int(self.cfg.gain_db)


def test_pluto_connection(uri: str) -> tuple[bool, Optional[str]]:
    """Attempt to create a Pluto connection for a URI."""

    try:
        dev = adi.Pluto(uri=uri)
        _ = dev.sample_rate
    except Exception as exc:  # pragma: no cover - hardware errors vary
        return False, str(exc)
    return True, None
