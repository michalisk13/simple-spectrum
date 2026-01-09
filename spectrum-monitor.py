"""
Measurement notes
-----------------
RBW (resolution bandwidth) in this implementation is derived from the FFT bin width
and the window's equivalent noise bandwidth (ENBW). For a window w[n] with length N:
    ENBW_bins = N * sum(w^2) / (sum(w)^2)
    RBW_Hz = (fs / N) * ENBW_bins
VBW (video bandwidth) is implemented as an exponential moving average (EMA) applied
in the linear power domain after detection. The EMA coefficient is based on the time
between updates (dt) and VBW: alpha = exp(-dt * 2π * VBW).
Amplitude modes:
    dBFS: Power referenced to full-scale (FS) ADC power.
    dBm: Requires calibration offset (dBFS -> dBm) and optional external gain/loss.
    dBm/Hz: Same as dBm but normalized by RBW (per-Hz).
Limitations: PlutoSDR absolute accuracy depends on gain linearity, front-end
calibration, and frequency response. External calibration is recommended.
"""

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph import SignalProxy
from pyqtgraph.exporters import ImageExporter
from pyqtgraph.Qt import QtCore, QtWidgets
import adi


CAL_PATH = os.path.join(os.path.dirname(__file__), "spectrum-monitor-calibration.json")
STATE_PATH = os.path.join(os.path.dirname(__file__), "spectrum-monitor-state.json")


@dataclass
class SpectrumConfig:
    """
    Configuration for the spectrum viewer.

    Notes
    Span is the sampled bandwidth.
    For a real time FFT spectrum, visible span equals sample rate.
    """

    uri: str = "ip:192.168.2.1"

    # Center frequency for the LO.
    center_hz: int = 2_437_000_000

    # Sample rate controls instantaneous span. RF BW should be >= sample rate.
    sample_rate_hz: int = 20_000_000
    rf_bw_hz: int = 20_000_000

    # Gain settings.
    gain_db: int = 55
    gain_mode: str = "manual"

    # FFT and update timing.
    fft_size: int = 8192
    update_ms: int = 100
    buffer_factor: int = 4
    overlap: float = 0.5

    # UI responsiveness (hover).
    hover_rate_hz: int = 25

    # RBW controls.
    rbw_mode: str = "Manual"
    rbw_hz: float = 4894.0
    window: str = "Blackman Harris"

    # VBW controls.
    vbw_mode: str = "Auto"
    vbw_hz: float = 3_000.0

    # Detector/trace controls.
    detector: str = "RMS"
    trace_type: str = "Clear Write"
    trace2_enabled: bool = False
    avg_count: int = 10
    avg_mode: str = "RMS"

    # Display scaling.
    ref_level_db: float = 0.0
    display_range_db: float = 100.0

    # Spectrogram defaults.
    spectrogram_mode: str = "PSD (Welch)"

    # Measurement helpers.
    measurement_mode: bool = False
    dc_remove: bool = True
    dc_blank_bins: int = 1


@dataclass
class Calibration:
    dbfs_to_dbm_offset: Optional[float] = None
    external_gain_db: float = 0.0

    @classmethod
    def load(cls) -> "Calibration":
        # Load calibration from JSON (if present); tolerate parse errors.
        if not os.path.exists(CAL_PATH):
            return cls()
        try:
            with open(CAL_PATH, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return cls()
        return cls(
            dbfs_to_dbm_offset=data.get("dbfs_to_dbm_offset"),
            external_gain_db=float(data.get("external_gain_db", 0.0)),
        )

    def save(self) -> None:
        # Persist calibration for future runs.
        data = {
            "dbfs_to_dbm_offset": self.dbfs_to_dbm_offset,
            "external_gain_db": self.external_gain_db,
        }
        with open(CAL_PATH, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def has_calibration(self) -> bool:
        return self.dbfs_to_dbm_offset is not None


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

    def apply_common(self):
        # Apply base sample rate and RF bandwidth settings.
        self.dev.sample_rate = int(self.cfg.sample_rate_hz)
        self.dev.rx_rf_bandwidth = int(self.cfg.rf_bw_hz)

    def set_center_hz(self, hz: int):
        hz = int(hz)

        # Clamp to Pluto tuning range.
        if hz < 325_000_000:
            hz = 325_000_000
        if hz > 3_800_000_000:
            hz = 3_800_000_000

        self.dev.rx_lo = hz
        self.cfg.center_hz = hz

    def set_span_hz(self, span_hz: int):
        span_hz = int(span_hz)

        # Use span as sample rate, keep RF BW at least as wide.
        self.cfg.sample_rate_hz = span_hz
        self.cfg.rf_bw_hz = max(int(span_hz), int(self.cfg.rf_bw_hz))

        self.dev.sample_rate = span_hz
        self.dev.rx_rf_bandwidth = int(self.cfg.rf_bw_hz)

        self.dev.rx_destroy_buffer()

    def set_fft_size(self, n: int, buffer_factor: int):
        n = int(n)
        self.cfg.fft_size = n
        self.cfg.buffer_factor = int(buffer_factor)
        # RX buffer holds multiple FFTs for overlap/detectors.
        self.dev.rx_buffer_size = n * int(buffer_factor)
        self.dev.rx_destroy_buffer()

    def set_gain_mode(self, mode: str):
        self.cfg.gain_mode = mode
        self.dev.gain_control_mode_chan0 = mode

    def set_gain_db(self, gain_db: int):
        gain_db = int(gain_db)
        # Clamp to Pluto-supported gain range.
        if gain_db < 0:
            gain_db = 0
        if gain_db > 70:
            gain_db = 70

        self.cfg.gain_db = gain_db
        self.dev.rx_hardwaregain_chan0 = gain_db

    def set_rf_bw(self, hz: int):
        hz = int(hz)
        self.cfg.rf_bw_hz = hz
        self.dev.rx_rf_bandwidth = hz

    def read_rx(self) -> np.ndarray:
        x = self.dev.rx()
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x.astype(np.complex64)

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

    def update_fft_size(self, n: int, window_name: Optional[str] = None):
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


class HoverReadout:
    """
    Keeps lightweight state needed for fast O(1) hover lookup.
    """

    def __init__(self):
        self.mag: Optional[np.ndarray] = None
        self.f0: Optional[float] = None
        self.df: Optional[float] = None
        self.n: Optional[int] = None
        self.last_idx: Optional[int] = None

    def update_axis(self, freqs: np.ndarray, mag: np.ndarray):
        self.mag = mag
        self.f0 = float(freqs[0])
        self.df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else None
        self.n = int(len(freqs))
        self.last_idx = None

    def nearest_bin(self, fx: float) -> Optional[Tuple[float, float, int]]:
        if self.mag is None or self.f0 is None or self.df is None or self.n is None:
            return None

        idx = int((float(fx) - self.f0) / self.df)
        if idx < 0:
            idx = 0
        elif idx >= self.n:
            idx = self.n - 1

        if self.last_idx is not None and idx == self.last_idx:
            return None

        self.last_idx = idx
        f_bin = self.f0 + idx * self.df
        m_bin = float(self.mag[idx])
        return f_bin, m_bin, idx


class SpectrumViewBox(pg.ViewBox):
    def __init__(self, owner: "SpectrumWindow"):
        super().__init__(enableMenu=False)
        self.owner = owner

    def wheelEvent(self, ev):
        if ev.modifiers() & QtCore.Qt.ShiftModifier:
            # Shift+wheel zooms span around the cursor without changing Y scale.
            delta = ev.angleDelta().y() if hasattr(ev, "angleDelta") else ev.delta()
            factor = 0.9 if delta > 0 else 1.1
            scene_pos = ev.scenePos()
            mouse_point = self.mapSceneToView(scene_pos)
            self.owner.zoom_span_at(float(mouse_point.x()), factor)
            ev.accept()
            return
        super().wheelEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        # Double-click zooms 2x around the cursor.
        scene_pos = ev.scenePos()
        mouse_point = self.mapSceneToView(scene_pos)
        self.owner.zoom_span_at(float(mouse_point.x()), 0.5)
        ev.accept()


class SpectrumWorker(QtCore.QThread):
    new_data = QtCore.pyqtSignal(dict)

    def __init__(self, sdr: PlutoSdr, proc: SpectrumProcessor, cfg: SpectrumConfig):
        super().__init__()
        self.sdr = sdr
        self.proc = proc
        self.cfg = cfg
        self._lock = QtCore.QMutex()
        self._running = True
        # Pending configuration updates applied on the worker thread to avoid UI races.
        self.pending_apply: Dict[str, object] = {}

    def stop(self):
        self._running = False

    def queue_config(self, updates: Dict[str, object]) -> None:
        # Merge updates so multiple UI actions coalesce into one worker-side apply.
        with QtCore.QMutexLocker(self._lock):
            self.pending_apply.update(updates)

    def run(self):
        next_time = time.monotonic()
        while self._running:
            with QtCore.QMutexLocker(self._lock):
                if self.pending_apply:
                    # Apply any queued SDR/FFT changes in a strict, safe order.
                    pending = dict(self.pending_apply)
                    self.pending_apply.clear()
                    for key, value in pending.items():
                        if key == "span_hz":
                            self.cfg.sample_rate_hz = int(value)
                            continue
                        if hasattr(self.cfg, key):
                            setattr(self.cfg, key, value)

                    span_hz = pending.get("span_hz")
                    if span_hz is not None:
                        # Span impacts sample rate; apply before RF BW and buffer sizing.
                        self.sdr.set_span_hz(span_hz)

                    rf_bw_hz = pending.get("rf_bw_hz")
                    if rf_bw_hz is not None:
                        self.sdr.set_rf_bw(rf_bw_hz)

                    fft_size = int(pending.get("fft_size", self.cfg.fft_size))
                    buffer_factor = int(pending.get("buffer_factor", self.cfg.buffer_factor))
                    if "fft_size" in pending or "buffer_factor" in pending:
                        # RX buffer sizing depends on FFT size and buffer factor.
                        self.sdr.set_fft_size(fft_size, buffer_factor)

                    window_name = pending.get("window", self.cfg.window)
                    if "fft_size" in pending or "window" in pending:
                        # Update processor once per apply to keep window/FFT consistent.
                        self.proc.update_fft_size(fft_size, window_name)

                # Snapshot config under lock to keep SDR/FFT settings coherent.
                detector = self.cfg.detector
                overlap = self.cfg.overlap
                dc_remove = self.cfg.dc_remove
                dc_blank_bins = self.cfg.dc_blank_bins
                fft_size = self.cfg.fft_size
                window_name = self.cfg.window
                if fft_size != self.proc.fft_size or window_name != self.proc.window_name:
                    self.proc.update_fft_size(fft_size, window_name)
                # Read one large buffer for overlap/detector processing.
                x = self.sdr.read_rx()
                fs = self.sdr.sample_rate
                lo = self.sdr.lo

            power, count = self.proc.process_buffer(
                x,
                fs_hz=fs,
                overlap=overlap,
                detector=detector,
                dc_remove=dc_remove,
                dc_blank_bins=dc_blank_bins,
            )
            spectrogram_mode = self.cfg.spectrogram_mode
            if spectrogram_mode == "Peak Hold":
                spectrogram_db = self.proc.compute_spectrogram_peak(
                    x,
                    fs_hz=fs,
                    overlap=overlap,
                    dc_remove=dc_remove,
                )
            elif spectrogram_mode == "Max Power":
                spectrogram_db = self.proc.compute_spectrogram_peak(
                    x,
                    fs_hz=fs,
                    overlap=overlap,
                    dc_remove=dc_remove,
                )
            else:
                spectrogram_db = self.proc.compute_spectrogram_slice(
                    x,
                    fs_hz=fs,
                    overlap=overlap,
                    dc_remove=dc_remove,
                )
            freqs = self.proc.freqs_hz(fs, lo)
            rbw = self.proc.rbw_hz(fs)

            payload = {
                "freqs": freqs,
                "power": power,
                "spectrogram_db": spectrogram_db,
                "rbw": rbw,
                "enbw_bins": self.proc.enbw_bins,
                "fft_count": count,
                "timestamp": time.monotonic(),
                "fs": fs,
                "lo": lo,
                "gain_db": self.sdr.gain_db,
                "rf_bw": self.sdr.rf_bw,
            }
            self.new_data.emit(payload)

            # Pace updates based on UI update interval.
            next_time = time.monotonic() + (self.cfg.update_ms / 1000.0)
            sleep_time = max(0.0, next_time - time.monotonic())
            time.sleep(sleep_time)


class SpectrumWindow(QtWidgets.QMainWindow):
    """
    Main UI class.
    All SDR actions go through PlutoSdr.
    All DSP goes through SpectrumProcessor.
    """

    def __init__(self, cfg: SpectrumConfig):
        super().__init__()
        self.cfg = cfg
        self.cal = Calibration.load()
        self.state = self._load_state()

        # Restore persisted analyzer settings for an instrument-like feel.
        self.cfg.center_hz = int(self.state.get("center_hz", self.cfg.center_hz))
        self.cfg.sample_rate_hz = int(self.state.get("span_hz", self.cfg.sample_rate_hz))
        self.cfg.rf_bw_hz = int(self.state.get("rf_bw_hz", self.cfg.rf_bw_hz))
        self.cfg.gain_mode = self.state.get("gain_mode", self.cfg.gain_mode)
        self.cfg.gain_db = int(self.state.get("gain_db", self.cfg.gain_db))
        self.cfg.measurement_mode = bool(self.state.get("measurement_mode", self.cfg.measurement_mode))
        self.cfg.rbw_mode = self.state.get("rbw_mode", self.cfg.rbw_mode)
        self.cfg.vbw_mode = self.state.get("vbw_mode", self.cfg.vbw_mode)
        self.cfg.rbw_hz = float(self.state.get("rbw_hz", self.cfg.rbw_hz))
        self.cfg.vbw_hz = float(self.state.get("vbw_hz", self.cfg.vbw_hz))
        self.cfg.window = self.state.get("window", self.cfg.window)
        self.cfg.detector = self.state.get("detector", self.cfg.detector)
        self.cfg.trace_type = self.state.get("trace1_mode", self.cfg.trace_type)
        self.cfg.trace2_enabled = bool(self.state.get("trace2_enabled", self.cfg.trace2_enabled))
        self.cfg.avg_count = int(self.state.get("avg_count", self.cfg.avg_count))
        self.cfg.avg_mode = self.state.get("avg_mode", self.cfg.avg_mode)
        self.cfg.overlap = float(self.state.get("overlap", self.cfg.overlap))
        self.cfg.buffer_factor = int(self.state.get("buffer_factor", self.cfg.buffer_factor))
        self.cfg.fft_size = int(self.state.get("fft_size", self.cfg.fft_size))
        self.cfg.update_ms = int(self.state.get("update_ms", self.cfg.update_ms))
        self.cfg.ref_level_db = float(self.state.get("ref_level_db", self.cfg.ref_level_db))
        self.cfg.display_range_db = float(self.state.get("display_range_db", self.cfg.display_range_db))
        self.auto_scale_enabled = bool(self.state.get("auto_scale", True))
        self.hold_across_settings = bool(self.state.get("hold_across_settings", False))
        self.run_state = bool(self.state.get("run_state", True))
        self.amp_mode_state = self.state.get("amp_mode", "dBFS")

        self.setWindowTitle("Pluto Spectrum Analyzer")

        pg.setConfigOptions(antialias=False)
        pg.setConfigOption("background", (10, 10, 10))
        pg.setConfigOption("foreground", "w")

        self.sdr = PlutoSdr(cfg)
        self.proc = SpectrumProcessor(cfg.fft_size, cfg.window)
        self.hover = HoverReadout()

        self.last_update_ts: Optional[float] = None
        self.vbw_state: Optional[np.ndarray] = None
        self.avg_trace: Optional[np.ndarray] = None
        self.max_trace: Optional[np.ndarray] = None
        self.min_trace: Optional[np.ndarray] = None
        self.trace2_max: Optional[np.ndarray] = None
        self.latest_power: Optional[np.ndarray] = None
        self.latest_display: Optional[np.ndarray] = None
        self.latest_freqs: Optional[np.ndarray] = None
        self.latest_payload: Optional[Dict] = None
        self.latest_rbw: Optional[float] = None
        self.latest_peaks: Optional[np.ndarray] = None
        self.marker1: Optional[int] = None
        self.marker2: Optional[int] = None
        self.last_ui_update_ts: float = 0.0
        self.last_auto_ref_update: float = 0.0
        self.auto_ref_tau_s: float = 0.8
        self.auto_ref_hysteresis_db: float = 1.0
        self.auto_ref_headroom_db: float = 5.0
        self.auto_ref_max_step_db: float = 2.0
        self.single_pending: bool = False
        self.last_axis_span: Optional[float] = None
        self.last_axis_fft: Optional[int] = None
        self.help_overlays_enabled: bool = bool(self.state.get("help_overlays", True))
        self.trace_legend_enabled: bool = bool(self.state.get("trace_legend", True))
        self.spectrogram_enabled: bool = bool(self.state.get("spectrogram_enabled", False))
        self.spectrogram_rows: int = int(self.state.get("spectrogram_rows", 200))
        self.spectrogram_rate: float = float(self.state.get("spectrogram_rate", 15.0))
        self.spectrogram_span_s: float = float(
            self.state.get(
                "spectrogram_span_s",
                self.spectrogram_rows / max(self.spectrogram_rate, 1e-3),
            )
        )
        self.spectrogram_rows = max(1, int(round(self.spectrogram_span_s * self.spectrogram_rate)))
        self.cfg.spectrogram_mode = self.state.get("spectrogram_mode", self.cfg.spectrogram_mode)
        self.spectrogram_scale_mode: str = self.state.get(
            "spectrogram_scale_mode",
            "Auto (5-95%)",
        )
        self.spectrogram_cmap: str = self.state.get("spectrogram_cmap", "viridis")
        self.spectrogram_min_db: float = float(self.state.get("spectrogram_min_db", -120.0))
        self.spectrogram_max_db: float = float(self.state.get("spectrogram_max_db", 0.0))
        self.spectrogram_floor_db: Optional[float] = None
        self.spectrogram_buffer: Optional[np.ndarray] = None
        self.spectrogram_last_ts: float = 0.0
        self.peak_table_indices: list[int] = []

        self._build_ui()
        self._wire_events()

        self.worker = SpectrumWorker(self.sdr, self.proc, cfg)
        self.worker.new_data.connect(self.on_new_data)
        self.worker.start()

        self._apply_initial_state()

        # UI refresh timer for display update rate limiting (~10 Hz).
        self.ui_timer = QtCore.QTimer()
        self.ui_timer.timeout.connect(self.refresh_display)
        self.ui_timer.start(100)

    def closeEvent(self, event):
        # Persist state for predictable startup behavior.
        self._save_state()
        self.worker.stop()
        self.worker.wait(1000)
        event.accept()

    def _build_ui(self):
        self._build_menu()
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)
        layout.setSpacing(6)

        brand_layout = QtWidgets.QHBoxLayout()
        brand_label = QtWidgets.QLabel("Pluto Spectrum Analyzer")
        brand_label.setStyleSheet("QLabel { font-size: 16px; font-weight: 600; color: #38d0d4; }")
        brand_sub = QtWidgets.QLabel("Real time FFT")
        brand_sub.setStyleSheet("QLabel { color: #9aa0a6; }")
        brand_layout.addWidget(brand_label)
        brand_layout.addWidget(brand_sub)
        brand_layout.addStretch(1)
        layout.addLayout(brand_layout)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter, 1)

        control_panel = QtWidgets.QScrollArea()
        control_panel.setWidgetResizable(True)
        control_panel.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        splitter.addWidget(control_panel)

        control_contents = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_contents)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_panel.setWidget(control_contents)

        self.control_toolbox = QtWidgets.QToolBox()
        control_layout.addWidget(self.control_toolbox)
        control_layout.addStretch(1)

        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(plot_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        rf_widget = QtWidgets.QWidget()
        rf_layout = QtWidgets.QGridLayout(rf_widget)
        rf_layout.setHorizontalSpacing(6)
        rf_layout.setVerticalSpacing(4)

        rf_layout.addWidget(QtWidgets.QLabel("Center"), 0, 0)
        self.freq_edit = QtWidgets.QLineEdit()
        self.freq_edit.setFixedWidth(90)
        self.freq_edit.setAlignment(QtCore.Qt.AlignRight)
        rf_layout.addWidget(self.freq_edit, 0, 1)

        self.freq_unit = QtWidgets.QComboBox()
        self.freq_unit.addItems(["Hz", "kHz", "MHz", "GHz"])
        self.freq_unit.setCurrentText("MHz")
        self.freq_unit.setFixedWidth(70)
        rf_layout.addWidget(self.freq_unit, 0, 2)

        self.set_btn = QtWidgets.QPushButton("Set")
        rf_layout.addWidget(self.set_btn, 0, 3)

        rf_layout.addWidget(QtWidgets.QLabel("Span"), 1, 0)
        self.span_edit = QtWidgets.QLineEdit()
        self.span_edit.setFixedWidth(90)
        self.span_edit.setAlignment(QtCore.Qt.AlignRight)
        rf_layout.addWidget(self.span_edit, 1, 1)

        self.span_unit = QtWidgets.QComboBox()
        self.span_unit.addItems(["Hz", "kHz", "MHz"])
        self.span_unit.setCurrentText("MHz")
        self.span_unit.setFixedWidth(70)
        rf_layout.addWidget(self.span_unit, 1, 2)

        self.apply_span_btn = QtWidgets.QPushButton("Apply")
        rf_layout.addWidget(self.apply_span_btn, 1, 3)

        rf_layout.addWidget(QtWidgets.QLabel("RF BW"), 2, 0)
        self.rfbw_edit = QtWidgets.QLineEdit()
        self.rfbw_edit.setFixedWidth(90)
        self.rfbw_edit.setAlignment(QtCore.Qt.AlignRight)
        rf_layout.addWidget(self.rfbw_edit, 2, 1)

        self.rfbw_unit = QtWidgets.QComboBox()
        self.rfbw_unit.addItems(["Hz", "kHz", "MHz"])
        self.rfbw_unit.setCurrentText("MHz")
        self.rfbw_unit.setFixedWidth(70)
        rf_layout.addWidget(self.rfbw_unit, 2, 2)

        self.apply_rfbw_btn = QtWidgets.QPushButton("Set")
        rf_layout.addWidget(self.apply_rfbw_btn, 2, 3)

        rf_layout.addWidget(QtWidgets.QLabel("Gain mode"), 3, 0)
        self.gainmode_cb = QtWidgets.QComboBox()
        self.gainmode_cb.addItems(["manual", "fast_attack", "slow_attack", "hybrid"])
        rf_layout.addWidget(self.gainmode_cb, 3, 1)

        rf_layout.addWidget(QtWidgets.QLabel("Gain dB"), 3, 2)
        self.gain_edit = QtWidgets.QLineEdit()
        self.gain_edit.setFixedWidth(60)
        self.gain_edit.setAlignment(QtCore.Qt.AlignRight)
        rf_layout.addWidget(self.gain_edit, 3, 3)

        self.apply_gain_btn = QtWidgets.QPushButton("Apply")
        rf_layout.addWidget(self.apply_gain_btn, 3, 4)

        self.measurement_cb = QtWidgets.QCheckBox("Measurement mode")
        rf_layout.addWidget(self.measurement_cb, 4, 0, 1, 3)

        self.control_toolbox.addItem(rf_widget, "RF")

        sweep_widget = QtWidgets.QWidget()
        sweep_layout = QtWidgets.QGridLayout(sweep_widget)
        sweep_layout.setHorizontalSpacing(6)
        sweep_layout.setVerticalSpacing(4)

        sweep_layout.addWidget(QtWidgets.QLabel("RBW"), 0, 0)
        self.rbw_mode_cb = QtWidgets.QComboBox()
        self.rbw_mode_cb.addItems(["Auto", "Manual"])
        self.rbw_mode_cb.setCurrentText(self.cfg.rbw_mode)
        sweep_layout.addWidget(self.rbw_mode_cb, 0, 1)

        self.rbw_edit = QtWidgets.QLineEdit()
        self.rbw_edit.setFixedWidth(80)
        self.rbw_edit.setAlignment(QtCore.Qt.AlignRight)
        sweep_layout.addWidget(self.rbw_edit, 0, 2)

        self.rbw_apply_btn = QtWidgets.QPushButton("Apply")
        sweep_layout.addWidget(self.rbw_apply_btn, 0, 3)

        sweep_layout.addWidget(QtWidgets.QLabel("VBW"), 1, 0)
        self.vbw_mode_cb = QtWidgets.QComboBox()
        self.vbw_mode_cb.addItems(["Off", "Auto", "Manual"])
        self.vbw_mode_cb.setCurrentText(self.cfg.vbw_mode)
        sweep_layout.addWidget(self.vbw_mode_cb, 1, 1)

        self.vbw_edit = QtWidgets.QLineEdit()
        self.vbw_edit.setFixedWidth(80)
        self.vbw_edit.setAlignment(QtCore.Qt.AlignRight)
        sweep_layout.addWidget(self.vbw_edit, 1, 2)

        sweep_layout.addWidget(QtWidgets.QLabel("Detector"), 2, 0)
        self.detector_cb = QtWidgets.QComboBox()
        self.detector_cb.addItems(["Sample", "Peak", "RMS"])
        sweep_layout.addWidget(self.detector_cb, 2, 1)

        sweep_layout.addWidget(QtWidgets.QLabel("Window"), 2, 2)
        self.window_cb = QtWidgets.QComboBox()
        self.window_cb.addItems(["Hann", "Blackman Harris", "Flat top"])
        sweep_layout.addWidget(self.window_cb, 2, 3)

        sweep_layout.addWidget(QtWidgets.QLabel("Update Hz"), 3, 2)
        self.update_hz_edit = QtWidgets.QLineEdit()
        self.update_hz_edit.setFixedWidth(60)
        self.update_hz_edit.setAlignment(QtCore.Qt.AlignRight)
        sweep_layout.addWidget(self.update_hz_edit, 3, 3)

        self.control_toolbox.addItem(sweep_widget, "Sweep / FFT")

        trace_widget = QtWidgets.QWidget()
        trace_layout = QtWidgets.QGridLayout(trace_widget)
        trace_layout.setHorizontalSpacing(6)
        trace_layout.setVerticalSpacing(4)

        trace_layout.addWidget(QtWidgets.QLabel("Trace 1"), 0, 0)
        self.trace1_mode_cb = QtWidgets.QComboBox()
        self.trace1_mode_cb.addItems(["Clear Write", "Average", "Max Hold", "Min Hold"])
        trace_layout.addWidget(self.trace1_mode_cb, 0, 1)

        self.clear_trace1_btn = QtWidgets.QPushButton("Clear Trace 1")
        trace_layout.addWidget(self.clear_trace1_btn, 0, 2, 1, 2)

        trace_layout.addWidget(QtWidgets.QLabel("Trace 2"), 1, 0)
        self.trace2_mode_cb = QtWidgets.QComboBox()
        self.trace2_mode_cb.addItems(["Off", "Max Hold"])
        trace_layout.addWidget(self.trace2_mode_cb, 1, 1)

        self.clear_trace2_btn = QtWidgets.QPushButton("Clear Trace 2")
        trace_layout.addWidget(self.clear_trace2_btn, 1, 2, 1, 2)

        trace_layout.addWidget(QtWidgets.QLabel("Avg count"), 2, 0)
        self.avg_edit = QtWidgets.QLineEdit()
        self.avg_edit.setFixedWidth(60)
        self.avg_edit.setAlignment(QtCore.Qt.AlignRight)
        trace_layout.addWidget(self.avg_edit, 2, 1)

        self.avg_mode_cb = QtWidgets.QComboBox()
        self.avg_mode_cb.addItems(["RMS", "Log"])
        self.avg_mode_cb.setCurrentText(self.cfg.avg_mode)
        trace_layout.addWidget(self.avg_mode_cb, 2, 2)

        trace_layout.addWidget(QtWidgets.QLabel("Amplitude"), 3, 0)
        self.amp_mode_cb = QtWidgets.QComboBox()
        self._update_amp_modes()
        trace_layout.addWidget(self.amp_mode_cb, 3, 1)

        trace_layout.addWidget(QtWidgets.QLabel("Ref level"), 3, 2)
        self.ref_edit = QtWidgets.QLineEdit()
        self.ref_edit.setFixedWidth(70)
        self.ref_edit.setAlignment(QtCore.Qt.AlignRight)
        trace_layout.addWidget(self.ref_edit, 3, 3)

        trace_layout.addWidget(QtWidgets.QLabel("Scale (dB/div)"), 4, 0)
        self.scale_edit = QtWidgets.QLineEdit()
        self.scale_edit.setFixedWidth(60)
        self.scale_edit.setAlignment(QtCore.Qt.AlignRight)
        trace_layout.addWidget(self.scale_edit, 4, 1)

        self.range_label = QtWidgets.QLabel("Range: 100 dB")
        trace_layout.addWidget(self.range_label, 4, 2, 1, 2)

        self.auto_ref_cb = QtWidgets.QCheckBox("Auto Scale")
        self.auto_ref_cb.setChecked(True)
        trace_layout.addWidget(self.auto_ref_cb, 5, 0, 1, 2)

        self.hold_across_cb = QtWidgets.QCheckBox("Hold across settings")
        trace_layout.addWidget(self.hold_across_cb, 5, 2, 1, 2)

        self.dc_remove_cb = QtWidgets.QCheckBox("DC remove")
        trace_layout.addWidget(self.dc_remove_cb, 6, 0, 1, 2)

        self.run_stop_btn = QtWidgets.QPushButton("Run")
        self.single_btn = QtWidgets.QPushButton("Single")
        trace_layout.addWidget(self.run_stop_btn, 7, 0, 1, 2)
        trace_layout.addWidget(self.single_btn, 7, 2, 1, 2)

        self.save_csv_btn = QtWidgets.QPushButton("Save trace CSV")
        trace_layout.addWidget(self.save_csv_btn, 8, 0, 1, 4)

        self.control_toolbox.addItem(trace_widget, "Traces & Scaling")

        markers_widget = QtWidgets.QWidget()
        markers_layout = QtWidgets.QVBoxLayout(markers_widget)
        markers_layout.setSpacing(4)
        self.marker_info = QtWidgets.QLabel("M1: --\nM2: --\nΔ : --\nNoise: --")
        self.marker_info.setStyleSheet("QLabel { font-family: monospace; }")
        markers_layout.addWidget(self.marker_info)

        self.peak_table = QtWidgets.QTableWidget(5, 2)
        self.peak_table.setHorizontalHeaderLabels(["Freq", "Amp"])
        self.peak_table.verticalHeader().setVisible(False)
        self.peak_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.peak_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.peak_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.peak_table.horizontalHeader().setStretchLastSection(True)
        self.peak_table.setFixedHeight(140)
        markers_layout.addWidget(self.peak_table)

        marker_btns = QtWidgets.QHBoxLayout()
        self.marker_peak_btn = QtWidgets.QPushButton("Peak")
        self.marker_left_btn = QtWidgets.QPushButton("Next ◀")
        self.marker_right_btn = QtWidgets.QPushButton("Next ▶")
        self.clear_markers_btn = QtWidgets.QPushButton("Clear")
        marker_btns.addWidget(self.marker_peak_btn)
        marker_btns.addWidget(self.marker_left_btn)
        marker_btns.addWidget(self.marker_right_btn)
        marker_btns.addWidget(self.clear_markers_btn)
        markers_layout.addLayout(marker_btns)

        self.marker_to_center_btn = QtWidgets.QPushButton("Marker to Center")
        markers_layout.addWidget(self.marker_to_center_btn)

        self.control_toolbox.addItem(markers_widget, "Markers")

        presets_widget = QtWidgets.QWidget()
        presets_layout = QtWidgets.QGridLayout(presets_widget)
        presets_layout.setHorizontalSpacing(6)
        presets_layout.setVerticalSpacing(4)

        self.preset_fast_btn = QtWidgets.QPushButton("Fast View")
        self.preset_wide_btn = QtWidgets.QPushButton("Wide Scan")
        self.preset_measure_btn = QtWidgets.QPushButton("Measure")
        presets_layout.addWidget(self.preset_fast_btn, 0, 0)
        presets_layout.addWidget(self.preset_wide_btn, 0, 1)
        presets_layout.addWidget(self.preset_measure_btn, 0, 2)

        presets_layout.addWidget(QtWidgets.QLabel("Measure detector"), 1, 0)
        self.measure_detector_cb = QtWidgets.QComboBox()
        self.measure_detector_cb.addItems(["Peak", "RMS"])
        presets_layout.addWidget(self.measure_detector_cb, 1, 1)

        self.control_toolbox.addItem(presets_widget, "Presets")

        spectrogram_widget = QtWidgets.QWidget()
        spectrogram_layout = QtWidgets.QGridLayout(spectrogram_widget)
        spectrogram_layout.setHorizontalSpacing(6)
        spectrogram_layout.setVerticalSpacing(4)

        spectrogram_layout.addWidget(QtWidgets.QLabel("Mode"), 0, 0)
        self.spectrogram_mode_cb = QtWidgets.QComboBox()
        self.spectrogram_mode_cb.addItems(["PSD (Welch)", "Peak Hold", "Max Power"])
        self.spectrogram_mode_cb.setCurrentText(self.cfg.spectrogram_mode)
        spectrogram_layout.addWidget(self.spectrogram_mode_cb, 0, 1, 1, 3)

        spectrogram_layout.addWidget(QtWidgets.QLabel("Time Resolution (slices/s)"), 1, 0)
        self.spectrogram_speed_cb = QtWidgets.QComboBox()
        self.spectrogram_speed_cb.addItems(["5", "10", "15", "20", "30"])
        self.spectrogram_speed_cb.setCurrentText(f"{int(self.spectrogram_rate)}")
        spectrogram_layout.addWidget(self.spectrogram_speed_cb, 1, 1)
        self.spectrogram_perf_label = QtWidgets.QLabel("↓ resolution")
        self.spectrogram_perf_label.setStyleSheet("QLabel { color: #f4b400; }")
        self.spectrogram_perf_label.setVisible(False)
        spectrogram_layout.addWidget(self.spectrogram_perf_label, 1, 2, 1, 2)

        spectrogram_layout.addWidget(QtWidgets.QLabel("Time Span (s)"), 2, 0)
        self.spectrogram_depth_cb = QtWidgets.QComboBox()
        self.spectrogram_depth_cb.addItems(["5", "10", "15", "20", "30", "60"])
        self.spectrogram_depth_cb.setCurrentText(f"{int(round(self.spectrogram_span_s))}")
        spectrogram_layout.addWidget(self.spectrogram_depth_cb, 2, 1)

        spectrogram_layout.addWidget(QtWidgets.QLabel("Colormap"), 3, 0)
        self.spectrogram_cmap_cb = QtWidgets.QComboBox()
        self.spectrogram_cmap_cb.addItems(["viridis", "plasma"])
        self.spectrogram_cmap_cb.setCurrentText(self.spectrogram_cmap)
        spectrogram_layout.addWidget(self.spectrogram_cmap_cb, 3, 1)

        spectrogram_layout.addWidget(QtWidgets.QLabel("Scale"), 4, 0)
        self.spectrogram_scale_cb = QtWidgets.QComboBox()
        self.spectrogram_scale_cb.addItems(["Auto (5-95%)", "Fixed floor", "Manual (Expert)"])
        self.spectrogram_scale_cb.setCurrentText(self.spectrogram_scale_mode)
        spectrogram_layout.addWidget(self.spectrogram_scale_cb, 4, 1)

        self.spectrogram_min_edit = QtWidgets.QLineEdit(f"{self.spectrogram_min_db:.0f}")
        self.spectrogram_min_edit.setFixedWidth(60)
        self.spectrogram_min_edit.setAlignment(QtCore.Qt.AlignRight)
        self.spectrogram_max_edit = QtWidgets.QLineEdit(f"{self.spectrogram_max_db:.0f}")
        self.spectrogram_max_edit.setFixedWidth(60)
        self.spectrogram_max_edit.setAlignment(QtCore.Qt.AlignRight)
        spectrogram_layout.addWidget(QtWidgets.QLabel("dB min"), 5, 0)
        spectrogram_layout.addWidget(self.spectrogram_min_edit, 5, 1)
        spectrogram_layout.addWidget(QtWidgets.QLabel("dB max"), 5, 2)
        spectrogram_layout.addWidget(self.spectrogram_max_edit, 5, 3)

        self.control_toolbox.addItem(spectrogram_widget, "Spectrogram")

        advanced_widget = QtWidgets.QWidget()
        advanced_layout = QtWidgets.QGridLayout(advanced_widget)
        advanced_layout.setHorizontalSpacing(6)
        advanced_layout.setVerticalSpacing(4)

        advanced_layout.addWidget(QtWidgets.QLabel("Overlap"), 0, 0)
        self.overlap_edit = QtWidgets.QLineEdit()
        self.overlap_edit.setFixedWidth(60)
        self.overlap_edit.setAlignment(QtCore.Qt.AlignRight)
        advanced_layout.addWidget(self.overlap_edit, 0, 1)

        advanced_layout.addWidget(QtWidgets.QLabel("Buffer factor"), 1, 0)
        self.buffer_factor_edit = QtWidgets.QLineEdit()
        self.buffer_factor_edit.setFixedWidth(60)
        self.buffer_factor_edit.setAlignment(QtCore.Qt.AlignRight)
        advanced_layout.addWidget(self.buffer_factor_edit, 1, 1)

        advanced_layout.addWidget(QtWidgets.QLabel("DC blank bins"), 2, 0)
        self.dc_blank_edit = QtWidgets.QLineEdit()
        self.dc_blank_edit.setFixedWidth(50)
        self.dc_blank_edit.setAlignment(QtCore.Qt.AlignRight)
        advanced_layout.addWidget(self.dc_blank_edit, 2, 1)

        advanced_layout.addWidget(QtWidgets.QLabel("ENBW (bins)"), 3, 0)
        self.enbw_label = QtWidgets.QLabel("--")
        advanced_layout.addWidget(self.enbw_label, 3, 1)

        self.calibration_btn = QtWidgets.QPushButton("Calibration...")
        advanced_layout.addWidget(self.calibration_btn, 4, 0, 1, 2)

        self.control_toolbox.addItem(advanced_widget, "Advanced")
        self.control_toolbox.setCurrentIndex(0)

        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("QLabel { padding: 4px; font-family: monospace; }")

        # Main plot + optional spectrogram waterfall below.
        self.plotw = pg.GraphicsLayoutWidget()
        plot_layout.addWidget(self.plotw, 1)
        plot_layout.addWidget(self.status)

        self.plot = self.plotw.addPlot(viewBox=SpectrumViewBox(self))
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "Magnitude", units="dBFS")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setMouseEnabled(x=True, y=False)

        self.spectrogram_plot = self.plotw.addPlot(row=1, col=0)
        self.spectrogram_plot.setLabel("bottom", "Frequency", units="Hz")
        self.spectrogram_plot.setLabel("left", "Time (s ago)", units="s")
        self.spectrogram_plot.showGrid(x=True, y=False, alpha=0.2)
        self.spectrogram_plot.setMouseEnabled(x=False, y=False)
        self.spectrogram_plot.setXLink(self.plot)
        self.spectrogram_image = pg.ImageItem()
        self.spectrogram_plot.addItem(self.spectrogram_image)
        self._update_spectrogram_colormap()
        if not self.spectrogram_enabled:
            self.spectrogram_plot.hide()
        self.plotw.ci.layout.setRowStretchFactor(0, 3)
        self.plotw.ci.layout.setRowStretchFactor(1, 1)

        self.curve_trace1 = self.plot.plot(pen=pg.mkPen("w", width=1))
        self.curve_trace2 = self.plot.plot(pen=pg.mkPen("y", width=2, style=QtCore.Qt.DashLine))
        self.plot.setClipToView(True)
        self.curve_trace1.setClipToView(True)
        self.curve_trace2.setClipToView(True)

        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#8a8a8a"))
        self.plot.addItem(self.vline, ignoreBounds=True)

        self.marker1_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ffd200"))
        self.marker2_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ff4ff0"))
        self.plot.addItem(self.marker1_line, ignoreBounds=True)
        self.plot.addItem(self.marker2_line, ignoreBounds=True)
        self.marker1_line.hide()
        self.marker2_line.hide()

        self.hover_text = pg.TextItem(
            "",
            anchor=(0, 1),
            color=(255, 255, 255),
            fill=pg.mkBrush(0, 0, 0, 180),
        )
        self.hover_text.setZValue(1e6)
        self.plot.addItem(self.hover_text)

        self.trace_legend_item = pg.TextItem("", anchor=(0, 0), color=(200, 200, 200))
        self.trace_legend_item.setZValue(1e5)
        self.plot.addItem(self.trace_legend_item)

        self._mouse_proxy = SignalProxy(
            self.plot.scene().sigMouseMoved,
            rateLimit=int(self.cfg.hover_rate_hz),
            slot=self.on_mouse_moved,
        )
        self._click_proxy = SignalProxy(
            self.plot.scene().sigMouseClicked,
            rateLimit=10,
            slot=self.on_mouse_clicked,
        )

        self.help_overlay = QtWidgets.QLabel("", self)
        self.help_overlay.setStyleSheet(
            "QLabel { background-color: rgba(20, 20, 20, 220); color: #ffffff; "
            "border-radius: 6px; padding: 6px; }"
        )
        self.help_overlay.setWordWrap(True)
        self.help_overlay.setFixedWidth(260)
        self.help_overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.help_overlay.hide()

        # Map widgets to help text for tooltips and hover overlays.
        self.help_texts: Dict[QtWidgets.QWidget, str] = {}
        self._setup_help_overlays()

    def _build_menu(self):
        menu = self.menuBar()

        # File actions for export/exit.
        file_menu = menu.addMenu("File")
        self.save_screenshot_action = QtWidgets.QAction("Save screenshot", self)
        self.exit_action = QtWidgets.QAction("Exit", self)
        file_menu.addAction(self.save_screenshot_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        # View toggles for overlays and optional panels.
        view_menu = menu.addMenu("View")
        self.help_overlay_action = QtWidgets.QAction("Show help overlays", self, checkable=True)
        self.help_overlay_action.setChecked(self.help_overlays_enabled)
        self.trace_legend_action = QtWidgets.QAction("Show trace legend", self, checkable=True)
        self.trace_legend_action.setChecked(self.trace_legend_enabled)
        self.spectrogram_action = QtWidgets.QAction("Show spectrogram panel", self, checkable=True)
        self.spectrogram_action.setChecked(self.spectrogram_enabled)
        view_menu.addAction(self.help_overlay_action)
        view_menu.addAction(self.trace_legend_action)
        view_menu.addAction(self.spectrogram_action)

        # Tools for calibration and resetting UI state.
        tools_menu = menu.addMenu("Tools")
        self.calibration_action = QtWidgets.QAction("Calibration...", self)
        self.reset_state_action = QtWidgets.QAction("Reset state", self)
        tools_menu.addAction(self.calibration_action)
        tools_menu.addAction(self.reset_state_action)

        # Preset shortcuts (mirrors the preset buttons).
        presets_menu = menu.addMenu("Presets")
        self.preset_fast_action = QtWidgets.QAction("Fast View", self)
        self.preset_wide_action = QtWidgets.QAction("Wide Scan", self)
        self.preset_measure_action = QtWidgets.QAction("Measure", self)
        presets_menu.addAction(self.preset_fast_action)
        presets_menu.addAction(self.preset_wide_action)
        presets_menu.addAction(self.preset_measure_action)

    def _register_help(self, widget: QtWidgets.QWidget, text: str) -> None:
        # ToolTip shows on long hover; overlay shows on enter/leave.
        widget.setToolTip(text)
        self.help_texts[widget] = text
        widget.installEventFilter(self)

    def _setup_help_overlays(self) -> None:
        self._register_help(
            self.freq_edit,
            "Center frequency (Hz). Sets the LO tuning point around which span is drawn. Default 2.437 GHz.",
        )
        self._register_help(
            self.freq_unit,
            "Center frequency units. Typical Pluto defaults near 2.4 GHz.",
        )
        self._register_help(
            self.set_btn,
            "Apply center frequency. Updates the LO tuning immediately.",
        )
        self._register_help(
            self.span_edit,
            "Span/sample rate (Hz). Wider span shows more spectrum but increases noise floor. Default 20 MHz.",
        )
        self._register_help(
            self.span_unit,
            "Span units. Span equals sample rate and sets visible bandwidth.",
        )
        self._register_help(
            self.apply_span_btn,
            "Apply span. Changes sample rate and updates RBW strategy.",
        )
        self._register_help(
            self.rfbw_edit,
            "RF bandwidth (Hz). Should be ≥ span. Typical Pluto default is 20 MHz.",
        )
        self._register_help(
            self.rfbw_unit,
            "RF bandwidth units. Keep RF BW at or above span.",
        )
        self._register_help(
            self.apply_rfbw_btn,
            "Apply RF bandwidth. Keeps analog filter wide enough for span.",
        )
        self._register_help(
            self.gainmode_cb,
            "Gain control mode. Manual holds gain fixed; AGC modes track signal changes. Default manual.",
        )
        self._register_help(
            self.gain_edit,
            "Manual gain in dB. Higher gain raises noise floor and risk of clipping. Default 55 dB.",
        )
        self._register_help(
            self.apply_gain_btn,
            "Apply manual gain. Sets Pluto hardware gain in dB.",
        )
        self._register_help(
            self.measurement_cb,
            "Measurement mode. Locks gain to manual for repeatable readings.",
        )
        self._register_help(
            self.rbw_mode_cb,
            "RBW mode. Auto picks FFT size from span; Manual uses the RBW value. Default Manual.",
        )
        self._register_help(
            self.rbw_edit,
            "Resolution bandwidth (Hz). Smaller RBW separates closer signals but slows updates. RBW depends on FFT size and window ENBW.",
        )
        self._register_help(
            self.rbw_apply_btn,
            "Apply RBW. Updates FFT size and RBW based on selection.",
        )
        self._register_help(
            self.vbw_mode_cb,
            "VBW mode. Off disables smoothing; Auto tracks RBW/10; Manual uses VBW value. Default Auto.",
        )
        self._register_help(
            self.vbw_edit,
            "Video bandwidth (Hz). Lower VBW smooths noise but slows response.",
        )
        self._register_help(
            self.detector_cb,
            "Detector selection. RMS is stable; Peak finds maxima; Sample is instantaneous. Default RMS.",
        )
        self._register_help(
            self.window_cb,
            "FFT window. Changes sidelobe performance and ENBW. Default Blackman Harris.",
        )
        self._register_help(
            self.overlap_edit,
            "Overlap fraction (0-1). Higher overlap improves detection but uses more CPU. Default 0.5.",
        )
        self._register_help(
            self.buffer_factor_edit,
            "Buffer factor. Multiplies FFT size to capture more slices per read. Higher values cost memory.",
        )
        self._register_help(
            self.update_hz_edit,
            "Update rate (Hz). Higher rates refresh faster but cost CPU. Default 10 Hz.",
        )
        self._register_help(
            self.trace1_mode_cb,
            "Trace 1 mode. Live, Average, Max Hold, or Min Hold display style.",
        )
        self._register_help(
            self.trace2_mode_cb,
            "Trace 2 mode. Optional Max Hold overlay for comparisons.",
        )
        self._register_help(
            self.avg_edit,
            "Average count. Higher values smooth noise but slow settling. Default 10.",
        )
        self._register_help(
            self.avg_mode_cb,
            "Average mode. RMS averages linear power; Log averages in dB.",
        )
        self._register_help(
            self.amp_mode_cb,
            "Amplitude mode. dBFS is uncalibrated; dBm modes need calibration. Default dBFS.",
        )
        self._register_help(
            self.ref_edit,
            "Reference level (dB). Top of the graticule scale. Default 0 dBFS.",
        )
        self._register_help(
            self.scale_edit,
            "Scale (dB/div). Larger values show more dynamic range. Default 10 dB/div.",
        )
        self._register_help(
            self.auto_ref_cb,
            "Auto scale. Tracks peaks and adjusts reference level automatically.",
        )
        self._register_help(
            self.dc_remove_cb,
            "DC remove. Subtracts mean to reduce DC spike at center.",
        )
        self._register_help(
            self.dc_blank_edit,
            "DC blank bins. Masks bins around center to suppress LO leakage.",
        )
        self._register_help(
            self.run_stop_btn,
            "Run/Stop. Pause updates to freeze the current trace.",
        )
        self._register_help(
            self.single_btn,
            "Single capture. Acquire one trace and then stop.",
        )
        self._register_help(
            self.marker_peak_btn,
            "Markers. Peak sets M1 to strongest signal; Next scans to neighbor peaks.",
        )
        self._register_help(
            self.marker_left_btn,
            "Markers. Move M1 to the previous detected peak.",
        )
        self._register_help(
            self.marker_right_btn,
            "Markers. Move M1 to the next detected peak.",
        )
        self._register_help(
            self.clear_markers_btn,
            "Markers. Clear all marker readouts and cursor lines.",
        )
        self._register_help(
            self.marker_to_center_btn,
            "Marker to Center. Tunes LO to the marker frequency.",
        )
        self._register_help(
            self.preset_fast_btn,
            "Preset: Fast View. Wider RBW for quick updates.",
        )
        self._register_help(
            self.preset_wide_btn,
            "Preset: Wide Scan. Default Pluto span and balanced windowing.",
        )
        self._register_help(
            self.preset_measure_btn,
            "Preset: Measure. Narrower RBW for better resolution.",
        )
        self._register_help(
            self.measure_detector_cb,
            "Measure preset detector. Controls detector type for measurement preset.",
        )
        self._register_help(
            self.spectrogram_mode_cb,
            "Spectrogram mode. PSD averages FFTs (Welch). Peak Hold tracks maxima. Max Power is experimental.",
        )
        self._register_help(
            self.spectrogram_speed_cb,
            "Spectrogram time resolution in slices per second.",
        )
        self._register_help(
            self.spectrogram_depth_cb,
            "Spectrogram time span in seconds.",
        )
        self._register_help(
            self.spectrogram_scale_cb,
            "Color scale mode. Auto uses percentiles; Fixed locks the noise floor; Manual is expert-only.",
        )
        self._register_help(
            self.calibration_btn,
            "Calibration tools for dBFS to dBm conversion and external gain.",
        )

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Enter and obj in self.help_texts:
            if self.help_overlays_enabled:
                # Show a floating helper label near the hovered widget.
                text = self.help_texts.get(obj, "")
                self.help_overlay.setText(text)
                self.help_overlay.adjustSize()
                pos = obj.mapTo(self, QtCore.QPoint(0, obj.height() + 6))
                x = min(pos.x(), self.width() - self.help_overlay.width() - 8)
                y = min(pos.y(), self.height() - self.help_overlay.height() - 8)
                self.help_overlay.move(max(8, x), max(8, y))
                self.help_overlay.show()
        elif event.type() == QtCore.QEvent.Leave and obj in self.help_texts:
            self.help_overlay.hide()
        return super().eventFilter(obj, event)

    def _wire_events(self):
        self.set_btn.clicked.connect(self.on_set_center)
        self.apply_span_btn.clicked.connect(self.on_apply_span)
        self.apply_rfbw_btn.clicked.connect(self.on_apply_rfbw)

        self.gainmode_cb.currentTextChanged.connect(self.on_gainmode_changed)
        self.apply_gain_btn.clicked.connect(self.on_apply_gain)
        self.measurement_cb.toggled.connect(self.on_measurement_mode)

        self.rbw_mode_cb.currentTextChanged.connect(self.on_rbw_mode)
        self.rbw_apply_btn.clicked.connect(self.on_apply_rbw)
        self.window_cb.currentTextChanged.connect(self.on_window_changed)

        self.vbw_mode_cb.currentTextChanged.connect(self.on_vbw_mode)
        self.vbw_edit.editingFinished.connect(self.on_vbw_changed)
        self.detector_cb.currentTextChanged.connect(self.on_detector_changed)
        self.overlap_edit.editingFinished.connect(self.on_overlap_changed)
        self.update_hz_edit.editingFinished.connect(self.on_update_rate_changed)
        self.buffer_factor_edit.editingFinished.connect(self.on_buffer_factor_changed)

        self.trace1_mode_cb.currentTextChanged.connect(self.on_trace1_mode_changed)
        self.trace2_mode_cb.currentTextChanged.connect(self.on_trace2_mode_changed)
        self.clear_trace1_btn.clicked.connect(self.on_clear_trace1)
        self.clear_trace2_btn.clicked.connect(self.on_clear_trace2)

        self.avg_edit.editingFinished.connect(self.on_avg_changed)
        self.avg_mode_cb.currentTextChanged.connect(self.on_avg_changed)

        self.amp_mode_cb.currentTextChanged.connect(self.on_amp_mode_changed)
        self.auto_ref_cb.toggled.connect(self.on_auto_ref_toggled)
        self.scale_edit.editingFinished.connect(self.on_scale_changed)
        self.ref_edit.editingFinished.connect(self.on_ref_changed)
        self.hold_across_cb.toggled.connect(self.on_hold_across_changed)

        self.dc_remove_cb.toggled.connect(self.on_dc_remove)
        self.dc_blank_edit.editingFinished.connect(self.on_dc_blank_changed)
        self.run_stop_btn.clicked.connect(self.on_run_stop)
        self.single_btn.clicked.connect(self.on_single)

        self.marker_peak_btn.clicked.connect(self.on_marker_peak)
        self.marker_left_btn.clicked.connect(self.on_marker_left)
        self.marker_right_btn.clicked.connect(self.on_marker_right)
        self.clear_markers_btn.clicked.connect(self.on_marker_clear)
        self.marker_to_center_btn.clicked.connect(self.on_marker_to_center)

        self.preset_fast_btn.clicked.connect(lambda: self.apply_preset("Fast View"))
        self.preset_wide_btn.clicked.connect(lambda: self.apply_preset("Wide Scan"))
        self.preset_measure_btn.clicked.connect(lambda: self.apply_preset("Measure"))

        self.save_csv_btn.clicked.connect(self.on_save_trace_csv)
        self.peak_table.cellClicked.connect(self.on_peak_table_clicked)

        self.save_screenshot_action.triggered.connect(self.on_save_screenshot)
        self.exit_action.triggered.connect(self.close)
        self.help_overlay_action.toggled.connect(self.on_toggle_help_overlays)
        self.trace_legend_action.toggled.connect(self.on_toggle_trace_legend)
        self.spectrogram_action.toggled.connect(self.on_toggle_spectrogram)
        self.calibration_action.triggered.connect(self.on_open_calibration)
        self.reset_state_action.triggered.connect(self.on_reset_state)
        self.calibration_btn.clicked.connect(self.on_open_calibration)
        self.preset_fast_action.triggered.connect(lambda: self.apply_preset("Fast View"))
        self.preset_wide_action.triggered.connect(lambda: self.apply_preset("Wide Scan"))
        self.preset_measure_action.triggered.connect(lambda: self.apply_preset("Measure"))

        self.spectrogram_mode_cb.currentTextChanged.connect(self.on_spectrogram_mode)
        self.spectrogram_speed_cb.currentTextChanged.connect(self.on_spectrogram_speed)
        self.spectrogram_depth_cb.currentTextChanged.connect(self.on_spectrogram_depth)
        self.spectrogram_scale_cb.currentTextChanged.connect(self.on_spectrogram_scale_mode)
        self.spectrogram_cmap_cb.currentTextChanged.connect(self.on_spectrogram_cmap)
        self.spectrogram_min_edit.editingFinished.connect(self.on_spectrogram_range)
        self.spectrogram_max_edit.editingFinished.connect(self.on_spectrogram_range)

    def _apply_initial_state(self):
        # Restore UI first.
        self._sync_center_edit()
        self._sync_span_edit()
        self._sync_rfbw_edit()
        self.gainmode_cb.setCurrentText(self.cfg.gain_mode)
        self.gain_edit.setText(str(self.cfg.gain_db))
        self.measurement_cb.setChecked(self.cfg.measurement_mode)

        self.rbw_mode_cb.setCurrentText(self.cfg.rbw_mode)
        self.rbw_edit.setText(f"{self.cfg.rbw_hz:.0f}")
        self.vbw_mode_cb.setCurrentText(self.cfg.vbw_mode)
        self.vbw_edit.setText(f"{self.cfg.vbw_hz:.0f}")
        self.detector_cb.setCurrentText(self.cfg.detector)
        self.window_cb.setCurrentText(self.cfg.window)
        self.overlap_edit.setText(f"{self.cfg.overlap:.2f}")
        self.buffer_factor_edit.setText(str(self.cfg.buffer_factor))
        update_hz = max(0.1, 1000.0 / float(self.cfg.update_ms))
        self.update_hz_edit.setText(f"{update_hz:.1f}")

        self.trace1_mode_cb.setCurrentText(self.cfg.trace_type)
        self.trace2_mode_cb.setCurrentText("Max Hold" if self.cfg.trace2_enabled else "Off")
        self.clear_trace2_btn.setEnabled(self.cfg.trace2_enabled)
        self._update_trace1_pen(self.cfg.trace_type)
        self.avg_edit.setText(str(self.cfg.avg_count))
        self.avg_mode_cb.setCurrentText(self.cfg.avg_mode)

        self._update_amp_modes()
        if self.amp_mode_state in [self.amp_mode_cb.itemText(i) for i in range(self.amp_mode_cb.count())]:
            self.amp_mode_cb.setCurrentText(self.amp_mode_state)
        else:
            self.amp_mode_cb.setCurrentText("dBFS")

        self.ref_edit.setText(f"{self.cfg.ref_level_db:.1f}")
        self.scale_edit.setText(f"{self.cfg.display_range_db / 10.0:.1f}")
        self.auto_ref_cb.setChecked(self.auto_scale_enabled)
        self.hold_across_cb.setChecked(self.hold_across_settings)
        self.dc_remove_cb.setChecked(self.cfg.dc_remove)
        self.dc_blank_edit.setText(str(self.cfg.dc_blank_bins))
        self.spectrogram_mode_cb.setCurrentText(self.cfg.spectrogram_mode)
        self.spectrogram_scale_cb.setCurrentText(self.spectrogram_scale_mode)
        self.spectrogram_rate = self._set_combo_to_nearest(
            self.spectrogram_speed_cb, self.spectrogram_rate
        )
        self.spectrogram_span_s = self._set_combo_to_nearest(
            self.spectrogram_depth_cb, self.spectrogram_span_s
        )
        self.spectrogram_rows = max(1, int(round(self.spectrogram_span_s * self.spectrogram_rate)))
        self._update_spectrogram_scale_controls()

        self.run_stop_btn.setText("Stop" if self.run_state else "Run")
        self.ref_edit.setEnabled(not self.auto_scale_enabled)

        # Apply SDR settings after UI restore.
        with QtCore.QMutexLocker(self.worker._lock):
            self.sdr.set_center_hz(self.cfg.center_hz)
            self.sdr.set_gain_mode(self.cfg.gain_mode)
            self.sdr.set_gain_db(self.cfg.gain_db)
        self.worker.queue_config(
            {
                "span_hz": self.cfg.sample_rate_hz,
                "rf_bw_hz": self.cfg.rf_bw_hz,
                "fft_size": self.cfg.fft_size,
                "buffer_factor": self.cfg.buffer_factor,
                "window": self.cfg.window,
                "detector": self.cfg.detector,
                "overlap": self.cfg.overlap,
                "vbw_mode": self.cfg.vbw_mode,
                "vbw_hz": self.cfg.vbw_hz,
            }
        )

        self.on_gainmode_changed(self.cfg.gain_mode)
        self.on_measurement_mode(self.cfg.measurement_mode)
        self.on_rbw_mode(self.cfg.rbw_mode)
        self.on_vbw_mode(self.cfg.vbw_mode)
        self.on_detector_changed(self.cfg.detector)
        self._update_range_label()
        self.snap_x_to_span()
        self._clear_trace_state()
        self._update_trace_legend()
        self._apply_spectrogram_visibility()

    def _update_amp_modes(self):
        current = self.amp_mode_cb.currentText() if hasattr(self, "amp_mode_cb") else None
        self.amp_mode_cb.clear()
        self.amp_mode_cb.addItem("dBFS")
        if self.cal.has_calibration():
            self.amp_mode_cb.addItem("dBm")
            self.amp_mode_cb.addItem("dBm/Hz")
        if current and current in [self.amp_mode_cb.itemText(i) for i in range(self.amp_mode_cb.count())]:
            self.amp_mode_cb.setCurrentText(current)

    def _set_combo_to_nearest(self, combo: QtWidgets.QComboBox, value: float) -> float:
        values = []
        indices = []
        for idx in range(combo.count()):
            try:
                values.append(float(combo.itemText(idx)))
                indices.append(idx)
            except ValueError:
                continue
        if not values:
            return value
        best_pos = min(range(len(values)), key=lambda i: abs(values[i] - value))
        best_idx = indices[best_pos]
        combo.blockSignals(True)
        combo.setCurrentIndex(best_idx)
        combo.blockSignals(False)
        return values[best_pos]

    def _load_state(self) -> Dict:
        if not os.path.exists(STATE_PATH):
            return {}
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_state(self) -> None:
        data = {
            "center_hz": self.cfg.center_hz,
            "span_hz": self.cfg.sample_rate_hz,
            "rf_bw_hz": self.cfg.rf_bw_hz,
            "gain_mode": self.cfg.gain_mode,
            "gain_db": self.cfg.gain_db,
            "measurement_mode": self.cfg.measurement_mode,
            "rbw_mode": self.cfg.rbw_mode,
            "rbw_hz": self.cfg.rbw_hz,
            "vbw_mode": self.cfg.vbw_mode,
            "vbw_hz": self.cfg.vbw_hz,
            "window": self.cfg.window,
            "overlap": self.cfg.overlap,
            "buffer_factor": self.cfg.buffer_factor,
            "detector": self.cfg.detector,
            "trace1_mode": self.cfg.trace_type,
            "trace2_enabled": self.cfg.trace2_enabled,
            "avg_count": self.cfg.avg_count,
            "avg_mode": self.cfg.avg_mode,
            "amp_mode": self.amp_mode_cb.currentText(),
            "ref_level_db": self.cfg.ref_level_db,
            "display_range_db": self.cfg.display_range_db,
            "auto_scale": self.auto_scale_enabled,
            "hold_across_settings": self.hold_across_settings,
            "update_ms": self.cfg.update_ms,
            "fft_size": self.cfg.fft_size,
            "run_state": self.run_state,
            "help_overlays": self.help_overlays_enabled,
            "trace_legend": self.trace_legend_enabled,
            "spectrogram_enabled": self.spectrogram_enabled,
            "spectrogram_rows": self.spectrogram_rows,
            "spectrogram_rate": self.spectrogram_rate,
            "spectrogram_span_s": self.spectrogram_span_s,
            "spectrogram_mode": self.cfg.spectrogram_mode,
            "spectrogram_scale_mode": self.spectrogram_scale_mode,
            "spectrogram_cmap": self.spectrogram_cmap,
            "spectrogram_min_db": self.spectrogram_min_db,
            "spectrogram_max_db": self.spectrogram_max_db,
        }
        with open(STATE_PATH, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def _set_cfg(self, **kwargs) -> None:
        # Update shared config with worker lock to avoid race conditions.
        with QtCore.QMutexLocker(self.worker._lock):
            for key, value in kwargs.items():
                setattr(self.cfg, key, value)

    def _queue_pending_config(self, **kwargs) -> None:
        self._set_cfg(**kwargs)
        self.worker.queue_config(kwargs)

    def _parse_value_to_hz(self, value_str: str, unit: str) -> int:
        value = float(value_str.strip())
        scale = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}[unit]
        return int(value * scale)

    def _sync_center_edit(self):
        unit = self.freq_unit.currentText()
        inv = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}[unit]
        self.freq_edit.setText(f"{self.cfg.center_hz / inv:.6g}")

    def _sync_span_edit(self):
        unit = self.span_unit.currentText()
        inv = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6}[unit]
        self.span_edit.setText(f"{self.cfg.sample_rate_hz / inv:.6g}")

    def _sync_rfbw_edit(self):
        unit = self.rfbw_unit.currentText()
        inv = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6}[unit]
        self.rfbw_edit.setText(f"{self.cfg.rf_bw_hz / inv:.6g}")

    def snap_x_to_span(self):
        # Keep the visible span centered on the LO.
        lo = float(self.sdr.lo)
        sr = float(self.sdr.sample_rate)
        self.plot.setXRange(lo - sr / 2.0, lo + sr / 2.0, padding=0.0)

    def on_set_center(self):
        try:
            hz = self._parse_value_to_hz(self.freq_edit.text(), self.freq_unit.currentText())
            with QtCore.QMutexLocker(self.worker._lock):
                self.sdr.set_center_hz(hz)
            self._sync_center_edit()
            self.snap_x_to_span()
            self.status.setText(f"Center set to {self.cfg.center_hz} Hz")
        except Exception as exc:
            self.status.setText(f"Bad center freq: {exc}")

    def on_apply_span(self):
        try:
            span_hz = self._parse_value_to_hz(self.span_edit.text(), self.span_unit.currentText())
            rf_bw_hz = max(span_hz, self.cfg.rf_bw_hz)
            self._set_cfg(sample_rate_hz=span_hz, rf_bw_hz=rf_bw_hz)
            self.worker.queue_config({"span_hz": span_hz, "rf_bw_hz": rf_bw_hz})
            self._sync_span_edit()
            self._sync_rfbw_edit()
            self._apply_rbw_strategy()
            self.snap_x_to_span()
            self._clear_trace_state_if_needed()
            self.status.setText(f"Span set to {span_hz} Hz")
        except Exception as exc:
            self.status.setText(f"Span apply failed: {exc}")

    def on_apply_rfbw(self):
        try:
            hz = self._parse_value_to_hz(self.rfbw_edit.text(), self.rfbw_unit.currentText())
            self._set_cfg(rf_bw_hz=hz)
            self.worker.queue_config({"rf_bw_hz": hz})
            self._sync_rfbw_edit()
            self.status.setText(f"RF BW set to {hz} Hz")
        except Exception as exc:
            self.status.setText(f"RF BW set failed: {exc}")

    def on_gainmode_changed(self, mode: str):
        try:
            with QtCore.QMutexLocker(self.worker._lock):
                self.sdr.set_gain_mode(mode)

            is_manual = mode == "manual"
            self.gain_edit.setEnabled(is_manual)
            self.apply_gain_btn.setEnabled(is_manual)

            self.status.setText(f"Gain mode set to {mode}")
        except Exception as exc:
            self.status.setText(f"Gain mode error: {exc}")

    def on_apply_gain(self):
        try:
            g = int(float(self.gain_edit.text().strip()))
            with QtCore.QMutexLocker(self.worker._lock):
                self.sdr.set_gain_db(g)
            self.status.setText(f"Manual gain set to {self.cfg.gain_db} dB")
        except Exception as exc:
            self.status.setText(f"Gain error: {exc}")

    def on_measurement_mode(self, checked: bool):
        self._set_cfg(measurement_mode=checked)
        if checked:
            self.gainmode_cb.setCurrentText("manual")
            self.gainmode_cb.setEnabled(False)
            self.gain_edit.setEnabled(True)
            self.apply_gain_btn.setEnabled(True)
            self.status.setText("Measurement mode enabled: manual gain enforced")
        else:
            self.gainmode_cb.setEnabled(True)

    def on_rbw_mode(self, mode: str):
        self._set_cfg(rbw_mode=mode)
        manual = mode == "Manual"
        self.rbw_edit.setEnabled(manual)
        self.rbw_apply_btn.setEnabled(manual)
        # Auto RBW recomputes FFT size to keep a manageable number of points.
        if not manual:
            self._apply_rbw_strategy()
            self._clear_trace_state_if_needed()

    def on_apply_rbw(self):
        try:
            rbw = float(self.rbw_edit.text().strip())
            if rbw <= 0:
                raise ValueError("RBW must be > 0")
            self._set_cfg(rbw_hz=rbw)
            # Manual RBW maps to nearest supported FFT size.
            self._apply_rbw_strategy()
        except Exception as exc:
            self.status.setText(f"RBW apply failed: {exc}")

    def on_window_changed(self, window: str):
        self._set_cfg(window=window)
        # Window choice affects ENBW, so recalc RBW/FFT size.
        self._apply_rbw_strategy()
        self._clear_trace_state_if_needed()

    def on_vbw_mode(self, mode: str):
        self._set_cfg(vbw_mode=mode)
        self.worker.queue_config({"vbw_mode": mode})
        self.vbw_edit.setEnabled(mode == "Manual")

    def on_vbw_changed(self):
        try:
            vbw = float(self.vbw_edit.text().strip())
        except ValueError:
            vbw = 0.0
        self._set_cfg(vbw_hz=vbw)
        self.worker.queue_config({"vbw_hz": vbw})

    def on_detector_changed(self, mode: str):
        self._set_cfg(detector=mode)
        self.worker.queue_config({"detector": mode})

    def on_overlap_changed(self):
        try:
            overlap = float(self.overlap_edit.text().strip())
        except ValueError:
            overlap = self.cfg.overlap
        overlap = min(0.95, max(0.0, overlap))
        self.overlap_edit.setText(f"{overlap:.2f}")
        self._set_cfg(overlap=overlap)
        self.worker.queue_config({"overlap": overlap})

    def on_buffer_factor_changed(self):
        try:
            buffer_factor = int(self.buffer_factor_edit.text().strip())
        except ValueError:
            buffer_factor = self.cfg.buffer_factor
        buffer_factor = max(1, buffer_factor)
        self.buffer_factor_edit.setText(str(buffer_factor))
        self._set_cfg(buffer_factor=buffer_factor)
        self.worker.queue_config({"buffer_factor": buffer_factor})

    def on_update_rate_changed(self):
        try:
            hz = float(self.update_hz_edit.text().strip())
        except ValueError:
            hz = max(0.1, 1000.0 / float(self.cfg.update_ms))
        hz = max(0.1, min(60.0, hz))
        self.update_hz_edit.setText(f"{hz:.1f}")
        update_ms = int(max(50, min(2000, 1000.0 / hz)))
        self._set_cfg(update_ms=update_ms)

    def on_trace1_mode_changed(self, mode: str):
        self._set_cfg(trace_type=mode)
        if mode != "Average":
            self.avg_trace = None
        if mode != "Max Hold":
            self.max_trace = None
        if mode != "Min Hold":
            self.min_trace = None
        self._update_trace1_pen(mode)

    def on_trace2_mode_changed(self, mode: str):
        enabled = mode == "Max Hold"
        self._set_cfg(trace2_enabled=enabled)
        self.clear_trace2_btn.setEnabled(enabled)
        if not enabled:
            self.trace2_max = None
            self.curve_trace2.hide()
        self._update_trace_legend()

    def on_avg_changed(self):
        try:
            avg_count = max(1, int(float(self.avg_edit.text().strip())))
        except ValueError:
            avg_count = 10
        self._set_cfg(avg_count=avg_count, avg_mode=self.avg_mode_cb.currentText())

    def on_amp_mode_changed(self, mode: str):
        # Only allow calibrated modes if a calibration offset is available.
        if mode != "dBFS" and not self.cal.has_calibration():
            self.amp_mode_cb.setCurrentText("dBFS")
            return
        self.plot.setLabel("left", "Magnitude", units=mode)

    def on_auto_ref_toggled(self, checked: bool):
        self.auto_scale_enabled = checked
        self.ref_edit.setEnabled(not checked)
        if not checked:
            try:
                self.cfg.ref_level_db = float(self.ref_edit.text().strip())
            except ValueError:
                pass

    def on_hold_across_changed(self, checked: bool):
        self.hold_across_settings = checked

    def on_scale_changed(self):
        try:
            scale = float(self.scale_edit.text().strip())
        except ValueError:
            scale = self.cfg.display_range_db / 10.0
        if scale <= 0:
            scale = 10.0
        self.cfg.display_range_db = scale * 10.0
        self._update_range_label()
        if self.latest_freqs is not None:
            self._update_axis_ticks(self.latest_freqs)

    def on_ref_changed(self):
        try:
            self.cfg.ref_level_db = float(self.ref_edit.text().strip())
        except ValueError:
            pass

    def _update_range_label(self):
        self.range_label.setText(f"Range: {self.cfg.display_range_db:.0f} dB")

    def _clear_trace_state(self):
        self.avg_trace = None
        self.max_trace = None
        self.min_trace = None
        self.trace2_max = None

    def _clear_trace_state_if_needed(self):
        if not self.hold_across_settings:
            self._clear_trace_state()
            self.status.setText("Trace holds cleared (settings changed)")

    def _update_trace1_pen(self, mode: str) -> None:
        if mode == "Average":
            self.curve_trace1.setPen(pg.mkPen("#66d16f", width=1))
        elif mode == "Max Hold":
            self.curve_trace1.setPen(pg.mkPen("#ffd200", width=1))
        elif mode == "Min Hold":
            self.curve_trace1.setPen(pg.mkPen("#8ad8ff", width=1))
        else:
            self.curve_trace1.setPen(pg.mkPen("w", width=1))

        self._update_trace_legend()

    def _update_trace_legend(self) -> None:
        if not self.trace_legend_enabled:
            self.trace_legend_item.hide()
            return
        trace1 = self.trace1_mode_cb.currentText()
        trace2 = self.trace2_mode_cb.currentText()
        lines = [
            f"Trace1: {trace1}",
            "Live: white  Avg: green  Max Hold: yellow",
        ]
        if trace2 != "Off":
            lines.append("Trace2: Max Hold (yellow dash)")
        self.trace_legend_item.setText("\n".join(lines))
        self.trace_legend_item.show()

    def on_clear_trace1(self):
        self.avg_trace = None
        self.max_trace = None
        self.min_trace = None

    def on_clear_trace2(self):
        self.trace2_max = None

    def on_run_stop(self):
        self.run_state = not self.run_state
        self.run_stop_btn.setText("Stop" if self.run_state else "Run")
        if self.run_state:
            self.single_pending = False

    def on_single(self):
        self.single_pending = True
        self.run_state = True
        self.run_stop_btn.setText("Stop")

    def apply_preset(self, preset_name: str):
        if preset_name == "Fast View":
            span_hz = 10_000_000
            fft_size = 2048
            update_hz = 15.0
            window = "Hann"
            detector = "RMS"
            overlap = 0.5
            vbw_mode = "Auto"
            vbw_hz = self.cfg.vbw_hz
        elif preset_name == "Wide Scan":
            span_hz = 20_000_000
            fft_size = 8192
            update_hz = 10.0
            window = "Blackman Harris"
            detector = "RMS"
            overlap = 0.5
            vbw_mode = "Auto"
            vbw_hz = self.cfg.vbw_hz
        else:
            span_hz = 5_000_000
            fft_size = 16384
            update_hz = 5.0
            window = "Blackman Harris"
            detector = self.measure_detector_cb.currentText()
            overlap = 0.5
            vbw_mode = "Manual"
            vbw_hz = 2000.0

        self._set_cfg(
            sample_rate_hz=span_hz,
            rf_bw_hz=span_hz,
            fft_size=fft_size,
            update_ms=int(max(50, min(2000, 1000.0 / update_hz))),
            window=window,
            detector=detector,
            overlap=overlap,
            vbw_mode=vbw_mode,
            vbw_hz=vbw_hz,
            rbw_mode="Manual",
        )
        self.worker.queue_config(
            {
                "span_hz": span_hz,
                "rf_bw_hz": span_hz,
                "fft_size": fft_size,
                "buffer_factor": self.cfg.buffer_factor,
                "window": window,
                "detector": detector,
                "overlap": overlap,
                "vbw_mode": vbw_mode,
                "vbw_hz": vbw_hz,
            }
        )

        temp_proc = SpectrumProcessor(fft_size, window)
        self.cfg.rbw_hz = temp_proc.rbw_hz(float(span_hz))

        self._sync_span_edit()
        self._sync_rfbw_edit()
        self.rbw_mode_cb.setCurrentText(self.cfg.rbw_mode)
        self.rbw_edit.setText(f"{self.cfg.rbw_hz:.0f}")
        self.vbw_mode_cb.setCurrentText(self.cfg.vbw_mode)
        self.vbw_edit.setText(f"{self.cfg.vbw_hz:.0f}")
        self.detector_cb.setCurrentText(self.cfg.detector)
        self.window_cb.setCurrentText(self.cfg.window)
        self.overlap_edit.setText(f"{self.cfg.overlap:.2f}")
        self.update_hz_edit.setText(f"{1000.0 / float(self.cfg.update_ms):.1f}")

        self._clear_trace_state()
        self.snap_x_to_span()
        self.status.setText(f"Preset applied: {preset_name}")

    def on_dc_remove(self, checked: bool):
        self._set_cfg(dc_remove=checked)

    def on_dc_blank_changed(self):
        try:
            bins = max(0, int(float(self.dc_blank_edit.text().strip())))
        except ValueError:
            bins = 1
        self._set_cfg(dc_blank_bins=bins)

    def _apply_calibration_values(self, offset_text: str, gain_text: str) -> None:
        offset_text = offset_text.strip()
        self.cal.dbfs_to_dbm_offset = float(offset_text) if offset_text else None
        self.cal.external_gain_db = float(gain_text.strip()) if gain_text.strip() else 0.0
        self.cal.save()
        self._update_amp_modes()

    def _calibrate_from_peak(self, tone_dbm_text: str, gain_text: str) -> None:
        if self.latest_power is None:
            raise ValueError("No data to calibrate")
        tone_dbm = float(tone_dbm_text.strip())
        peak_dbfs = float(np.max(self.power_to_dbfs(self.latest_power)))
        self.cal.dbfs_to_dbm_offset = tone_dbm - peak_dbfs
        self.cal.external_gain_db = float(gain_text.strip()) if gain_text.strip() else 0.0
        self.cal.save()
        self._update_amp_modes()

    def on_marker_peak(self):
        if self.latest_display is None:
            return
        self.marker1 = int(np.argmax(self.latest_display))
        self._update_markers()

    def on_marker_left(self):
        self._move_marker_peak(direction=-1)

    def on_marker_right(self):
        self._move_marker_peak(direction=1)

    def on_marker_clear(self):
        self.marker1 = None
        self.marker2 = None
        self.marker1_line.hide()
        self.marker2_line.hide()
        self.marker_info.setText("M1: --\nM2: --\nΔ : --\nNoise: --")

    def on_marker_to_center(self):
        if self.marker1 is None or self.latest_freqs is None:
            return
        target = int(self.latest_freqs[self.marker1])
        with QtCore.QMutexLocker(self.worker._lock):
            self.sdr.set_center_hz(target)
        self._sync_center_edit()
        self.snap_x_to_span()

    def on_open_calibration(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Calibration")
        layout = QtWidgets.QVBoxLayout(dialog)
        form = QtWidgets.QFormLayout()

        # Local edits apply on Save; peak calibration is a helper action.
        offset_edit = QtWidgets.QLineEdit(
            "" if self.cal.dbfs_to_dbm_offset is None else f"{self.cal.dbfs_to_dbm_offset:.2f}"
        )
        offset_edit.setAlignment(QtCore.Qt.AlignRight)
        gain_edit = QtWidgets.QLineEdit(f"{self.cal.external_gain_db:.2f}")
        gain_edit.setAlignment(QtCore.Qt.AlignRight)
        tone_edit = QtWidgets.QLineEdit("-30")
        tone_edit.setAlignment(QtCore.Qt.AlignRight)

        form.addRow("dBFS to dBm offset", offset_edit)
        form.addRow("External gain (dB)", gain_edit)
        form.addRow("Calibrate from peak (tone dBm)", tone_edit)
        layout.addLayout(form)

        calibrate_btn = QtWidgets.QPushButton("Calibrate from peak")
        layout.addWidget(calibrate_btn)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        layout.addWidget(buttons)

        def apply_calibration():
            try:
                self._apply_calibration_values(offset_edit.text(), gain_edit.text())
                self.status.setText("Calibration updated")
                dialog.accept()
            except Exception as exc:
                self.status.setText(f"Calibration error: {exc}")

        def calibrate_from_peak():
            try:
                self._calibrate_from_peak(tone_edit.text(), gain_edit.text())
                if self.cal.dbfs_to_dbm_offset is not None:
                    offset_edit.setText(f"{self.cal.dbfs_to_dbm_offset:.2f}")
                self.status.setText("Calibration updated from peak")
            except Exception as exc:
                self.status.setText(f"Calibration error: {exc}")

        calibrate_btn.clicked.connect(calibrate_from_peak)
        buttons.accepted.connect(apply_calibration)
        buttons.rejected.connect(dialog.reject)
        dialog.exec()

    def on_reset_state(self):
        if os.path.exists(STATE_PATH):
            try:
                os.remove(STATE_PATH)
            except OSError:
                pass
        default_cfg = SpectrumConfig(uri=self.cfg.uri)
        self.state = {}
        self._set_cfg(**default_cfg.__dict__)
        self.help_overlays_enabled = True
        self.trace_legend_enabled = True
        self.spectrogram_enabled = False
        self.spectrogram_rate = 15.0
        self.spectrogram_span_s = 200.0 / self.spectrogram_rate
        self.spectrogram_rows = max(1, int(round(self.spectrogram_span_s * self.spectrogram_rate)))
        self.cfg.spectrogram_mode = default_cfg.spectrogram_mode
        self.spectrogram_scale_mode = "Auto (5-95%)"
        self.spectrogram_cmap = "viridis"
        self.spectrogram_min_db = -120.0
        self.spectrogram_max_db = 0.0
        self.spectrogram_floor_db = None
        self.help_overlay_action.setChecked(self.help_overlays_enabled)
        self.trace_legend_action.setChecked(self.trace_legend_enabled)
        self.spectrogram_action.setChecked(self.spectrogram_enabled)
        self._apply_initial_state()
        self.status.setText("State reset to defaults")

    def on_save_trace_csv(self):
        if self.latest_freqs is None or self.latest_display is None:
            self.status.setText("No trace data to save")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save trace CSV", "spectrum-trace.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        data = np.column_stack((self.latest_freqs, self.latest_display))
        header = "frequency_hz,amplitude"
        np.savetxt(path, data, delimiter=",", header=header, comments="")
        self.status.setText(f"Saved trace to {path}")

    def on_peak_table_clicked(self, row: int, _column: int):
        if row < 0 or row >= len(self.peak_table_indices):
            return
        self.marker1 = self.peak_table_indices[row]
        self._update_markers()

    def on_save_screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save screenshot", "spectrum.png", "PNG Files (*.png)"
        )
        if not path:
            return
        exporter = ImageExporter(self.plot)
        exporter.export(path)
        self.status.setText(f"Saved screenshot to {path}")

    def on_toggle_help_overlays(self, checked: bool):
        self.help_overlays_enabled = checked
        if not checked:
            self.help_overlay.hide()

    def on_toggle_trace_legend(self, checked: bool):
        self.trace_legend_enabled = checked
        self._update_trace_legend()

    def on_toggle_spectrogram(self, checked: bool):
        self.spectrogram_enabled = checked
        self._apply_spectrogram_visibility()

    def on_spectrogram_speed(self, value: str):
        try:
            self.spectrogram_rate = float(value)
        except ValueError:
            self.spectrogram_rate = 15.0
        self.spectrogram_rows = max(1, int(round(self.spectrogram_span_s * self.spectrogram_rate)))
        self.spectrogram_buffer = None
        self.spectrogram_perf_label.setVisible(False)

    def on_spectrogram_depth(self, value: str):
        try:
            self.spectrogram_span_s = float(value)
        except ValueError:
            self.spectrogram_span_s = max(1.0, self.spectrogram_rows / max(self.spectrogram_rate, 1e-3))
        self.spectrogram_rows = max(1, int(round(self.spectrogram_span_s * self.spectrogram_rate)))
        self.spectrogram_buffer = None

    def on_spectrogram_mode(self, value: str):
        self._set_cfg(spectrogram_mode=value)

    def on_spectrogram_scale_mode(self, value: str):
        self.spectrogram_scale_mode = value
        if value == "Fixed floor":
            self.spectrogram_floor_db = None
        self._update_spectrogram_scale_controls()

    def on_spectrogram_cmap(self, value: str):
        self.spectrogram_cmap = value
        self._update_spectrogram_colormap()

    def on_spectrogram_range(self):
        try:
            self.spectrogram_min_db = float(self.spectrogram_min_edit.text().strip())
        except ValueError:
            self.spectrogram_min_db = -120.0
        try:
            self.spectrogram_max_db = float(self.spectrogram_max_edit.text().strip())
        except ValueError:
            self.spectrogram_max_db = 0.0

    def _update_spectrogram_colormap(self) -> None:
        cmap = pg.colormap.get(self.spectrogram_cmap)
        self.spectrogram_image.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))

    def _update_spectrogram_scale_controls(self) -> None:
        manual = self.spectrogram_scale_mode == "Manual (Expert)"
        self.spectrogram_min_edit.setEnabled(manual and self.spectrogram_enabled)
        self.spectrogram_max_edit.setEnabled(manual and self.spectrogram_enabled)

    def zoom_span_at(self, center_hz: float, factor: float) -> None:
        # Zoom span around cursor and enqueue SDR changes safely.
        span = int(max(1_000, min(20_000_000, self.cfg.sample_rate_hz * factor)))
        with QtCore.QMutexLocker(self.worker._lock):
            self.sdr.set_center_hz(int(center_hz))
        rf_bw = max(span, self.cfg.rf_bw_hz)
        self._set_cfg(sample_rate_hz=span, rf_bw_hz=rf_bw)
        self.worker.queue_config({"span_hz": span, "rf_bw_hz": rf_bw})
        self._sync_center_edit()
        self._sync_span_edit()
        self._sync_rfbw_edit()
        self._apply_rbw_strategy()
        self.snap_x_to_span()

    def _move_marker_peak(self, direction: int):
        if self.latest_display is None:
            return
        if self.marker1 is None:
            self.on_marker_peak()
            return
        idx = self.marker1
        peaks = self.latest_peaks.tolist() if self.latest_peaks is not None else []
        if not peaks:
            return
        if direction < 0:
            left_peaks = [p for p in peaks if p < idx]
            if left_peaks:
                self.marker1 = left_peaks[-1]
        else:
            right_peaks = [p for p in peaks if p > idx]
            if right_peaks:
                self.marker1 = right_peaks[0]
        self._update_markers()

    def _compute_peaks(self, data: np.ndarray) -> np.ndarray:
        if len(data) < 3:
            return np.array([], dtype=int)
        peaks = np.where((data[1:-1] > data[:-2]) & (data[1:-1] > data[2:]))[0] + 1
        return peaks.astype(int)

    def _update_peak_table(self) -> None:
        self.peak_table.clearContents()
        self.peak_table_indices = []
        if self.latest_display is None or self.latest_freqs is None or self.latest_peaks is None:
            return
        # Rank top peaks by amplitude for quick marker placement.
        peaks = self.latest_peaks
        if len(peaks) == 0:
            return
        peak_vals = self.latest_display[peaks]
        order = np.argsort(peak_vals)[::-1][:5]
        for row, idx in enumerate(order):
            peak_idx = int(peaks[idx])
            self.peak_table_indices.append(peak_idx)
            freq = self._format_freq(self.latest_freqs[peak_idx])
            amp = self._format_amp(self.latest_display[peak_idx])
            self.peak_table.setItem(row, 0, QtWidgets.QTableWidgetItem(freq))
            self.peak_table.setItem(row, 1, QtWidgets.QTableWidgetItem(amp))

    def _decimate_for_display(
        self, freqs: np.ndarray, display: np.ndarray, target_points: int = 2001
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(display)
        if n <= target_points:
            return freqs, display
        bins = max(1, target_points // 2)
        stride = n / float(bins)
        out_f = []
        out_y = []
        for i in range(bins):
            start = int(i * stride)
            end = int(min(n, (i + 1) * stride))
            if end <= start:
                continue
            seg = display[start:end]
            seg_f = freqs[start:end]
            min_idx = int(np.argmin(seg))
            max_idx = int(np.argmax(seg))
            if min_idx <= max_idx:
                order = [min_idx, max_idx]
            else:
                order = [max_idx, min_idx]
            for idx in order:
                out_f.append(seg_f[idx])
                out_y.append(seg[idx])
        return np.array(out_f), np.array(out_y)

    def _apply_rbw_strategy(self):
        fs = float(self.cfg.sample_rate_hz)
        window = self.cfg.window
        current_fft = self.cfg.fft_size

        if self.cfg.rbw_mode == "Auto":
            # Aim for typical SA point count (approx 2001).
            desired_points = 2001
            target_rbw = fs / desired_points
        else:
            target_rbw = max(1.0, self.cfg.rbw_hz)

        # Use window ENBW to map desired RBW to FFT size.
        temp_proc = SpectrumProcessor(current_fft, window)
        enbw_bins = temp_proc.enbw_bins
        target_n = int(round((fs * enbw_bins) / target_rbw))
        target_n = max(1024, min(262144, target_n))

        # Restrict to power-of-two FFT sizes for speed.
        allowed = [2 ** n for n in range(10, 19)]
        fft_size = min(allowed, key=lambda n: abs(n - target_n))

        buffer_factor = max(2, self.cfg.buffer_factor)
        self._set_cfg(fft_size=fft_size, buffer_factor=buffer_factor)
        self.worker.queue_config({"fft_size": fft_size, "buffer_factor": buffer_factor, "window": window})
        temp_proc = SpectrumProcessor(fft_size, window)
        self.cfg.rbw_hz = temp_proc.rbw_hz(fs)
        self.rbw_edit.setText(f"{self.cfg.rbw_hz:.0f}")
        self.status.setText(f"RBW updated: {self.cfg.rbw_hz:.0f} Hz")
        self._clear_trace_state_if_needed()

    def _update_ref_level(self, display: np.ndarray) -> None:
        if not self.auto_scale_enabled:
            return
        now = time.monotonic()
        peak = float(np.max(display))
        target = peak + self.auto_ref_headroom_db
        ref_level = self.cfg.ref_level_db

        if abs(target - ref_level) < self.auto_ref_hysteresis_db:
            return

        dt = max(1e-3, now - self.last_auto_ref_update) if self.last_auto_ref_update else 0.1
        self.last_auto_ref_update = now
        alpha = 1.0 - math.exp(-dt / self.auto_ref_tau_s)
        ref_level = ref_level + (target - ref_level) * alpha
        delta = ref_level - self.cfg.ref_level_db
        if delta > self.auto_ref_max_step_db:
            ref_level = self.cfg.ref_level_db + self.auto_ref_max_step_db
        elif delta < -self.auto_ref_max_step_db:
            ref_level = self.cfg.ref_level_db - self.auto_ref_max_step_db
        self.cfg.ref_level_db = ref_level
        self.ref_edit.setText(f"{ref_level:.1f}")

    def _nice_step(self, span: float) -> float:
        base = span / 10.0
        exp = math.floor(math.log10(base)) if base > 0 else 0
        magnitude = 10 ** exp
        frac = base / magnitude if magnitude != 0 else base
        if frac <= 1:
            nice = 1
        elif frac <= 2:
            nice = 2
        elif frac <= 5:
            nice = 5
        else:
            nice = 10
        return nice * magnitude

    def _update_axis_ticks(self, freqs: np.ndarray) -> None:
        if len(freqs) < 2:
            return
        span = float(freqs[-1] - freqs[0])
        major = self._nice_step(span)
        minor = major / 5.0
        axis_x = self.plot.getAxis("bottom")
        axis_x.setTickSpacing(major=major, minor=minor)

        scale_div = max(1.0, self.cfg.display_range_db / 10.0)
        axis_y = self.plot.getAxis("left")
        axis_y.setTickSpacing(major=scale_div, minor=scale_div / 2.0)

    def on_mouse_moved(self, evt):
        pos = evt[0]
        vb = self.plot.vb

        if not vb.sceneBoundingRect().contains(pos):
            self.hover_text.setText("")
            return

        mouse_point = vb.mapSceneToView(pos)
        fx = float(mouse_point.x())
        self.vline.setPos(fx)

        hit = self.hover.nearest_bin(fx)
        if hit is None:
            text = f"{fx / 1e6:.6f} MHz"
        else:
            f_bin, m_bin, _ = hit
            text = f"{f_bin / 1e6:.6f} MHz\n{m_bin:.2f} {self.amp_mode_cb.currentText()}"

        (xmin, xmax), (ymin, ymax) = vb.viewRange()
        dx = (xmax - xmin) * 0.01
        dy = (ymax - ymin) * 0.05

        self.hover_text.setText(text)
        self.hover_text.setPos(mouse_point.x() + dx, mouse_point.y() - dy)

    def on_mouse_clicked(self, evt):
        if self.latest_freqs is None or self.latest_display is None:
            return
        mouse_event = evt[0]
        if mouse_event.button() not in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            return
        vb = self.plot.vb
        mouse_point = vb.mapSceneToView(mouse_event.scenePos())
        fx = float(mouse_point.x())
        idx = int(np.argmin(np.abs(self.latest_freqs - fx)))
        if mouse_event.button() == QtCore.Qt.LeftButton:
            self.marker1 = idx
        else:
            self.marker2 = idx
        self._update_markers()

    def _format_freq(self, freq_hz: float) -> str:
        if freq_hz >= 1e9:
            return f"{freq_hz/1e9:.6f} GHz"
        if freq_hz >= 1e6:
            return f"{freq_hz/1e6:.6f} MHz"
        if freq_hz >= 1e3:
            return f"{freq_hz/1e3:.3f} kHz"
        return f"{freq_hz:.0f} Hz"

    def _format_amp(self, value: float) -> str:
        return f"{value:.2f} {self.amp_mode_cb.currentText()}"

    def _update_markers(self):
        if self.latest_freqs is None or self.latest_display is None:
            return
        info = {"M1": "--", "M2": "--", "Δ": "--", "Noise": "--"}

        if self.marker1 is not None:
            f1 = self.latest_freqs[self.marker1]
            a1 = self.latest_display[self.marker1]
            self.marker1_line.setPos(f1)
            self.marker1_line.show()
            info["M1"] = f"{self._format_freq(f1):<14} {self._format_amp(a1)}"

            if self.latest_power is not None and self.latest_rbw is not None:
                noise_dbfs = float(self.power_to_dbfs(np.array([self.latest_power[self.marker1]]))[0])
                noise_dbfs_hz = noise_dbfs - 10.0 * math.log10(max(self.latest_rbw, 1e-12))
                info["Noise"] = f"{noise_dbfs:.2f} dBFS / {noise_dbfs_hz:.2f} dBFS/Hz"
        else:
            self.marker1_line.hide()

        if self.marker2 is not None:
            f2 = self.latest_freqs[self.marker2]
            a2 = self.latest_display[self.marker2]
            self.marker2_line.setPos(f2)
            self.marker2_line.show()
            info["M2"] = f"{self._format_freq(f2):<14} {self._format_amp(a2)}"
        else:
            self.marker2_line.hide()

        if self.marker1 is not None and self.marker2 is not None:
            # Delta marker readout and channel power between markers.
            df = abs(self.latest_freqs[self.marker2] - self.latest_freqs[self.marker1])
            da = self.latest_display[self.marker2] - self.latest_display[self.marker1]
            info["Δ"] = f"{self._format_freq(df):<14} {da:.2f} {self.amp_mode_cb.currentText()}"

            band_power = self._channel_power(self.marker1, self.marker2)
            if band_power is not None:
                info["Ch"] = f"Ch: {band_power:.2f} {self.amp_mode_cb.currentText()}"

        lines = [
            f"M1: {info.get('M1', '--')}",
            f"M2: {info.get('M2', '--')}",
            f"Δ : {info.get('Δ', '--')}",
            f"Noise: {info.get('Noise', '--')}",
        ]
        if "Ch" in info:
            lines.append(info["Ch"])
        self.marker_info.setText("\n".join(lines))

    def _channel_power(self, idx1: int, idx2: int) -> Optional[float]:
        if self.latest_power is None:
            return None
        lo = min(idx1, idx2)
        hi = max(idx1, idx2)
        # Integrate linear power between markers.
        power_sum = float(np.sum(self.latest_power[lo : hi + 1]))
        return self._power_to_display(np.array([power_sum]), rbw_hz=self.cfg.rbw_hz)[0]

    def power_to_dbfs(self, power: np.ndarray) -> np.ndarray:
        # dBFS is referenced to full-scale RMS power (normalized).
        return 10.0 * np.log10(np.maximum(power, 1e-20))

    def _power_to_display(self, power: np.ndarray, rbw_hz: float) -> np.ndarray:
        dbfs = self.power_to_dbfs(power)
        mode = self.amp_mode_cb.currentText()
        if mode == "dBFS":
            return dbfs
        if not self.cal.has_calibration():
            return dbfs
        # Apply calibration and external gain/loss.
        dbm = dbfs + float(self.cal.dbfs_to_dbm_offset) + float(self.cal.external_gain_db)
        if mode == "dBm":
            return dbm
        if mode == "dBm/Hz":
            # Normalize to 1 Hz using RBW (approximate ENBW).
            return dbm - 10.0 * np.log10(max(rbw_hz, 1e-12))
        return dbfs

    def _apply_vbw(self, power: np.ndarray, rbw_hz: float, timestamp: float) -> np.ndarray:
        mode = self.vbw_mode_cb.currentText()
        if mode == "Off":
            self.vbw_state = None
            self.last_update_ts = timestamp
            return power
        if mode == "Auto":
            vbw = rbw_hz / 10.0
        else:
            vbw = max(0.0, float(self.cfg.vbw_hz))
        if vbw <= 0.0:
            self.vbw_state = None
            self.last_update_ts = timestamp
            return power
        if self.last_update_ts is None:
            self.last_update_ts = timestamp
        dt = max(1e-3, timestamp - self.last_update_ts)
        self.last_update_ts = timestamp
        # VBW smoothing as EMA in linear power domain.
        alpha = math.exp(-dt * 2.0 * math.pi * vbw)
        if self.vbw_state is None or self.vbw_state.shape != power.shape:
            # Reset VBW state if FFT size changed (shape mismatch).
            self.vbw_state = power.copy()
        else:
            self.vbw_state = alpha * self.vbw_state + (1.0 - alpha) * power
        return self.vbw_state

    def _apply_spectrogram_visibility(self) -> None:
        if self.spectrogram_enabled:
            self.spectrogram_plot.show()
        else:
            self.spectrogram_plot.hide()
        # Disable/enable controls to match visibility.
        for widget in (
            self.spectrogram_mode_cb,
            self.spectrogram_speed_cb,
            self.spectrogram_depth_cb,
            self.spectrogram_scale_cb,
            self.spectrogram_cmap_cb,
            self.spectrogram_min_edit,
            self.spectrogram_max_edit,
        ):
            widget.setEnabled(self.spectrogram_enabled)
        if not self.spectrogram_enabled:
            self.spectrogram_perf_label.setVisible(False)
        self._update_spectrogram_scale_controls()

    def _update_spectrogram(self, freqs: np.ndarray, spec_db: np.ndarray, timestamp: float) -> None:
        if not self.spectrogram_enabled:
            return
        if self.spectrogram_rate <= 0:
            return
        if timestamp - self.spectrogram_last_ts < 1.0 / float(self.spectrogram_rate):
            return
        # Use decimated trace for performance and roll a fixed-size buffer.
        actual_dt = timestamp - self.spectrogram_last_ts if self.spectrogram_last_ts else 0.0
        self.spectrogram_last_ts = timestamp
        dec_freqs, dec_spec = self._decimate_for_display(freqs, spec_db)

        rows = int(self.spectrogram_rows)
        cols = len(dec_spec)
        if cols == 0 or rows <= 0:
            return
        if self.spectrogram_buffer is None or self.spectrogram_buffer.shape != (rows, cols):
            self.spectrogram_buffer = np.full((rows, cols), self.spectrogram_min_db, dtype=np.float32)
            self.spectrogram_floor_db = None
        self.spectrogram_buffer = np.roll(self.spectrogram_buffer, -1, axis=0)
        self.spectrogram_buffer[-1, :] = dec_spec.astype(np.float32)

        if self.spectrogram_scale_mode == "Manual (Expert)":
            levels = (self.spectrogram_min_db, self.spectrogram_max_db)
        else:
            if self.spectrogram_buffer.size:
                low, high = np.percentile(self.spectrogram_buffer, [5, 95])
            else:
                low, high = self.spectrogram_min_db, self.spectrogram_max_db
            if self.spectrogram_scale_mode == "Fixed floor":
                if self.spectrogram_floor_db is None:
                    self.spectrogram_floor_db = float(low)
                low = self.spectrogram_floor_db
            levels = (float(low), float(high))
        if levels[1] <= levels[0]:
            levels = (levels[0] - 1.0, levels[0] + 1.0)

        if actual_dt > 0.0:
            target_dt = 1.0 / float(self.spectrogram_rate)
            if actual_dt > target_dt * 1.5:
                new_rate = max(5.0, self.spectrogram_rate * 0.8)
                adjusted = self._set_combo_to_nearest(self.spectrogram_speed_cb, new_rate)
                if adjusted != self.spectrogram_rate:
                    self.spectrogram_rate = adjusted
                    self.spectrogram_rows = max(1, int(round(self.spectrogram_span_s * self.spectrogram_rate)))
                    self.spectrogram_buffer = None
                    self.spectrogram_perf_label.setVisible(True)
                    return
                self.spectrogram_perf_label.setVisible(True)

        self.spectrogram_image.setImage(self.spectrogram_buffer, autoLevels=False)
        self.spectrogram_image.setLevels(levels)
        if len(dec_freqs) >= 2:
            time_span = rows / max(self.spectrogram_rate, 1e-3)
            rect = QtCore.QRectF(
                dec_freqs[0],
                -time_span,
                dec_freqs[-1] - dec_freqs[0],
                time_span,
            )
            self.spectrogram_image.setRect(rect)
            self.spectrogram_plot.setXRange(dec_freqs[0], dec_freqs[-1], padding=0.0)

    def on_new_data(self, payload: Dict):
        # Store latest payload for the UI refresh timer.
        self.latest_payload = payload

    def _apply_average(self, power: np.ndarray) -> np.ndarray:
        count = max(1, self.cfg.avg_count)
        if self.avg_trace is None:
            self.avg_trace = power.copy()
        else:
            if self.cfg.avg_mode == "Log":
                avg_db = self.power_to_dbfs(self.avg_trace)
                curr_db = self.power_to_dbfs(power)
                avg_db = avg_db + (curr_db - avg_db) / float(count)
                self.avg_trace = np.power(10.0, avg_db / 10.0)
            else:
                self.avg_trace = self.avg_trace + (power - self.avg_trace) / float(count)
        return self.avg_trace

    def refresh_display(self):
        if self.latest_payload is None:
            return
        if not self.run_state and not self.single_pending:
            return
        payload = self.latest_payload
        power = payload["power"]
        rbw = payload["rbw"]
        timestamp = payload["timestamp"]
        freqs = payload["freqs"]
        spectrogram_db = payload.get("spectrogram_db")

        # Apply VBW smoothing in linear power.
        power = self._apply_vbw(power, rbw, timestamp)
        live_display = self._power_to_display(power, rbw_hz=rbw)
        if spectrogram_db is not None:
            self._update_spectrogram(freqs, spectrogram_db, timestamp)

        trace1_mode = self.cfg.trace_type
        if trace1_mode == "Average":
            power_trace = self._apply_average(power)
            display = self._power_to_display(power_trace, rbw_hz=rbw)
        elif trace1_mode == "Max Hold":
            if self.max_trace is None or len(self.max_trace) != len(power):
                self.max_trace = power.copy()
            else:
                self.max_trace = np.maximum(self.max_trace, power)
            display = self._power_to_display(self.max_trace, rbw_hz=rbw)
        elif trace1_mode == "Min Hold":
            if self.min_trace is None or len(self.min_trace) != len(power):
                self.min_trace = power.copy()
            else:
                self.min_trace = np.minimum(self.min_trace, power)
            display = self._power_to_display(self.min_trace, rbw_hz=rbw)
        else:
            display = live_display

        if self.cfg.trace2_enabled:
            if self.trace2_max is None or len(self.trace2_max) != len(power):
                self.trace2_max = power.copy()
            else:
                self.trace2_max = np.maximum(self.trace2_max, power)
            overlay_display = self._power_to_display(self.trace2_max, rbw_hz=rbw)
        else:
            overlay_display = None

        dec_freqs, dec_display = self._decimate_for_display(freqs, display)
        self.curve_trace1.setData(dec_freqs, dec_display)
        self.curve_trace1.show()

        if overlay_display is not None:
            dec_freqs2, dec_overlay = self._decimate_for_display(freqs, overlay_display)
            self.curve_trace2.setData(dec_freqs2, dec_overlay)
            self.curve_trace2.show()
        else:
            self.curve_trace2.hide()

        self.latest_power = power
        self.latest_display = display
        self.latest_freqs = freqs
        self.latest_rbw = rbw
        self.latest_peaks = self._compute_peaks(display)
        self._update_peak_table()
        self.hover.update_axis(freqs, display)
        self._update_markers()

        self._update_ref_level(display)
        span = float(freqs[-1] - freqs[0]) if len(freqs) > 1 else 0.0
        if self.last_axis_span != span or self.last_axis_fft != len(freqs):
            self._update_axis_ticks(freqs)
            self.last_axis_span = span
            self.last_axis_fft = len(freqs)

        ref_level = self.cfg.ref_level_db
        display_range = self.cfg.display_range_db
        self.plot.setYRange(ref_level - display_range, ref_level, padding=0.0)
        if self.trace_legend_enabled:
            (xmin, xmax), (ymin, ymax) = self.plot.viewRange()
            self.trace_legend_item.setPos(xmin + (xmax - xmin) * 0.02, ymax - (ymax - ymin) * 0.05)

        # Update status readout with current instrument state.
        self.cfg.rbw_hz = rbw
        self.enbw_label.setText(f"{payload['enbw_bins']:.2f}")
        self.status.setText(
            " ".join(
                [
                    f"LO {int(payload['lo'])}",
                    f"SR {int(payload['fs'])}",
                    f"RBW {rbw:.1f} Hz",
                    f"FFT {self.cfg.fft_size}",
                    f"Gain {payload['gain_db']} dB",
                    f"RF BW {int(payload['rf_bw'])}",
                ]
            )
        )

        if self.single_pending:
            self.single_pending = False
            self.run_state = False
            self.run_stop_btn.setText("Run")

def main():
    cfg = SpectrumConfig(uri="ip:192.168.2.1")
    app = pg.mkQApp("Pluto Spectrum Analyzer")
    win = SpectrumWindow(cfg)
    win.resize(1400, 820)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
