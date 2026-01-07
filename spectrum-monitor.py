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
    fft_size: int = 131072
    update_ms: int = 200
    buffer_factor: int = 4
    overlap: float = 0.5

    # UI responsiveness (hover).
    hover_rate_hz: int = 25

    # RBW controls.
    rbw_mode: str = "Auto"
    rbw_hz: float = 10_000.0
    window: str = "Hann"

    # VBW controls.
    vbw_mode: str = "Off"
    vbw_hz: float = 3_000.0

    # Detector/trace controls.
    detector: str = "RMS"
    trace_type: str = "Live"
    avg_count: int = 10
    avg_mode: str = "RMS"

    # Display scaling.
    ref_level_db: float = 0.0
    display_range_db: float = 100.0

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
            windowed = segment * self.window
            spectrum = np.fft.fftshift(np.fft.fft(windowed))
            # Normalize by coherent gain to keep dBFS stable across windows.
            power = (np.abs(spectrum) / (n * self.coherent_gain)) ** 2

            if dc_blank_bins > 0:
                # Blank bins around DC to suppress LO leakage.
                mid = n // 2
                lo = max(mid - dc_blank_bins, 0)
                hi = min(mid + dc_blank_bins + 1, n)
                power[lo:hi] = np.min(power)

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


class SpectrumWorker(QtCore.QThread):
    new_data = QtCore.pyqtSignal(dict)

    def __init__(self, sdr: PlutoSdr, proc: SpectrumProcessor, cfg: SpectrumConfig):
        super().__init__()
        self.sdr = sdr
        self.proc = proc
        self.cfg = cfg
        self._lock = QtCore.QMutex()
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        next_time = time.monotonic()
        while self._running:
            with QtCore.QMutexLocker(self._lock):
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
            freqs = self.proc.freqs_hz(fs, lo)
            rbw = self.proc.rbw_hz(fs)

            payload = {
                "freqs": freqs,
                "power": power,
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
            next_time += self.cfg.update_ms / 1000.0
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
        self.cfg.ref_level_db = float(self.state.get("ref_level_db", self.cfg.ref_level_db))
        self.cfg.display_range_db = float(self.state.get("display_range_db", self.cfg.display_range_db))
        self.cfg.rbw_mode = self.state.get("rbw_mode", self.cfg.rbw_mode)
        self.cfg.vbw_mode = self.state.get("vbw_mode", self.cfg.vbw_mode)
        self.cfg.rbw_hz = float(self.state.get("rbw_hz", self.cfg.rbw_hz))
        self.cfg.vbw_hz = float(self.state.get("vbw_hz", self.cfg.vbw_hz))
        self.cfg.window = self.state.get("window", self.cfg.window)
        self.cfg.detector = self.state.get("detector", self.cfg.detector)
        self.cfg.trace_type = self.state.get("trace_mode", self.cfg.trace_type)
        self.cfg.update_ms = int(self.state.get("update_ms", self.cfg.update_ms))
        self.auto_scale_enabled = bool(self.state.get("auto_scale", True))
        self.peak_hold_enabled = bool(self.state.get("peak_hold_enabled", False))

        self.setWindowTitle("Pluto Spectrum Viewer")

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
        self.latest_power: Optional[np.ndarray] = None
        self.latest_display: Optional[np.ndarray] = None
        self.latest_freqs: Optional[np.ndarray] = None
        self.latest_payload: Optional[Dict] = None
        self.marker1: Optional[int] = None
        self.marker2: Optional[int] = None
        self.last_ui_update_ts: float = 0.0
        self.last_auto_ref_update: float = 0.0
        self.auto_ref_tau_s: float = 0.8
        self.auto_ref_hysteresis_db: float = 1.5

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
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)
        layout.setSpacing(6)

        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(8)
        layout.addLayout(row1)
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(8)
        layout.addLayout(row2)

        rf_box = QtWidgets.QGroupBox("RF")
        rf_layout = QtWidgets.QGridLayout(rf_box)
        rf_layout.setHorizontalSpacing(6)
        rf_layout.setVerticalSpacing(4)

        rf_layout.addWidget(QtWidgets.QLabel("Center"), 0, 0)
        self.freq_edit = QtWidgets.QLineEdit("2437")
        self.freq_edit.setFixedWidth(90)
        rf_layout.addWidget(self.freq_edit, 0, 1)

        self.freq_unit = QtWidgets.QComboBox()
        self.freq_unit.addItems(["Hz", "kHz", "MHz", "GHz"])
        self.freq_unit.setCurrentText("MHz")
        self.freq_unit.setFixedWidth(70)
        rf_layout.addWidget(self.freq_unit, 0, 2)

        self.set_btn = QtWidgets.QPushButton("Set")
        rf_layout.addWidget(self.set_btn, 0, 3)

        rf_layout.addWidget(QtWidgets.QLabel("Span"), 1, 0)
        self.span_cb = QtWidgets.QComboBox()
        self.span_cb.addItems(["1 MHz", "5 MHz", "10 MHz", "20 MHz", "30 MHz", "40 MHz", "50 MHz"])
        self.span_cb.setCurrentText("20 MHz")
        self.span_cb.setFixedWidth(90)
        rf_layout.addWidget(self.span_cb, 1, 1)

        self.apply_span_btn = QtWidgets.QPushButton("Apply")
        rf_layout.addWidget(self.apply_span_btn, 1, 2)

        rf_layout.addWidget(QtWidgets.QLabel("RF BW"), 1, 3)
        self.rfbw_edit = QtWidgets.QLineEdit(str(self.cfg.rf_bw_hz))
        self.rfbw_edit.setFixedWidth(90)
        rf_layout.addWidget(self.rfbw_edit, 1, 4)

        self.apply_rfbw_btn = QtWidgets.QPushButton("Set")
        rf_layout.addWidget(self.apply_rfbw_btn, 1, 5)

        rf_layout.addWidget(QtWidgets.QLabel("Gain mode"), 2, 0)
        self.gainmode_cb = QtWidgets.QComboBox()
        self.gainmode_cb.addItems(["manual", "fast_attack", "slow_attack", "hybrid"])
        self.gainmode_cb.setCurrentText(self.cfg.gain_mode)
        rf_layout.addWidget(self.gainmode_cb, 2, 1)

        rf_layout.addWidget(QtWidgets.QLabel("Gain dB"), 2, 2)
        self.gain_edit = QtWidgets.QLineEdit(str(self.cfg.gain_db))
        self.gain_edit.setFixedWidth(60)
        rf_layout.addWidget(self.gain_edit, 2, 3)

        self.apply_gain_btn = QtWidgets.QPushButton("Apply")
        rf_layout.addWidget(self.apply_gain_btn, 2, 4)

        self.measurement_cb = QtWidgets.QCheckBox("Measurement mode")
        self.measurement_cb.setChecked(self.cfg.measurement_mode)
        rf_layout.addWidget(self.measurement_cb, 3, 0, 1, 3)

        row1.addWidget(rf_box, 3)

        sweep_box = QtWidgets.QGroupBox("Sweep")
        sweep_layout = QtWidgets.QGridLayout(sweep_box)
        sweep_layout.setHorizontalSpacing(6)
        sweep_layout.setVerticalSpacing(4)

        sweep_layout.addWidget(QtWidgets.QLabel("RBW"), 0, 0)
        self.rbw_mode_cb = QtWidgets.QComboBox()
        self.rbw_mode_cb.addItems(["Auto", "Manual"])
        self.rbw_mode_cb.setCurrentText(self.cfg.rbw_mode)
        sweep_layout.addWidget(self.rbw_mode_cb, 0, 1)

        self.rbw_edit = QtWidgets.QLineEdit(f"{self.cfg.rbw_hz:.0f}")
        self.rbw_edit.setFixedWidth(80)
        sweep_layout.addWidget(self.rbw_edit, 0, 2)

        self.rbw_apply_btn = QtWidgets.QPushButton("Apply")
        sweep_layout.addWidget(self.rbw_apply_btn, 0, 3)

        sweep_layout.addWidget(QtWidgets.QLabel("VBW"), 1, 0)
        self.vbw_mode_cb = QtWidgets.QComboBox()
        self.vbw_mode_cb.addItems(["Off", "Auto", "Manual"])
        self.vbw_mode_cb.setCurrentText(self.cfg.vbw_mode)
        sweep_layout.addWidget(self.vbw_mode_cb, 1, 1)

        self.vbw_edit = QtWidgets.QLineEdit(f"{self.cfg.vbw_hz:.0f}")
        self.vbw_edit.setFixedWidth(80)
        sweep_layout.addWidget(self.vbw_edit, 1, 2)

        sweep_layout.addWidget(QtWidgets.QLabel("Window"), 2, 0)
        self.window_cb = QtWidgets.QComboBox()
        self.window_cb.addItems(["Hann", "Blackman Harris", "Flat top"])
        self.window_cb.setCurrentText(self.cfg.window)
        sweep_layout.addWidget(self.window_cb, 2, 1)

        sweep_layout.addWidget(QtWidgets.QLabel("Detector"), 2, 2)
        self.detector_cb = QtWidgets.QComboBox()
        self.detector_cb.addItems(["Sample", "Peak", "RMS"])
        self.detector_cb.setCurrentText(self.cfg.detector)
        sweep_layout.addWidget(self.detector_cb, 2, 3)

        sweep_layout.addWidget(QtWidgets.QLabel("Update (ms)"), 3, 0)
        self.update_ms_edit = QtWidgets.QLineEdit(str(self.cfg.update_ms))
        self.update_ms_edit.setFixedWidth(70)
        sweep_layout.addWidget(self.update_ms_edit, 3, 1)

        row1.addWidget(sweep_box, 2)

        trace_box = QtWidgets.QGroupBox("Trace / Display")
        trace_layout = QtWidgets.QGridLayout(trace_box)
        trace_layout.setHorizontalSpacing(6)
        trace_layout.setVerticalSpacing(4)

        self.trace_group = QtWidgets.QButtonGroup(trace_box)
        self.trace_live_rb = QtWidgets.QRadioButton("Live")
        self.trace_avg_rb = QtWidgets.QRadioButton("Average")
        self.trace_peak_rb = QtWidgets.QRadioButton("Peak Hold")
        self.trace_group.addButton(self.trace_live_rb)
        self.trace_group.addButton(self.trace_avg_rb)
        self.trace_group.addButton(self.trace_peak_rb)
        self.trace_live_rb.setChecked(True)

        trace_layout.addWidget(QtWidgets.QLabel("Trace mode"), 0, 0)
        trace_layout.addWidget(self.trace_live_rb, 0, 1)
        trace_layout.addWidget(self.trace_avg_rb, 0, 2)
        trace_layout.addWidget(self.trace_peak_rb, 0, 3)

        self.peak_hold_toggle = QtWidgets.QCheckBox("Peak Hold ON")
        self.peak_hold_toggle.setChecked(False)
        trace_layout.addWidget(self.peak_hold_toggle, 1, 0, 1, 2)

        self.clear_peak_btn = QtWidgets.QPushButton("Clear Peak Hold")
        trace_layout.addWidget(self.clear_peak_btn, 1, 2, 1, 2)

        trace_layout.addWidget(QtWidgets.QLabel("Avg count"), 2, 0)
        self.avg_edit = QtWidgets.QLineEdit(str(self.cfg.avg_count))
        self.avg_edit.setFixedWidth(60)
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
        self.ref_edit = QtWidgets.QLineEdit(f"{self.cfg.ref_level_db:.1f}")
        self.ref_edit.setFixedWidth(70)
        trace_layout.addWidget(self.ref_edit, 3, 3)

        trace_layout.addWidget(QtWidgets.QLabel("Scale (dB/div)"), 4, 0)
        self.scale_edit = QtWidgets.QLineEdit("10")
        self.scale_edit.setFixedWidth(60)
        trace_layout.addWidget(self.scale_edit, 4, 1)

        self.range_label = QtWidgets.QLabel("Range: 100 dB")
        trace_layout.addWidget(self.range_label, 4, 2, 1, 2)

        self.auto_ref_cb = QtWidgets.QCheckBox("Auto Scale")
        self.auto_ref_cb.setChecked(True)
        trace_layout.addWidget(self.auto_ref_cb, 5, 0, 1, 2)

        self.dc_remove_cb = QtWidgets.QCheckBox("DC remove")
        self.dc_remove_cb.setChecked(self.cfg.dc_remove)
        trace_layout.addWidget(self.dc_remove_cb, 5, 2)

        trace_layout.addWidget(QtWidgets.QLabel("DC blank bins"), 5, 3)
        self.dc_blank_edit = QtWidgets.QLineEdit(str(self.cfg.dc_blank_bins))
        self.dc_blank_edit.setFixedWidth(50)
        trace_layout.addWidget(self.dc_blank_edit, 5, 4)

        row2.addWidget(trace_box, 3)

        markers_box = QtWidgets.QGroupBox("Markers")
        markers_layout = QtWidgets.QVBoxLayout(markers_box)
        markers_layout.setSpacing(4)
        self.marker_info = QtWidgets.QLabel("M1: --\nM2: --\nΔ : --")
        self.marker_info.setStyleSheet("QLabel { font-family: monospace; }")
        markers_layout.addWidget(self.marker_info)

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

        row2.addWidget(markers_box, 2)

        cal_box = QtWidgets.QGroupBox("Calibration")
        cal_layout = QtWidgets.QGridLayout(cal_box)
        cal_layout.setHorizontalSpacing(6)
        cal_layout.setVerticalSpacing(4)

        cal_layout.addWidget(QtWidgets.QLabel("Offset dB"), 0, 0)
        self.cal_offset_edit = QtWidgets.QLineEdit(
            "" if self.cal.dbfs_to_dbm_offset is None else f"{self.cal.dbfs_to_dbm_offset:.2f}"
        )
        self.cal_offset_edit.setFixedWidth(70)
        cal_layout.addWidget(self.cal_offset_edit, 0, 1)

        cal_layout.addWidget(QtWidgets.QLabel("Ext gain"), 0, 2)
        self.cal_gain_edit = QtWidgets.QLineEdit(f"{self.cal.external_gain_db:.2f}")
        self.cal_gain_edit.setFixedWidth(60)
        cal_layout.addWidget(self.cal_gain_edit, 0, 3)

        self.apply_cal_btn = QtWidgets.QPushButton("Apply")
        cal_layout.addWidget(self.apply_cal_btn, 0, 4)

        cal_layout.addWidget(QtWidgets.QLabel("Tone dBm"), 1, 0)
        self.tone_dbm_edit = QtWidgets.QLineEdit("-30")
        self.tone_dbm_edit.setFixedWidth(60)
        cal_layout.addWidget(self.tone_dbm_edit, 1, 1)

        self.calibrate_btn = QtWidgets.QPushButton("Calibrate")
        cal_layout.addWidget(self.calibrate_btn, 1, 2, 1, 2)

        row2.addWidget(cal_box, 2)

        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("QLabel { padding: 4px; }")
        layout.addWidget(self.status)

        self.plotw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plotw, 1)

        self.plot = self.plotw.addPlot()
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "Magnitude", units="dBFS")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setMouseEnabled(x=True, y=False)

        self.curve_live = self.plot.plot(pen=pg.mkPen("w", width=1))
        self.curve_avg = self.plot.plot(pen=pg.mkPen("c", width=1))
        self.curve_peak = self.plot.plot(pen=pg.mkPen("y", width=2, style=QtCore.Qt.DashLine))
        self.plot.setClipToView(True)
        self.curve_live.setClipToView(True)
        self.curve_avg.setClipToView(True)
        self.curve_peak.setClipToView(True)

        self.vline = pg.InfiniteLine(angle=90, movable=False)
        self.plot.addItem(self.vline, ignoreBounds=True)

        self.marker1_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("y"))
        self.marker2_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("m"))
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
        self.trace_live_rb.toggled.connect(self.on_trace_mode_changed)
        self.trace_avg_rb.toggled.connect(self.on_trace_mode_changed)
        self.trace_peak_rb.toggled.connect(self.on_trace_mode_changed)
        self.peak_hold_toggle.toggled.connect(self.on_peak_hold_toggled)
        self.clear_peak_btn.clicked.connect(self.on_clear_peak_hold)

        self.avg_edit.editingFinished.connect(self.on_avg_changed)
        self.avg_mode_cb.currentTextChanged.connect(self.on_avg_changed)

        self.amp_mode_cb.currentTextChanged.connect(self.on_amp_mode_changed)
        self.auto_ref_cb.toggled.connect(self.on_auto_ref_toggled)
        self.scale_edit.editingFinished.connect(self.on_scale_changed)
        self.ref_edit.editingFinished.connect(self.on_ref_changed)

        self.dc_remove_cb.toggled.connect(self.on_dc_remove)
        self.dc_blank_edit.editingFinished.connect(self.on_dc_blank_changed)
        self.update_ms_edit.editingFinished.connect(self.on_update_ms_changed)

        self.apply_cal_btn.clicked.connect(self.on_apply_calibration)
        self.calibrate_btn.clicked.connect(self.on_calibrate_peak)

        self.marker_peak_btn.clicked.connect(self.on_marker_peak)
        self.marker_left_btn.clicked.connect(self.on_marker_left)
        self.marker_right_btn.clicked.connect(self.on_marker_right)
        self.clear_markers_btn.clicked.connect(self.on_marker_clear)

    def _apply_initial_state(self):
        # Ensure UI matches current SDR state.
        self._sync_center_edit()
        self.on_gainmode_changed(self.gainmode_cb.currentText())
        self.on_measurement_mode(self.measurement_cb.isChecked())
        self.on_rbw_mode(self.rbw_mode_cb.currentText())
        self.on_vbw_mode(self.vbw_mode_cb.currentText())
        self.on_detector_changed(self.detector_cb.currentText())
        self.on_update_ms_changed()
        self.auto_ref_cb.setChecked(self.auto_scale_enabled)
        self._set_trace_mode(self.cfg.trace_type)
        if self.cfg.trace_type != "Average":
            self.peak_hold_toggle.setChecked(self.peak_hold_enabled)
        self.ref_edit.setText(f"{self.cfg.ref_level_db:.1f}")
        self.scale_edit.setText(f"{self.cfg.display_range_db / 10.0:.1f}")
        self._update_range_label()
        self.snap_x_to_span()

    def _update_amp_modes(self):
        current = self.amp_mode_cb.currentText() if hasattr(self, "amp_mode_cb") else None
        self.amp_mode_cb.clear()
        self.amp_mode_cb.addItem("dBFS")
        if self.cal.has_calibration():
            self.amp_mode_cb.addItem("dBm")
            self.amp_mode_cb.addItem("dBm/Hz")
        if current and current in [self.amp_mode_cb.itemText(i) for i in range(self.amp_mode_cb.count())]:
            self.amp_mode_cb.setCurrentText(current)

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
            "ref_level_db": self.cfg.ref_level_db,
            "display_range_db": self.cfg.display_range_db,
            "rbw_mode": self.cfg.rbw_mode,
            "rbw_hz": self.cfg.rbw_hz,
            "vbw_mode": self.cfg.vbw_mode,
            "vbw_hz": self.cfg.vbw_hz,
            "window": self.cfg.window,
            "detector": self.cfg.detector,
            "trace_mode": self.cfg.trace_type,
            "update_ms": self.cfg.update_ms,
            "auto_scale": self.auto_ref_cb.isChecked(),
            "peak_hold_enabled": self.peak_hold_toggle.isChecked(),
        }
        with open(STATE_PATH, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def _set_cfg(self, **kwargs) -> None:
        # Update shared config with worker lock to avoid race conditions.
        with QtCore.QMutexLocker(self.worker._lock):
            for key, value in kwargs.items():
                setattr(self.cfg, key, value)

    def _parse_freq_to_hz(self, value_str: str, unit: str) -> int:
        value = float(value_str.strip())
        scale = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}[unit]
        return int(value * scale)

    def _sync_center_edit(self):
        unit = self.freq_unit.currentText()
        inv = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}[unit]
        self.freq_edit.setText(f"{self.cfg.center_hz / inv:.6g}")

    def snap_x_to_span(self):
        # Keep the visible span centered on the LO.
        lo = float(self.sdr.lo)
        sr = float(self.sdr.sample_rate)
        self.plot.setXRange(lo - sr / 2.0, lo + sr / 2.0, padding=0.0)

    def on_set_center(self):
        try:
            hz = self._parse_freq_to_hz(self.freq_edit.text(), self.freq_unit.currentText())
            with QtCore.QMutexLocker(self.worker._lock):
                self.sdr.set_center_hz(hz)
            self._sync_center_edit()
            self.snap_x_to_span()
            self.status.setText(f"Center set to {self.cfg.center_hz} Hz")
        except Exception as exc:
            self.status.setText(f"Bad center freq: {exc}")

    def on_apply_span(self):
        try:
            text = self.span_cb.currentText().strip()
            mhz = float(text.split()[0])
            span_hz = int(mhz * 1e6)

            with QtCore.QMutexLocker(self.worker._lock):
                self.sdr.set_span_hz(span_hz)
            self._apply_rbw_strategy()
            self.snap_x_to_span()
            self.max_trace = None
            self.status.setText(f"Span set to {mhz:g} MHz")
        except Exception as exc:
            self.status.setText(f"Span apply failed: {exc}")

    def on_apply_rfbw(self):
        try:
            hz = int(float(self.rfbw_edit.text().strip()))
            with QtCore.QMutexLocker(self.worker._lock):
                self.sdr.set_rf_bw(hz)
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

    def on_vbw_mode(self, mode: str):
        self._set_cfg(vbw_mode=mode)
        self.vbw_edit.setEnabled(mode == "Manual")

    def on_vbw_changed(self):
        try:
            vbw = float(self.vbw_edit.text().strip())
        except ValueError:
            vbw = 0.0
        self._set_cfg(vbw_hz=vbw)

    def on_detector_changed(self, mode: str):
        self._set_cfg(detector=mode)

    def _set_trace_mode(self, mode: str) -> None:
        if mode == "Average":
            self.trace_avg_rb.setChecked(True)
        elif mode == "Peak Hold":
            self.trace_peak_rb.setChecked(True)
        else:
            self.trace_live_rb.setChecked(True)

    def on_trace_mode_changed(self):
        if self.trace_avg_rb.isChecked():
            mode = "Average"
        elif self.trace_peak_rb.isChecked():
            mode = "Peak Hold"
        else:
            mode = "Live"
        self._set_cfg(trace_type=mode)
        # Clear incompatible state when switching modes.
        if mode != "Average":
            self.avg_trace = None
        if mode != "Peak Hold":
            self.max_trace = None
        if mode == "Average":
            self.peak_hold_toggle.setChecked(False)
            self.peak_hold_toggle.setEnabled(False)
        else:
            self.peak_hold_toggle.setEnabled(True)
            if mode == "Peak Hold":
                self.peak_hold_toggle.setChecked(True)

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

    def on_scale_changed(self):
        try:
            scale = float(self.scale_edit.text().strip())
        except ValueError:
            scale = self.cfg.display_range_db / 10.0
        if scale <= 0:
            scale = 10.0
        self.cfg.display_range_db = scale * 10.0
        self._update_range_label()

    def on_ref_changed(self):
        try:
            self.cfg.ref_level_db = float(self.ref_edit.text().strip())
        except ValueError:
            pass

    def _update_range_label(self):
        self.range_label.setText(f"Range: {self.cfg.display_range_db:.0f} dB")

    def on_peak_hold_toggled(self, checked: bool):
        self.peak_hold_enabled = checked
        if not checked:
            self.curve_peak.hide()
        else:
            self.curve_peak.show()

    def on_clear_peak_hold(self):
        self.max_trace = None

    def on_update_ms_changed(self):
        try:
            update_ms = int(float(self.update_ms_edit.text().strip()))
        except ValueError:
            update_ms = self.cfg.update_ms
        update_ms = max(50, min(2000, update_ms))
        self.update_ms_edit.setText(str(update_ms))
        self._set_cfg(update_ms=update_ms)

    def on_dc_remove(self, checked: bool):
        self._set_cfg(dc_remove=checked)

    def on_dc_blank_changed(self):
        try:
            bins = max(0, int(float(self.dc_blank_edit.text().strip())))
        except ValueError:
            bins = 1
        self._set_cfg(dc_blank_bins=bins)

    def on_apply_calibration(self):
        try:
            offset_text = self.cal_offset_edit.text().strip()
            self.cal.dbfs_to_dbm_offset = float(offset_text) if offset_text else None
            self.cal.external_gain_db = float(self.cal_gain_edit.text().strip())
            self.cal.save()
            # Update amplitude mode list to enable/disable dBm options.
            self._update_amp_modes()
            self.status.setText("Calibration updated")
        except Exception as exc:
            self.status.setText(f"Calibration error: {exc}")

    def on_calibrate_peak(self):
        if self.latest_display is None or self.latest_power is None:
            self.status.setText("No data to calibrate")
            return
        try:
            tone_dbm = float(self.tone_dbm_edit.text().strip())
            peak_dbfs = float(np.max(self.power_to_dbfs(self.latest_power)))
            # Offset aligns measured peak to the known tone power.
            self.cal.dbfs_to_dbm_offset = tone_dbm - peak_dbfs
            self.cal.external_gain_db = float(self.cal_gain_edit.text().strip())
            self.cal.save()
            self.cal_offset_edit.setText(f"{self.cal.dbfs_to_dbm_offset:.2f}")
            self._update_amp_modes()
            self.status.setText("Calibration updated from peak")
        except Exception as exc:
            self.status.setText(f"Calibration error: {exc}")

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
        self.marker_info.setText("M1: --\nM2: --\nΔ : --")

    def _move_marker_peak(self, direction: int):
        if self.latest_display is None:
            return
        if self.marker1 is None:
            self.on_marker_peak()
            return
        data = self.latest_display
        idx = self.marker1
        peaks = self._find_peaks(data)
        if not peaks:
            return
        peaks = sorted(peaks)
        if direction < 0:
            left_peaks = [p for p in peaks if p < idx]
            if left_peaks:
                self.marker1 = left_peaks[-1]
        else:
            right_peaks = [p for p in peaks if p > idx]
            if right_peaks:
                self.marker1 = right_peaks[0]
        self._update_markers()

    def _find_peaks(self, data: np.ndarray) -> list:
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i - 1] and data[i] > data[i + 1]:
                peaks.append(i)
        return peaks

    def _apply_rbw_strategy(self):
        with QtCore.QMutexLocker(self.worker._lock):
            fs = self.sdr.sample_rate
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
        with QtCore.QMutexLocker(self.worker._lock):
            # Apply FFT size and update RBW using the actual window.
            self.sdr.set_fft_size(fft_size, buffer_factor)
            self.proc.update_fft_size(fft_size, window)
            self.cfg.fft_size = fft_size
            self.cfg.rbw_hz = self.proc.rbw_hz(fs)
        self.rbw_edit.setText(f"{self.cfg.rbw_hz:.0f}")
        self.status.setText(f"RBW updated: {self.cfg.rbw_hz:.0f} Hz")
        if self.max_trace is not None:
            self.max_trace = None
            self.status.setText("RBW changed: Peak Hold cleared")

    def _update_ref_level(self, display: np.ndarray) -> None:
        if not self.auto_scale_enabled:
            return
        now = time.monotonic()
        peak = float(np.max(display))
        target = peak + 5.0
        ref_level = self.cfg.ref_level_db

        if abs(target - ref_level) < self.auto_ref_hysteresis_db:
            return

        dt = max(1e-3, now - self.last_auto_ref_update) if self.last_auto_ref_update else 0.1
        self.last_auto_ref_update = now
        alpha = 1.0 - math.exp(-dt / self.auto_ref_tau_s)
        ref_level = ref_level + (target - ref_level) * alpha
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
        info = {"M1": "--", "M2": "--", "Δ": "--"}

        if self.marker1 is not None:
            f1 = self.latest_freqs[self.marker1]
            a1 = self.latest_display[self.marker1]
            self.marker1_line.setPos(f1)
            self.marker1_line.show()
            info["M1"] = f"{self._format_freq(f1):<14} {self._format_amp(a1)}"
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
        if self.vbw_state is None:
            self.vbw_state = power.copy()
        else:
            self.vbw_state = alpha * self.vbw_state + (1.0 - alpha) * power
        return self.vbw_state

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
        payload = self.latest_payload
        power = payload["power"]
        rbw = payload["rbw"]
        timestamp = payload["timestamp"]
        freqs = payload["freqs"]

        # Apply VBW smoothing in linear power.
        power = self._apply_vbw(power, rbw, timestamp)
        live_display = self._power_to_display(power, rbw_hz=rbw)

        if self.peak_hold_enabled:
            if self.max_trace is None or len(self.max_trace) != len(power):
                self.max_trace = power.copy()
            else:
                self.max_trace = np.maximum(self.max_trace, power)
            peak_display = self._power_to_display(self.max_trace, rbw_hz=rbw)
        else:
            peak_display = None

        if self.cfg.trace_type == "Average":
            avg_power = self._apply_average(power)
            avg_display = self._power_to_display(avg_power, rbw_hz=rbw)
            self.curve_avg.setData(freqs, avg_display)
            self.curve_avg.show()
            self.curve_live.hide()
            self.curve_peak.hide()
            display = avg_display
        elif self.cfg.trace_type == "Peak Hold":
            self.curve_live.setData(freqs, live_display)
            self.curve_live.show()
            if peak_display is not None:
                self.curve_peak.setData(freqs, peak_display)
                self.curve_peak.show()
                display = peak_display
            else:
                self.curve_peak.hide()
                display = live_display
            self.curve_avg.hide()
        else:
            self.curve_live.setData(freqs, live_display)
            self.curve_live.show()
            if peak_display is not None:
                self.curve_peak.setData(freqs, peak_display)
                self.curve_peak.show()
            else:
                self.curve_peak.hide()
            self.curve_avg.hide()
            display = live_display

        self.latest_power = power
        self.latest_display = display
        self.latest_freqs = freqs
        self.hover.update_axis(freqs, display)
        self._update_markers()

        self._update_ref_level(display)
        self._update_axis_ticks(freqs)

        ref_level = self.cfg.ref_level_db
        display_range = self.cfg.display_range_db
        self.plot.setYRange(ref_level - display_range, ref_level, padding=0.0)

        # Update status readout with current instrument state.
        self.cfg.rbw_hz = rbw
        self.status.setText(
            " ".join(
                [
                    f"LO {int(payload['lo'])}",
                    f"SR {int(payload['fs'])}",
                    f"RBW {rbw:.1f} Hz",
                    f"ENBW {payload['enbw_bins']:.2f}",
                    f"FFT {self.cfg.fft_size}",
                    f"Gain {payload['gain_db']} dB",
                    f"RF BW {int(payload['rf_bw'])}",
                ]
            )
        )

def main():
    cfg = SpectrumConfig(uri="ip:192.168.2.1")
    app = pg.mkQApp("Pluto Spectrum Viewer")
    win = SpectrumWindow(cfg)
    win.resize(1400, 820)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
