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
    trace_type: str = "Clear"
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

        self.setWindowTitle("Pluto Spectrum Viewer")

        pg.setConfigOptions(antialias=False)

        self.sdr = PlutoSdr(cfg)
        self.proc = SpectrumProcessor(cfg.fft_size, cfg.window)
        self.hover = HoverReadout()

        self._build_ui()
        self._wire_events()

        self.worker = SpectrumWorker(self.sdr, self.proc, cfg)
        self.worker.new_data.connect(self.on_new_data)
        self.worker.start()

        self._apply_initial_state()

        self.last_update_ts: Optional[float] = None
        self.vbw_state: Optional[np.ndarray] = None
        self.avg_trace: Optional[np.ndarray] = None
        self.max_trace: Optional[np.ndarray] = None
        self.min_trace: Optional[np.ndarray] = None
        self.latest_power: Optional[np.ndarray] = None
        self.latest_display: Optional[np.ndarray] = None
        self.latest_freqs: Optional[np.ndarray] = None
        self.marker1: Optional[int] = None
        self.marker2: Optional[int] = None

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait(1000)
        event.accept()

    def _build_ui(self):
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)

        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(6)
        layout.addWidget(control_panel)

        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(10)
        control_layout.addLayout(row1)

        row1.addWidget(QtWidgets.QLabel("Center"))
        self.freq_edit = QtWidgets.QLineEdit("2437")
        self.freq_edit.setFixedWidth(90)
        row1.addWidget(self.freq_edit)

        self.freq_unit = QtWidgets.QComboBox()
        self.freq_unit.addItems(["Hz", "kHz", "MHz", "GHz"])
        self.freq_unit.setCurrentText("MHz")
        self.freq_unit.setFixedWidth(80)
        row1.addWidget(self.freq_unit)

        self.set_btn = QtWidgets.QPushButton("Set")
        row1.addWidget(self.set_btn)

        row1.addSpacing(16)
        row1.addWidget(QtWidgets.QLabel("Span"))
        self.span_cb = QtWidgets.QComboBox()
        self.span_cb.addItems(["1 MHz", "5 MHz", "10 MHz", "20 MHz", "30 MHz", "40 MHz", "50 MHz"])
        self.span_cb.setCurrentText("20 MHz")
        self.span_cb.setFixedWidth(90)
        row1.addWidget(self.span_cb)

        self.apply_span_btn = QtWidgets.QPushButton("Apply")
        row1.addWidget(self.apply_span_btn)

        row1.addWidget(QtWidgets.QLabel("RF BW"))
        self.rfbw_edit = QtWidgets.QLineEdit(str(self.cfg.rf_bw_hz))
        self.rfbw_edit.setFixedWidth(90)
        row1.addWidget(self.rfbw_edit)

        self.apply_rfbw_btn = QtWidgets.QPushButton("Set")
        row1.addWidget(self.apply_rfbw_btn)

        row1.addSpacing(16)
        row1.addWidget(QtWidgets.QLabel("Gain mode"))
        self.gainmode_cb = QtWidgets.QComboBox()
        self.gainmode_cb.addItems(["manual", "fast_attack", "slow_attack", "hybrid"])
        self.gainmode_cb.setCurrentText(self.cfg.gain_mode)
        row1.addWidget(self.gainmode_cb)

        row1.addWidget(QtWidgets.QLabel("Gain dB"))
        self.gain_edit = QtWidgets.QLineEdit(str(self.cfg.gain_db))
        self.gain_edit.setFixedWidth(60)
        row1.addWidget(self.gain_edit)

        self.apply_gain_btn = QtWidgets.QPushButton("Apply gain")
        row1.addWidget(self.apply_gain_btn)

        self.measurement_cb = QtWidgets.QCheckBox("Measurement mode")
        self.measurement_cb.setChecked(self.cfg.measurement_mode)
        row1.addWidget(self.measurement_cb)

        row1.addStretch(1)

        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(10)
        control_layout.addLayout(row2)

        row2.addWidget(QtWidgets.QLabel("RBW"))
        self.rbw_mode_cb = QtWidgets.QComboBox()
        self.rbw_mode_cb.addItems(["Auto", "Manual"])
        self.rbw_mode_cb.setCurrentText(self.cfg.rbw_mode)
        row2.addWidget(self.rbw_mode_cb)

        self.rbw_edit = QtWidgets.QLineEdit(f"{self.cfg.rbw_hz:.0f}")
        self.rbw_edit.setFixedWidth(90)
        row2.addWidget(self.rbw_edit)

        self.rbw_apply_btn = QtWidgets.QPushButton("Apply RBW")
        row2.addWidget(self.rbw_apply_btn)

        row2.addWidget(QtWidgets.QLabel("Window"))
        self.window_cb = QtWidgets.QComboBox()
        self.window_cb.addItems(["Hann", "Blackman Harris", "Flat top"])
        self.window_cb.setCurrentText(self.cfg.window)
        row2.addWidget(self.window_cb)

        row2.addWidget(QtWidgets.QLabel("VBW"))
        self.vbw_mode_cb = QtWidgets.QComboBox()
        self.vbw_mode_cb.addItems(["Off", "Auto", "Manual"])
        self.vbw_mode_cb.setCurrentText(self.cfg.vbw_mode)
        row2.addWidget(self.vbw_mode_cb)

        self.vbw_edit = QtWidgets.QLineEdit(f"{self.cfg.vbw_hz:.0f}")
        self.vbw_edit.setFixedWidth(80)
        row2.addWidget(self.vbw_edit)

        row2.addWidget(QtWidgets.QLabel("Detector"))
        self.detector_cb = QtWidgets.QComboBox()
        self.detector_cb.addItems(["Sample", "Peak", "RMS"])
        self.detector_cb.setCurrentText(self.cfg.detector)
        row2.addWidget(self.detector_cb)

        row2.addWidget(QtWidgets.QLabel("Trace"))
        self.trace_cb = QtWidgets.QComboBox()
        self.trace_cb.addItems(["Clear", "Max Hold", "Min Hold", "Average"])
        self.trace_cb.setCurrentText(self.cfg.trace_type)
        row2.addWidget(self.trace_cb)

        self.clear_trace_btn = QtWidgets.QPushButton("Clear trace")
        row2.addWidget(self.clear_trace_btn)

        row2.addStretch(1)

        row3 = QtWidgets.QHBoxLayout()
        row3.setSpacing(10)
        control_layout.addLayout(row3)

        row3.addWidget(QtWidgets.QLabel("Avg count"))
        self.avg_edit = QtWidgets.QLineEdit(str(self.cfg.avg_count))
        self.avg_edit.setFixedWidth(60)
        row3.addWidget(self.avg_edit)

        self.avg_mode_cb = QtWidgets.QComboBox()
        self.avg_mode_cb.addItems(["RMS", "Log"])
        self.avg_mode_cb.setCurrentText(self.cfg.avg_mode)
        row3.addWidget(self.avg_mode_cb)

        row3.addWidget(QtWidgets.QLabel("Amplitude"))
        self.amp_mode_cb = QtWidgets.QComboBox()
        self._update_amp_modes()
        row3.addWidget(self.amp_mode_cb)

        row3.addWidget(QtWidgets.QLabel("Ref level"))
        self.ref_edit = QtWidgets.QLineEdit(f"{self.cfg.ref_level_db:.1f}")
        self.ref_edit.setFixedWidth(70)
        row3.addWidget(self.ref_edit)

        row3.addWidget(QtWidgets.QLabel("Range"))
        self.range_edit = QtWidgets.QLineEdit(f"{self.cfg.display_range_db:.1f}")
        self.range_edit.setFixedWidth(70)
        row3.addWidget(self.range_edit)

        self.auto_ref_cb = QtWidgets.QCheckBox("Auto ref")
        self.auto_ref_cb.setChecked(True)
        row3.addWidget(self.auto_ref_cb)

        self.dc_remove_cb = QtWidgets.QCheckBox("DC remove")
        self.dc_remove_cb.setChecked(self.cfg.dc_remove)
        row3.addWidget(self.dc_remove_cb)

        row3.addWidget(QtWidgets.QLabel("DC blank bins"))
        self.dc_blank_edit = QtWidgets.QLineEdit(str(self.cfg.dc_blank_bins))
        self.dc_blank_edit.setFixedWidth(50)
        row3.addWidget(self.dc_blank_edit)

        row3.addStretch(1)

        row4 = QtWidgets.QHBoxLayout()
        row4.setSpacing(10)
        control_layout.addLayout(row4)

        row4.addWidget(QtWidgets.QLabel("Calibration"))
        self.cal_offset_edit = QtWidgets.QLineEdit(
            "" if self.cal.dbfs_to_dbm_offset is None else f"{self.cal.dbfs_to_dbm_offset:.2f}"
        )
        self.cal_offset_edit.setFixedWidth(80)
        row4.addWidget(self.cal_offset_edit)

        row4.addWidget(QtWidgets.QLabel("Ext gain"))
        self.cal_gain_edit = QtWidgets.QLineEdit(f"{self.cal.external_gain_db:.2f}")
        self.cal_gain_edit.setFixedWidth(60)
        row4.addWidget(self.cal_gain_edit)

        self.apply_cal_btn = QtWidgets.QPushButton("Apply cal")
        row4.addWidget(self.apply_cal_btn)

        row4.addWidget(QtWidgets.QLabel("Tone dBm"))
        self.tone_dbm_edit = QtWidgets.QLineEdit("-30")
        self.tone_dbm_edit.setFixedWidth(60)
        row4.addWidget(self.tone_dbm_edit)

        self.calibrate_btn = QtWidgets.QPushButton("Calibrate from peak")
        row4.addWidget(self.calibrate_btn)

        row4.addStretch(1)

        row5 = QtWidgets.QHBoxLayout()
        row5.setSpacing(10)
        control_layout.addLayout(row5)

        self.marker_info = QtWidgets.QLabel("Markers: --")
        row5.addWidget(self.marker_info)

        self.marker_peak_btn = QtWidgets.QPushButton("Peak")
        row5.addWidget(self.marker_peak_btn)

        self.marker_left_btn = QtWidgets.QPushButton("Next left")
        row5.addWidget(self.marker_left_btn)

        self.marker_right_btn = QtWidgets.QPushButton("Next right")
        row5.addWidget(self.marker_right_btn)

        self.clear_markers_btn = QtWidgets.QPushButton("Clear markers")
        row5.addWidget(self.clear_markers_btn)

        row5.addStretch(1)

        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("QLabel { padding: 4px; }")
        control_layout.addWidget(self.status)

        self.plotw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plotw, 1)

        self.plot = self.plotw.addPlot()
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "Magnitude", units="dBFS")
        self.plot.showGrid(x=True, y=True)
        self.plot.setMouseEnabled(x=True, y=False)

        self.curve = self.plot.plot()
        self.plot.setClipToView(True)
        self.curve.setClipToView(True)
        self.curve.setDownsampling(auto=True, method="peak")

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
        self.trace_cb.currentTextChanged.connect(self.on_trace_changed)
        self.clear_trace_btn.clicked.connect(self.on_clear_trace)

        self.avg_edit.editingFinished.connect(self.on_avg_changed)
        self.avg_mode_cb.currentTextChanged.connect(self.on_avg_changed)

        self.amp_mode_cb.currentTextChanged.connect(self.on_amp_mode_changed)
        self.auto_ref_cb.toggled.connect(self.on_auto_ref_toggled)

        self.dc_remove_cb.toggled.connect(self.on_dc_remove)
        self.dc_blank_edit.editingFinished.connect(self.on_dc_blank_changed)

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

    def on_trace_changed(self, mode: str):
        self._set_cfg(trace_type=mode)
        # Clear any stored trace state on mode switch.
        self.on_clear_trace()

    def on_clear_trace(self):
        self.max_trace = None
        self.min_trace = None
        self.avg_trace = None

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
        self.auto_ref_cb.setChecked(checked)

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
        self.marker_info.setText("Markers: --")

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

    def _update_markers(self):
        if self.latest_freqs is None or self.latest_display is None:
            return
        amp_unit = self.amp_mode_cb.currentText()
        info = []

        if self.marker1 is not None:
            f1 = self.latest_freqs[self.marker1]
            a1 = self.latest_display[self.marker1]
            self.marker1_line.setPos(f1)
            self.marker1_line.show()
            info.append(f"M1 {f1/1e6:.6f} MHz {a1:.2f} {amp_unit}")
        else:
            self.marker1_line.hide()

        if self.marker2 is not None:
            f2 = self.latest_freqs[self.marker2]
            a2 = self.latest_display[self.marker2]
            self.marker2_line.setPos(f2)
            self.marker2_line.show()
            info.append(f"M2 {f2/1e6:.6f} MHz {a2:.2f} {amp_unit}")
        else:
            self.marker2_line.hide()

        if self.marker1 is not None and self.marker2 is not None:
            # Delta marker readout and channel power between markers.
            df = abs(self.latest_freqs[self.marker2] - self.latest_freqs[self.marker1])
            da = self.latest_display[self.marker2] - self.latest_display[self.marker1]
            info.append(f"Δ {df/1e6:.6f} MHz {da:.2f} {amp_unit}")

            band_power = self._channel_power(self.marker1, self.marker2)
            if band_power is not None:
                info.append(f"Ch Pwr {band_power:.2f} {amp_unit}")

        if info:
            self.marker_info.setText(" | ".join(info))
        else:
            self.marker_info.setText("Markers: --")

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

    def _apply_trace_type(self, power: np.ndarray) -> np.ndarray:
        trace = self.trace_cb.currentText()
        if trace == "Clear":
            return power
        if trace == "Max Hold":
            # Peak hold across updates (linear power).
            if self.max_trace is None:
                self.max_trace = power.copy()
            else:
                self.max_trace = np.maximum(self.max_trace, power)
            return self.max_trace
        if trace == "Min Hold":
            # Min hold across updates (linear power).
            if self.min_trace is None:
                self.min_trace = power.copy()
            else:
                self.min_trace = np.minimum(self.min_trace, power)
            return self.min_trace
        if trace == "Average":
            # Running average (RMS or log-average).
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
        return power

    def on_new_data(self, payload: Dict):
        power = payload["power"]
        rbw = payload["rbw"]
        timestamp = payload["timestamp"]

        # Apply VBW and trace type before amplitude conversion.
        power = self._apply_vbw(power, rbw, timestamp)
        power = self._apply_trace_type(power)

        display = self._power_to_display(power, rbw_hz=rbw)

        self.latest_power = power
        self.latest_display = display
        self.latest_freqs = payload["freqs"]

        self.curve.setData(payload["freqs"], display)
        self.hover.update_axis(payload["freqs"], display)
        self._update_markers()

        if self.auto_ref_cb.isChecked():
            # Auto reference level tracks the latest peak.
            ref_level = float(np.max(display)) + 5.0
            self.ref_edit.setText(f"{ref_level:.1f}")
        else:
            try:
                ref_level = float(self.ref_edit.text().strip())
            except ValueError:
                ref_level = self.cfg.ref_level_db

        try:
            display_range = float(self.range_edit.text().strip())
        except ValueError:
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
