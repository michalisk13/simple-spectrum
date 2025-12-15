import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph import SignalProxy
import adi


@dataclass
class SpectrumConfig:
    """
    Configuration for the spectrum viewer.

    Notes
    Span is the sampled bandwidth.
    For a real time FFT spectrum, visible span equals sample rate.
    """
    uri: str = "ip:192.168.2.1"

    center_hz: int = 2_437_000_000

    sample_rate_hz: int = 20_000_000
    rf_bw_hz: int = 20_000_000

    gain_db: int = 55
    gain_mode: str = "manual"

    fft_size: int = 131072
    update_ms: int = 220

    hover_rate_hz: int = 20


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
        self.dev.sample_rate = int(self.cfg.sample_rate_hz)
        self.dev.rx_rf_bandwidth = int(self.cfg.rf_bw_hz)

    def set_center_hz(self, hz: int):
        hz = int(hz)

        if hz < 325_000_000:
            hz = 325_000_000
        if hz > 3_800_000_000:
            hz = 3_800_000_000

        self.dev.rx_lo = hz
        self.cfg.center_hz = hz

    def set_span_hz(self, span_hz: int):
        span_hz = int(span_hz)

        self.cfg.sample_rate_hz = span_hz
        self.cfg.rf_bw_hz = span_hz

        self.dev.sample_rate = span_hz
        self.dev.rx_rf_bandwidth = span_hz

        self.dev.rx_destroy_buffer()

    def set_fft_size(self, n: int):
        n = int(n)
        self.cfg.fft_size = n
        self.dev.rx_buffer_size = n
        self.dev.rx_destroy_buffer()

    def set_gain_mode(self, mode: str):
        self.cfg.gain_mode = mode
        self.dev.gain_control_mode_chan0 = mode

    def set_gain_db(self, gain_db: int):
        gain_db = int(gain_db)
        if gain_db < 0:
            gain_db = 0
        if gain_db > 70:
            gain_db = 70

        self.cfg.gain_db = gain_db
        self.dev.rx_hardwaregain_chan0 = gain_db

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
    Currently uses a PSD style estimate (power per Hz).
    """

    def __init__(self, fft_size: int):
        self.fft_size = int(fft_size)
        self.window = self._make_window(self.fft_size)

    @staticmethod
    def _make_window(n: int) -> np.ndarray:
        return np.hanning(n).astype(np.float32)

    def update_fft_size(self, n: int):
        self.fft_size = int(n)
        self.window = self._make_window(self.fft_size)

    def compute_psd_db(self, x: np.ndarray, fs_hz: float) -> np.ndarray:
        """
        Returns magnitude in dB for a PSD estimate.

        Steps
        Window the samples to reduce leakage.
        FFT then shift.
        PSD normalization uses window power and sample rate.
        Convert to dB with clamp to avoid log(0).
        """
        xw = x * self.window
        X = np.fft.fftshift(np.fft.fft(xw))

        win_power = float(np.sum(self.window ** 2))
        psd = (np.abs(X) ** 2) / (win_power * float(fs_hz))

        mag_db = 10.0 * np.log10(np.maximum(psd, 1e-20))
        return mag_db

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


class SpectrumWindow(QtWidgets.QMainWindow):
    """
    Main UI class.
    All SDR actions go through PlutoSdr.
    All DSP goes through SpectrumProcessor.
    """

    def __init__(self, cfg: SpectrumConfig):
        super().__init__()
        self.cfg = cfg

        self.setWindowTitle("Pluto Spectrum Viewer")

        pg.setConfigOptions(antialias=False)

        self.sdr = PlutoSdr(cfg)
        self.proc = SpectrumProcessor(cfg.fft_size)
        self.hover = HoverReadout()

        self._build_ui()
        self._wire_events()
        self._apply_initial_state()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(cfg.update_ms)

    def _build_ui(self):
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)

        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(4)
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

        row1.addSpacing(20)
        row1.addWidget(QtWidgets.QLabel("Span"))
        self.span_cb = QtWidgets.QComboBox()
        self.span_cb.addItems(["1 MHz", "5 MHz", "10 MHz", "20 MHz", "30 MHz", "40 MHz", "50 MHz"])
        self.span_cb.setCurrentText("20 MHz")
        self.span_cb.setFixedWidth(90)
        row1.addWidget(self.span_cb)

        self.apply_span_btn = QtWidgets.QPushButton("Apply")
        row1.addWidget(self.apply_span_btn)

        self.autospan_cb = QtWidgets.QCheckBox("Auto span")
        self.autospan_cb.setChecked(True)
        row1.addWidget(self.autospan_cb)

        row1.addSpacing(20)
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

        self.autoy_cb = QtWidgets.QCheckBox("Auto Y")
        self.autoy_cb.setChecked(True)
        row1.addWidget(self.autoy_cb)

        row1.addWidget(QtWidgets.QLabel("Y min"))
        self.ymin_edit = QtWidgets.QLineEdit("-120")
        self.ymin_edit.setFixedWidth(70)
        row1.addWidget(self.ymin_edit)

        row1.addWidget(QtWidgets.QLabel("Y max"))
        self.ymax_edit = QtWidgets.QLineEdit("0")
        self.ymax_edit.setFixedWidth(70)
        row1.addWidget(self.ymax_edit)

        self.apply_y_btn = QtWidgets.QPushButton("Apply Y")
        row1.addWidget(self.apply_y_btn)

        row1.addStretch(1)

        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("QLabel { padding: 4px; }")
        control_layout.addWidget(self.status)

        self.plotw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plotw, 1)

        self.plot = self.plotw.addPlot()
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "Magnitude", units="dB")
        self.plot.showGrid(x=True, y=True)
        self.plot.setMouseEnabled(x=True, y=False)

        self.curve = self.plot.plot()
        self.plot.setClipToView(True)
        self.curve.setClipToView(True)
        self.curve.setDownsampling(auto=True, method="peak")

        self.vline = pg.InfiniteLine(angle=90, movable=False)
        self.plot.addItem(self.vline, ignoreBounds=True)

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

    def _wire_events(self):
        self.set_btn.clicked.connect(self.on_set_center)
        self.apply_span_btn.clicked.connect(self.on_apply_span)

        self.gainmode_cb.currentTextChanged.connect(self.on_gainmode_changed)
        self.apply_gain_btn.clicked.connect(self.on_apply_gain)

        self.autoy_cb.toggled.connect(self.on_autoy_toggled)
        self.apply_y_btn.clicked.connect(self.apply_manual_y_once)

    def _apply_initial_state(self):
        self._sync_center_edit()
        self.on_gainmode_changed(self.gainmode_cb.currentText())
        self.on_autoy_toggled(self.autoy_cb.isChecked())
        self.snap_x_to_span()

    def _parse_freq_to_hz(self, value_str: str, unit: str) -> int:
        value = float(value_str.strip())
        scale = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}[unit]
        return int(value * scale)

    def _sync_center_edit(self):
        unit = self.freq_unit.currentText()
        inv = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}[unit]
        self.freq_edit.setText(f"{self.cfg.center_hz / inv:.6g}")

    def snap_x_to_span(self):
        lo = float(self.sdr.lo)
        sr = float(self.sdr.sample_rate)
        self.plot.setXRange(lo - sr / 2.0, lo + sr / 2.0, padding=0.0)

    def on_set_center(self):
        try:
            hz = self._parse_freq_to_hz(self.freq_edit.text(), self.freq_unit.currentText())
            self.sdr.set_center_hz(hz)
            self._sync_center_edit()
            self.snap_x_to_span()
            self.status.setText(f"Center set to {self.cfg.center_hz} Hz")
        except Exception as e:
            self.status.setText(f"Bad center freq: {e}")

    def on_apply_span(self):
        try:
            text = self.span_cb.currentText().strip()
            mhz = float(text.split()[0])
            span_hz = int(mhz * 1e6)

            self.sdr.set_span_hz(span_hz)

            self.sdr.set_fft_size(self.cfg.fft_size)
            self.proc.update_fft_size(self.cfg.fft_size)

            self.snap_x_to_span()
            self.status.setText(f"Span set to {mhz:g} MHz")
        except Exception as e:
            self.status.setText(f"Span apply failed: {e}")

    def on_gainmode_changed(self, mode: str):
        try:
            self.sdr.set_gain_mode(mode)

            is_manual = mode == "manual"
            self.gain_edit.setEnabled(is_manual)
            self.apply_gain_btn.setEnabled(is_manual)

            self.status.setText(f"Gain mode set to {mode}")
        except Exception as e:
            self.status.setText(f"Gain mode error: {e}")

    def on_apply_gain(self):
        try:
            g = int(float(self.gain_edit.text().strip()))
            self.sdr.set_gain_db(g)
            self.status.setText(f"Manual gain set to {self.cfg.gain_db} dB")
        except Exception as e:
            self.status.setText(f"Gain error: {e}")

    def on_autoy_toggled(self, checked: bool):
        self.ymin_edit.setEnabled(not checked)
        self.ymax_edit.setEnabled(not checked)
        self.apply_y_btn.setEnabled(not checked)

    def apply_manual_y_once(self):
        try:
            y0 = float(self.ymin_edit.text().strip())
            y1 = float(self.ymax_edit.text().strip())
            if y1 <= y0:
                raise ValueError("Y max must be greater than Y min")
            self.plot.setYRange(y0, y1, padding=0.0)
        except Exception as e:
            self.status.setText(f"Bad Y range: {e}")

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
            text = f"{f_bin / 1e6:.6f} MHz\n{m_bin:.2f} dB"

        (xmin, xmax), (ymin, ymax) = vb.viewRange()
        dx = (xmax - xmin) * 0.01
        dy = (ymax - ymin) * 0.05

        self.hover_text.setText(text)
        self.hover_text.setPos(mouse_point.x() + dx, mouse_point.y() - dy)

    def update_plot(self):
        try:
            x = self.sdr.read_rx()

            fs = self.sdr.sample_rate
            lo = self.sdr.lo

            mag = self.proc.compute_psd_db(x, fs)
            freqs = self.proc.freqs_hz(fs, lo)

            self.curve.setData(freqs, mag)
            self.hover.update_axis(freqs, mag)

            if self.autospan_cb.isChecked():
                self.snap_x_to_span()

            if self.autoy_cb.isChecked():
                peak = float(np.max(mag))
                floor = float(np.median(mag))
                top = peak + 5.0
                bottom = floor - 40.0

                if top > 20.0:
                    top = 20.0
                if bottom < -200.0:
                    bottom = -200.0

                self.plot.setYRange(bottom, top, padding=0.0)
            else:
                try:
                    y0 = float(self.ymin_edit.text().strip())
                    y1 = float(self.ymax_edit.text().strip())
                    if y1 > y0:
                        self.plot.setYRange(y0, y1, padding=0.0)
                except Exception:
                    pass

            peak = float(np.max(mag))
            self.status.setText(
                f"LO {int(lo)}  SR {int(fs)}  BW {int(self.sdr.rf_bw)}  "
                f"Gain {self.sdr.gain_db} dB  Peak {peak:.1f} dB"
            )
        except Exception as e:
            self.status.setText(f"Error: {e}")


def main():
    cfg = SpectrumConfig(uri="ip:192.168.2.1")
    app = pg.mkQApp("Pluto Spectrum Viewer")
    win = SpectrumWindow(cfg)
    win.resize(1200, 700)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
