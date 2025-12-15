import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import adi
from pyqtgraph import SignalProxy



class PlutoWifi24Spectrum(QtWidgets.QMainWindow):
    """
    Simple real time spectrum viewer for ADALM Pluto.

    Notes about scaling:
    This implementation computes a PSD style spectrum (power per Hz) using a
    window energy normalization and sample rate scaling. The values are relative,
    not calibrated dBm.
    """

    def __init__(self, uri="ip:192.168.2.1"):
        super().__init__()
        self.setWindowTitle("Pluto WiFi 2.4 Spectrum")

        # ----------------------------
        # Default SDR parameters
        # ----------------------------
        self.uri = uri
        self.center_hz = 2_437_000_000  # default WiFi channel 6 center
        self.sample_rate = 20_000_000   # visible span equals sample rate
        self.rf_bw = 20_000_000         # analog filter bandwidth
        self.gain_db = 55
        self.buf = 131072               # FFT size equals buffer size

        # ----------------------------
        # SDR initialization (Pluto wrapper)
        # ----------------------------
        self.sdr = adi.Pluto(uri=self.uri)

        # Force single RX channel and reset buffers early
        self.sdr.rx_enabled_channels = [0]
        self.sdr.rx_buffer_size = self.buf
        self.sdr.rx_destroy_buffer()

        # Apply base parameters
        self.sdr.sample_rate = int(self.sample_rate)
        self.sdr.rx_rf_bandwidth = int(self.rf_bw)
        self.sdr.gain_control_mode_chan0 = "manual"
        self.sdr.rx_hardwaregain_chan0 = int(self.gain_db)

        # Window for spectral leakage control
        self.window = np.hanning(self.buf).astype(np.float32)

        # ----------------------------
        # GUI layout
        # ----------------------------
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)

        # Top control panel with two rows:
        # Row 1: inputs and buttons
        # Row 2: status line
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(4)
        layout.addWidget(control_panel)

        # Row 1
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(10)
        control_layout.addLayout(row1)

        # Center frequency controls
        row1.addWidget(QtWidgets.QLabel("Center"))

        self.freq_edit = QtWidgets.QLineEdit("2437")
        self.freq_edit.setFixedWidth(90)
        self.freq_edit.setPlaceholderText("eg 2437")
        row1.addWidget(self.freq_edit)

        self.freq_unit = QtWidgets.QComboBox()
        self.freq_unit.addItems(["Hz", "kHz", "MHz", "GHz"])
        self.freq_unit.setCurrentText("MHz")
        self.freq_unit.setFixedWidth(80)
        row1.addWidget(self.freq_unit)

        self.set_btn = QtWidgets.QPushButton("Set")
        self.set_btn.clicked.connect(self.on_set_center)
        row1.addWidget(self.set_btn)

        # Span controls
        row1.addSpacing(20)
        row1.addWidget(QtWidgets.QLabel("Span"))

        self.span_cb = QtWidgets.QComboBox()
        self.span_cb.addItems(["1 MHz" ,"5 MHz", "10 MHz", "20 MHz", "30 MHz", "40 MHz",
                               "50 MHz"])
        self.span_cb.setCurrentText("20 MHz")
        self.span_cb.setFixedWidth(90)
        row1.addWidget(self.span_cb)

        self.apply_span_btn = QtWidgets.QPushButton("Apply")
        self.apply_span_btn.clicked.connect(self.on_apply_span)
        row1.addWidget(self.apply_span_btn)

        # Auto span toggle:
        # When enabled, the plot is forced to LO ± SR/2 every update.
        # When disabled, the user can zoom and pan without it snapping back.
        self.autospan_cb = QtWidgets.QCheckBox("Auto span")
        self.autospan_cb.setChecked(True)
        row1.addWidget(self.autospan_cb)

        # Y scale controls
        self.autoy_cb = QtWidgets.QCheckBox("Auto Y")
        self.autoy_cb.setChecked(True)
        self.autoy_cb.toggled.connect(self.on_autoy_toggled)
        row1.addWidget(self.autoy_cb)

        row1.addWidget(QtWidgets.QLabel("Y min"))
        self.ymin_edit = QtWidgets.QLineEdit("-120")
        self.ymin_edit.setFixedWidth(70)
        row1.addWidget(self.ymin_edit)

        row1.addWidget(QtWidgets.QLabel("Y max"))
        self.ymax_edit = QtWidgets.QLineEdit("0")
        self.ymax_edit.setFixedWidth(70)
        row1.addWidget(self.ymax_edit)

        # Apply manual Y once, without waiting for a timer tick
        self.apply_y_btn = QtWidgets.QPushButton("Apply Y")
        self.apply_y_btn.clicked.connect(self.apply_manual_y_once)
        row1.addWidget(self.apply_y_btn)


        # Keep controls left aligned
        row1.addStretch(1)

        # Row 2 status line
        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("QLabel { padding: 4px; }")
        control_layout.addWidget(self.status)

        # Plot area
        self.plotw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plotw, 1)

        self.plot = self.plotw.addPlot()
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "Magnitude", units="dB")
        self.plot.showGrid(x=True, y=True)

        # Disable y zoom by default, spectrum viewers usually zoom x only
        self.plot.setMouseEnabled(x=True, y=False)

        # Visible curve
        self.curve = self.plot.plot()

        # Speed up rendering for large arrays
        self.plot.setClipToView(True)
        self.curve.setClipToView(True)
        self.curve.setDownsampling(auto=True, method="peak")


        # Crosshair lines
        self.vline = pg.InfiniteLine(angle=90, movable=False)
        self.plot.addItem(self.vline, ignoreBounds=True)

        # Tooltip style hover box
        self.hover_text = pg.TextItem(
            "",
            anchor=(0, 1),  # top left of the box sits at the position we set
            color=(255, 255, 255),
            fill=pg.mkBrush(0, 0, 0, 180)  # semi transparent background
        )
        self.hover_text.setZValue(1e6)
        self.plot.addItem(self.hover_text)


        # Hover state for fast O(1) lookup
        self._last_mag = None
        self._last_f0 = None
        self._last_df = None
        self._last_n = None
        self._last_hover_idx = None


        # Keep last spectrum so we can query magnitude at the cursor
        self._last_freqs = None
        self._last_mag = None

        # Last hover index to avoid redundant updates
        self._last_hover_idx = None
        self._last_f0 = None
        self._last_df = None
        self._last_n = None

        # Mouse move handler, rate limited so it stays fast [update: Set the proxy rateLimit lower, to 20 Hz]
        self._mouse_proxy = SignalProxy(
            self.plot.scene().sigMouseMoved,
            rateLimit=20,
            slot=self.on_mouse_moved
        )


        # Now that UI exists, set center frequency safely
        self.set_center(self.center_hz)

        # Update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(220)

    # ----------------------------
    # Helpers
    # ----------------------------
    def _parse_freq_to_hz(self, value_str: str, unit: str) -> int:
        """
        Convert user input to Hz based on selected unit.
        Accepts floats so 2.437 GHz works.
        """
        value = float(value_str.strip())
        scale = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}[unit]
        return int(value * scale)

    def _update_freq_edit_from_center(self):
        """
        Keep the frequency textbox consistent with current LO and selected unit.
        This should only be called after widgets exist.
        """
        unit = self.freq_unit.currentText()
        inv = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}[unit]
        self.freq_edit.setText(f"{self.center_hz / inv:.6g}")

    # ----------------------------
    # SDR control
    # ----------------------------
    def set_center(self, hz: int):
        """
        Set Pluto LO, clamp to typical supported range.
        """
        hz = int(hz)
        if hz < 325_000_000:
            hz = 325_000_000
        if hz > 3_800_000_000:
            hz = 3_800_000_000

        self.sdr.rx_lo = hz
        self.center_hz = hz

        # Update UI if present
        if hasattr(self, "freq_edit") and hasattr(self, "freq_unit"):
            self._update_freq_edit_from_center()

        # Always snap view once after LO change so the plot stays consistent
        if hasattr(self, "plot"):
            self.snap_x_to_span()

    def on_set_center(self):
        """
        UI handler: set center frequency from textbox and unit selector.
        """
        try:
            hz = self._parse_freq_to_hz(self.freq_edit.text(), self.freq_unit.currentText())
            self.set_center(hz)
            self.status.setText(f"Center set to {hz} Hz")
        except Exception as e:
            self.status.setText(f"Bad center freq: {e}")

    def on_apply_span(self):
        """
        UI handler: apply a new span by changing sample rate and RF bandwidth.

        Important:
        Visible span equals sample rate. A larger sample rate gives a wider view.
        After changing sample rate or buffer sizes, destroy the RX buffer to force
        a clean reconfiguration.
        """
        try:
            text = self.span_cb.currentText().strip()
            mhz = float(text.split()[0])
            sr = int(mhz * 1e6)

            # Update SDR parameters
            self.sdr.sample_rate = sr
            self.sdr.rx_rf_bandwidth = sr

            # Reset buffers after rate change
            self.sdr.rx_destroy_buffer()

            # Keep FFT size fixed for simplicity
            self.buf = 131072
            self.sdr.rx_buffer_size = self.buf
            self.sdr.rx_destroy_buffer()

            # Rebuild window to match buffer size
            self.window = np.hanning(self.buf).astype(np.float32)

            self.status.setText(f"Span set to {mhz:g} MHz")

            # Auto snap
            self.snap_x_to_span()
        except Exception as e:
            self.status.setText(f"Span apply failed: {e}")

    def snap_x_to_span(self):
        lo = float(self.sdr.rx_lo)
        sr = float(self.sdr.sample_rate)
        self.plot.setXRange(lo - sr / 2.0, lo + sr / 2.0, padding=0.0)


    def on_autoy_toggled(self, checked: bool):
        # Disable manual fields when Auto Y is enabled
        self.ymin_edit.setEnabled(not checked)
        self.ymax_edit.setEnabled(not checked)
        self.apply_y_btn.setEnabled(not checked)

    def apply_manual_y_once(self):
        # Apply manual Y immediately when user clicks Apply Y
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

        # Only when inside plot area
        if not vb.sceneBoundingRect().contains(pos):
            self.hover_text.setText("")
            return

        # Convert to plot coordinates
        mouse_point = vb.mapSceneToView(pos)
        fx = float(mouse_point.x())

        # O(1) lookup needs spectrum state
        if self._last_mag is None or self._last_f0 is None:
            text = f"{fx/1e6:.6f} MHz"
        else:
            f0 = self._last_f0
            df = self._last_df
            n = self._last_n

            idx = int((fx - f0) / df)
            if idx < 0:
                idx = 0
            elif idx >= n:
                idx = n - 1

            if idx == self._last_hover_idx:
                return
            self._last_hover_idx = idx

            f_bin = f0 + idx * df
            m_bin = float(self._last_mag[idx])
            text = f"{f_bin/1e6:.6f} MHz\n{m_bin:.2f} dB"

        # Place tooltip slightly offset from cursor so it does not sit under it
        # Use view range to compute a sensible offset independent of zoom
        (xmin, xmax), (ymin, ymax) = vb.viewRange()
        dx = (xmax - xmin) * 0.01
        dy = (ymax - ymin) * 0.05

        self.hover_text.setText(text)
        self.hover_text.setPos(mouse_point.x() + dx, mouse_point.y() - dy)


    # ----------------------------
    # Streaming and plotting
    # ----------------------------
    def read_samples(self) -> np.ndarray:
        """
        Read complex samples from Pluto.
        pyadi-iio may return a list even for one channel, so unwrap if needed.
        """
        x = self.sdr.rx()
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x.astype(np.complex64)

    def update_plot(self):
        """
        Timer callback:
        Pull samples, compute PSD spectrum, update plot and status.
        """
        try:
            # Read baseband IQ samples
            x = self.read_samples()

            # Windowing reduces leakage
            x = x * self.window

            # FFT and center it
            X = np.fft.fftshift(np.fft.fft(x))

            # PSD estimate:
            # |X|^2 gives power per bin
            # divide by window energy and sample rate to get power per Hz
            Fs = float(self.sdr.sample_rate)
            psd = (np.abs(X) ** 2) / (np.sum(self.window ** 2) * Fs)

            # Log scale, clamp avoids log(0)
            mag = 10.0 * np.log10(np.maximum(psd, 1e-20))

            # Build absolute RF frequency axis
            sr = float(self.sdr.sample_rate)
            lo = float(self.sdr.rx_lo)
            freqs = np.fft.fftshift(np.fft.fftfreq(len(X), d=1.0 / sr)) + lo

            # Update curve
            self.curve.setData(freqs, mag)
            # Store parameters needed for hover readout (avoid argmin over full array)
            self._last_mag = mag
            self._last_f0 = float(freqs[0])
            self._last_df = float(freqs[1] - freqs[0])
            self._last_n = int(len(freqs))



            # Auto span behavior: snap x axis to LO ± SR/2 when enabled
            if self.autospan_cb.isChecked():
                self.plot.setXRange(lo - sr / 2.0, lo + sr / 2.0, padding=0.0)

            # Y range:
            # PSD values are often lower than dBFS style plots.
            if self.autoy_cb.isChecked():
                # Auto Y: keep the trace visible using a robust estimate
                peak = float(np.max(mag))
                floor = float(np.median(mag))

                top = peak + 5.0
                bottom = floor - 40.0

                # Safety clamps so it does not explode if something goes weird
                if top > 20.0:
                    top = 20.0
                if bottom < -200.0:
                    bottom = -200.0

                self.plot.setYRange(bottom, top, padding=0.0)
            else:
                # Manual Y: always use the user supplied limits
                try:
                    y0 = float(self.ymin_edit.text().strip())
                    y1 = float(self.ymax_edit.text().strip())
                    if y1 > y0:
                        self.plot.setYRange(y0, y1, padding=0.0)
                except Exception:
                    # Do not spam status on every frame if user is mid edit
                    pass

            peak = float(np.max(mag))
            self.status.setText(
                f"LO {int(lo)}  SR {int(sr)}  BW {int(self.sdr.rx_rf_bandwidth)}  "
                f"Gain {int(self.sdr.rx_hardwaregain_chan0)} dB  Peak {peak:.1f} dB"
            )
        except Exception as e:
            self.status.setText(f"Error: {e}")


def main():
    app = pg.mkQApp("Pluto WiFi 2.4 Spectrum")
    win = PlutoWifi24Spectrum(uri="ip:192.168.2.1")
    win.resize(1200, 700)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
