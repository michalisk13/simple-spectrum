import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import adi


def dbfs(x, eps=1e-12):
    return 20.0 * np.log10(np.maximum(np.abs(x), eps))


class PlutoWifi24Spectrum(QtWidgets.QMainWindow):
    def __init__(self, uri="ip:192.168.2.1"):
        super().__init__()
        self.setWindowTitle("Pluto WiFi 2.4 Spectrum")

        self.uri = uri
        self.center_hz = 2_437_000_000
        self.sample_rate = 20_000_000
        self.rf_bw = 20_000_000
        self.gain_db = 55
        self.buf = 131072

        self.sdr = adi.Pluto(uri=self.uri)
        self.sdr.rx_enabled_channels = [0]
        self.sdr.rx_buffer_size = self.buf
        self.sdr.rx_destroy_buffer()

        self.sdr.sample_rate = int(self.sample_rate)
        self.sdr.rx_rf_bandwidth = int(self.rf_bw)
        self.sdr.gain_control_mode_chan0 = "manual"
        self.sdr.rx_hardwaregain_chan0 = int(self.gain_db)

        self.set_center(self.center_hz)

        # Use hanning window
        self.window = np.hanning(self.buf).astype(np.float32)

        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        self.freq_edit = QtWidgets.QLineEdit(str(self.center_hz))
        self.freq_edit.setPlaceholderText("Center frequency in Hz, eg 2437000000")

        # Center frequency input
        controls.addWidget(QtWidgets.QLabel("Center Freq:"))
        self.freq_edit = QtWidgets.QLineEdit("2437")
        self.freq_edit.setPlaceholderText("eg 2437")
        self.freq_edit.setFixedWidth(120)
        controls.addWidget(self.freq_edit)

        self.freq_unit = QtWidgets.QComboBox()
        self.freq_unit.addItems(["Hz", "kHz", "MHz", "GHz"])
        self.freq_unit.setCurrentText("MHz")
        self.freq_unit.setFixedWidth(80)
        controls.addWidget(self.freq_unit)

        self.set_btn = QtWidgets.QPushButton("Set")
        controls.addWidget(self.set_btn)
        self.set_btn.clicked.connect(self.on_set_center)

        self.status = QtWidgets.QLabel("")
        controls.addWidget(self.status, 1)

        self.plotw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plotw, 1)

        self.plot = self.plotw.addPlot()
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "Magnitude", units="dBFS")
        self.plot.showGrid(x=True, y=True)

        self.curve = self.plot.plot()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(120)

    def set_center(self, hz):
        hz = int(hz)
        if hz < 325_000_000:
            hz = 325_000_000
        if hz > 3_800_000_000:
            hz = 3_800_000_000
        self.sdr.rx_lo = hz
        self.center_hz = hz

        # Keep UI consistent, show center in selected unit
        if hasattr(self, "freq_unit") and hasattr(self, "freq_edit"):
            unit = self.freq_unit.currentText()
            inv = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}[unit]
            self.freq_edit.setText(f"{self.center_hz / inv:.6g}")



    def on_set_center(self):
        try:
            hz = self._parse_freq_to_hz(self.freq_edit.text(), self.freq_unit.currentText())
            self.set_center(hz)
            self.status.setText(f"Center set to {hz} Hz")
        except Exception as e:
            self.status.setText(f"Bad center freq: {e}")

    def _parse_freq_to_hz(self, value_str: str, unit: str) -> int:
        value = float(value_str.strip())

        scale = {
            "Hz": 1.0,
            "kHz": 1e3,
            "MHz": 1e6,
            "GHz": 1e9,
        }[unit]
        return int(value * scale)

    def read_samples(self):
        x = self.sdr.rx()
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x.astype(np.complex64)

    def update_plot(self):
        try:
            # Read complex baseband samples from Pluto (ADC output)
            x = self.read_samples()
            # Apply window function to reduce spectral leakage (Hanning window defined before)
            x = x * self.window

            # # Compute complex FFT and shift DC to the center of the spectrum
            X = np.fft.fftshift(np.fft.fft(x))
            
            # Power Spectral Density (PSD) estimation:
            psd = (np.abs(X) ** 2) / (np.sum(self.window ** 2) * float(self.sdr.sample_rate))
            # Convert PSD to logarithmic scale (dB)
            mag = 10 * np.log10(np.maximum(psd, 1e-20))

            # FFT bin frequencies (baseband) shifted to absolute RF
            sr = float(self.sdr.sample_rate)
            lo = float(self.sdr.rx_lo)
            freqs = np.fft.fftshift(np.fft.fftfreq(len(X), d=1.0 / sr)) + lo

            self.curve.setData(freqs, mag)
            #  Set visible frequency span to LO ± sample_rate/2
            self.plot.setXRange(lo - sr / 2.0, lo + sr / 2.0, padding=0.0)
            self.plot.setYRange(-120, 0, padding=0.0)

            peak = float(np.max(mag))
            self.status.setText(
                f"LO {int(lo)}  SR {int(sr)}  BW {int(self.sdr.rx_rf_bandwidth)}  Gain {int(self.sdr.rx_hardwaregain_chan0)} dB  Peak {peak:.1f} dBFS"
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
