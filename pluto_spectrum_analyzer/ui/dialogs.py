"""Dialog windows for calibration, SDR settings, and about/help.

Defines modal dialogs used by the GUI. This module should not perform SDR I/O
beyond the connection test helper or embed DSP logic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_spectrum_analyzer import __version__
from pluto_spectrum_analyzer.sdr.pluto import test_pluto_connection


class CalibrationDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget,
        offset_text: str,
        gain_text: str,
        tone_text: str,
        apply_cb: Callable[[str, str], None],
        calibrate_cb: Callable[[str, str], None],
        refresh_cb: Optional[Callable[[], tuple[str, str]]] = None,
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.apply_cb = apply_cb
        self.calibrate_cb = calibrate_cb
        self.refresh_cb = refresh_cb
        self.status_cb = status_cb

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.offset_edit = QtWidgets.QLineEdit(offset_text)
        self.offset_edit.setAlignment(QtCore.Qt.AlignRight)
        self.gain_edit = QtWidgets.QLineEdit(gain_text)
        self.gain_edit.setAlignment(QtCore.Qt.AlignRight)
        self.tone_edit = QtWidgets.QLineEdit(tone_text)
        self.tone_edit.setAlignment(QtCore.Qt.AlignRight)

        form.addRow("dBFS to dBm offset", self.offset_edit)
        form.addRow("External gain (dB)", self.gain_edit)
        form.addRow("Calibrate from peak (tone dBm)", self.tone_edit)
        layout.addLayout(form)

        calibrate_btn = QtWidgets.QPushButton("Calibrate from peak")
        layout.addWidget(calibrate_btn)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        layout.addWidget(buttons)

        calibrate_btn.clicked.connect(self._calibrate_from_peak)
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)

    def _apply(self) -> None:
        try:
            self.apply_cb(self.offset_edit.text(), self.gain_edit.text())
            if self.status_cb:
                self.status_cb("Calibration updated")
            self.accept()
        except Exception as exc:
            if self.status_cb:
                self.status_cb(f"Calibration error: {exc}")

    def _calibrate_from_peak(self) -> None:
        try:
            self.calibrate_cb(self.tone_edit.text(), self.gain_edit.text())
            if self.refresh_cb:
                offset_text, gain_text = self.refresh_cb()
                self.offset_edit.setText(offset_text)
                self.gain_edit.setText(gain_text)
            if self.status_cb:
                self.status_cb("Calibration updated from peak")
        except Exception as exc:
            if self.status_cb:
                self.status_cb(f"Calibration error: {exc}")


class SdrSettingsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget,
        current_uri: str,
        recent_uris: list[str],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("SDR Settings")
        self._uri = current_uri

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.uri_edit = QtWidgets.QLineEdit(current_uri)
        self.uri_edit.setPlaceholderText("ip:192.168.2.1")
        form.addRow("SDR URI", self.uri_edit)

        if recent_uris:
            self.recent_combo = QtWidgets.QComboBox()
            self.recent_combo.addItems(recent_uris)
            self.recent_combo.currentTextChanged.connect(self.uri_edit.setText)
            form.addRow("Recent", self.recent_combo)

        layout.addLayout(form)

        self.test_btn = QtWidgets.QPushButton("Test Connection")
        layout.addWidget(self.test_btn)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        layout.addWidget(buttons)

        self.test_btn.clicked.connect(self._test_connection)
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)

    @property
    def uri(self) -> str:
        return self._uri

    def _test_connection(self) -> None:
        uri = self.uri_edit.text().strip()
        ok, err = test_pluto_connection(uri)
        if ok:
            QtWidgets.QMessageBox.information(self, "Connection", "Pluto SDR connected.")
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Connection",
                f"Failed to connect: {err or 'unknown error'}",
            )

    def _on_save(self) -> None:
        self._uri = self.uri_edit.text().strip()
        self.accept()


class DeviceInfoDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget,
        sample_rate: float,
        lo: float,
        rf_bw: float,
        gain_mode: str,
        gain_db: int,
        fft_size: int,
        buffer_factor: int,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Device Info")
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        form.addRow("Sample rate", QtWidgets.QLabel(f"{sample_rate:.0f} Hz"))
        form.addRow("LO", QtWidgets.QLabel(f"{lo:.0f} Hz"))
        form.addRow("RF bandwidth", QtWidgets.QLabel(f"{rf_bw:.0f} Hz"))
        form.addRow("Gain mode", QtWidgets.QLabel(gain_mode))
        form.addRow("Gain", QtWidgets.QLabel(f"{gain_db} dB"))
        form.addRow("FFT size", QtWidgets.QLabel(str(fft_size)))
        form.addRow("Buffer factor", QtWidgets.QLabel(str(buffer_factor)))
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class AboutDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("About")
        layout = QtWidgets.QVBoxLayout(self)

        version_label = QtWidgets.QLabel(f"Pluto Spectrum Analyzer v{__version__}")
        version_label.setStyleSheet("font-weight: 600;")
        build_label = QtWidgets.QLabel(
            f"Build date: {datetime.utcnow().strftime('%Y-%m-%d')}")
        inspiration_label = QtWidgets.QLabel("Inspired by PySDR")

        layout.addWidget(version_label)
        layout.addWidget(build_label)
        layout.addWidget(inspiration_label)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
