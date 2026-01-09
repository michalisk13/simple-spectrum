"""Worker thread for SDR acquisition and DSP.

Runs FFT processing off the UI thread and emits data packets to the GUI. This
module must not import UI classes and only deals with SDR/DSP state.
"""

from __future__ import annotations

import time
from typing import Dict

from pyqtgraph.Qt import QtCore

from pluto_spectrum_analyzer.config import SpectrumConfig
from pluto_spectrum_analyzer.dsp.processor import SpectrumProcessor
from pluto_spectrum_analyzer.sdr.pluto import PlutoSdr


class SpectrumWorker(QtCore.QThread):
    new_data = QtCore.pyqtSignal(dict)
    connection_error = QtCore.pyqtSignal(str)

    def __init__(self, sdr: PlutoSdr, proc: SpectrumProcessor, cfg: SpectrumConfig):
        super().__init__()
        self.sdr = sdr
        self.proc = proc
        self.cfg = cfg
        self._lock = QtCore.QMutex()
        self._running = True
        # Pending configuration updates applied on the worker thread to avoid UI races.
        self.pending_apply: Dict[str, object] = {}

    def stop(self) -> None:
        self._running = False

    def queue_config(self, updates: Dict[str, object]) -> None:
        # Merge updates so multiple UI actions coalesce into one worker-side apply.
        with QtCore.QMutexLocker(self._lock):
            self.pending_apply.update(updates)

    def run(self) -> None:
        next_time = time.monotonic()
        while self._running:
            try:
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
                    update_ms = int(self.cfg.update_ms)
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
                next_time = time.monotonic() + (update_ms / 1000.0)
                sleep_time = max(0.0, next_time - time.monotonic())
                time.sleep(sleep_time)
            except Exception as exc:
                self._running = False
                self.connection_error.emit(str(exc))
                return
