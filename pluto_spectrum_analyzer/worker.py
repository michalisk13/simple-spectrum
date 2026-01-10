"""Worker thread for SDR acquisition and DSP.

Runs FFT processing off the UI thread and emits protocol frames. This module
must not import UI classes and only deals with SDR/DSP state.
"""

from __future__ import annotations

import time
import threading
from typing import Dict, Optional

from pluto_spectrum_analyzer.config import SpectrumConfig
from pluto_spectrum_analyzer.dsp.processor import SpectrumProcessor
from pluto_spectrum_analyzer.protocol import (
    EngineSpectrogramFrame,
    EngineSpectrumFrame,
)
from pluto_spectrum_analyzer.sdr.pluto import PlutoSdr


class SpectrumWorker(threading.Thread):
    def __init__(
        self,
        sdr: PlutoSdr,
        proc: SpectrumProcessor,
        cfg: SpectrumConfig,
        frame_cb,
        error_cb,
        cfg_lock: Optional[threading.Lock] = None,
    ):
        super().__init__(daemon=True)
        self.sdr = sdr
        self.proc = proc
        self.cfg = cfg
        self._lock = cfg_lock or threading.Lock()
        self._running = threading.Event()
        self._running.set()
        self._frame_cb = frame_cb
        self._error_cb = error_cb
        # Pending configuration updates applied on the worker thread to avoid UI races.
        self.pending_apply: Dict[str, object] = {}
        self._stream_meta: Dict[str, object] = {
            "spectrogram_enabled": False,
            "spectrogram_rate": 15.0,
            "spectrogram_time_span_s": 10.0,
            "spectrogram_min_db": -120.0,
            "spectrogram_max_db": 0.0,
            "spectrogram_cmap": "viridis",
        }

    def stop(self) -> None:
        self._running.clear()

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def queue_config(self, updates: Dict[str, object]) -> None:
        # Merge updates so multiple UI actions coalesce into one worker-side apply.
        with self._lock:
            self.pending_apply.update(updates)

    def update_stream_metadata(self, updates: Dict[str, object]) -> None:
        with self._lock:
            self._stream_meta.update(updates)

    def run(self) -> None:
        next_time = time.monotonic()
        while self._running.is_set():
            try:
                with self._lock:
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

                        center_hz = pending.get("center_hz")
                        if center_hz is not None:
                            self.sdr.set_center_hz(int(center_hz))

                        gain_mode = pending.get("gain_mode")
                        if gain_mode is not None:
                            self.sdr.set_gain_mode(str(gain_mode))

                        gain_db = pending.get("gain_db")
                        if gain_db is not None:
                            self.sdr.set_gain_db(int(gain_db))

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
                    stream_meta = dict(self._stream_meta)

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
                freq_start = float(lo) - float(fs) / 2.0
                freq_stop = freq_start + float(fs) * (float(self.proc.fft_size - 1) / float(self.proc.fft_size))
                rbw = self.proc.rbw_hz(fs)

                ts_monotonic_ns = int(time.monotonic() * 1e9)
                spectrum_frame = EngineSpectrumFrame(
                    ts_monotonic_ns=ts_monotonic_ns,
                    freq_start_hz=freq_start,
                    freq_stop_hz=freq_stop,
                    n_bins=int(len(power)),
                    y=power,
                    y_units="linear",
                    rbw_hz=float(rbw),
                    vbw_hz=float(self.cfg.vbw_hz),
                    fft_size=int(self.proc.fft_size),
                    avg_mode=str(self.cfg.avg_mode),
                    avg_count=int(self.cfg.avg_count),
                    detector=str(self.cfg.detector),
                    trace_mode=str(self.cfg.trace_type),
                    averaging_alpha=None,
                    window=str(self.cfg.window),
                    sample_rate_hz=float(fs),
                    lo_hz=float(lo),
                    rf_bw_hz=float(self.sdr.rf_bw),
                    gain_db=float(self.sdr.gain_db),
                    fft_count=int(count),
                    enbw_bins=float(self.proc.enbw_bins),
                )
                spectrogram_frame = EngineSpectrogramFrame(
                    ts_monotonic_ns=ts_monotonic_ns,
                    row_ts_monotonic_ns=ts_monotonic_ns,
                    row_db=spectrogram_db,
                    col_count=int(len(spectrogram_db)),
                    db_min=float(stream_meta.get("spectrogram_min_db", -120.0)),
                    db_max=float(stream_meta.get("spectrogram_max_db", 0.0)),
                    colormap=str(stream_meta.get("spectrogram_cmap", "viridis")),
                    time_span_s=float(stream_meta.get("spectrogram_time_span_s", 0.0)),
                    slice_rate_hz=float(stream_meta.get("spectrogram_rate", 0.0)),
                )
                self._frame_cb(spectrum_frame)
                self._frame_cb(spectrogram_frame)

                # Pace updates based on UI update interval.
                next_time = time.monotonic() + (update_ms / 1000.0)
                sleep_time = max(0.0, next_time - time.monotonic())
                time.sleep(sleep_time)
            except Exception as exc:
                self._running.clear()
                self._error_cb(str(exc))
                return
