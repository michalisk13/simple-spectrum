"""Headless engine for SDR lifecycle and frame streaming."""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional

from pluto_spectrum_analyzer.config import SpectrumConfig
from pluto_spectrum_analyzer.dsp.processor import SpectrumProcessor
from pluto_spectrum_analyzer.protocol import (
    EngineErrorFrame,
    EngineFrame,
    EngineStatusFrame,
)
from pluto_spectrum_analyzer.sdr.pluto import PlutoSdr
from pluto_spectrum_analyzer.worker import SpectrumWorker


FrameCallback = Callable[[EngineFrame], None]


class Engine:
    """Owns SDR lifecycle, worker lifecycle, and streaming frames."""

    def __init__(self, cfg: SpectrumConfig):
        self.cfg = cfg
        self._cfg_lock = threading.Lock()
        self._sdr: Optional[PlutoSdr] = None
        self._proc = SpectrumProcessor(cfg.fft_size, cfg.window)
        self._worker: Optional[SpectrumWorker] = None
        self._subscribers: list[FrameCallback] = []
        self._stream_meta: Dict[str, object] = {
            "spectrogram_enabled": False,
            "spectrogram_rate": 15.0,
            "spectrogram_time_span_s": 10.0,
            "spectrogram_min_db": -120.0,
            "spectrogram_max_db": 0.0,
            "spectrogram_cmap": "viridis",
        }
        self._status = EngineStatusFrame(
            ts_monotonic_ns=self._now_ns(),
            connected=False,
            uri=self.cfg.uri,
            device_name="unknown",
            center_hz=float(self.cfg.center_hz),
            span_hz=float(self.cfg.sample_rate_hz),
            sample_rate_hz=float(self.cfg.sample_rate_hz),
            rf_bw_hz=float(self.cfg.rf_bw_hz),
            gain_mode=str(self.cfg.gain_mode),
            gain_db=float(self.cfg.gain_db),
            fft_size=int(self.cfg.fft_size),
            window=str(self.cfg.window),
            rbw_hz=float(self.cfg.rbw_hz),
            vbw_hz=float(self.cfg.vbw_hz),
            update_hz_target=self._target_update_hz(),
            update_hz_actual=0.0,
            frame_processing_ms_avg=0.0,
            frames_dropped=0,
            spectrogram_enabled=bool(self._stream_meta["spectrogram_enabled"]),
            spectrogram_rate=float(self._stream_meta["spectrogram_rate"]),
            spectrogram_time_span_s=float(self._stream_meta["spectrogram_time_span_s"]),
            message="disconnected",
        )
        self._last_error: Optional[EngineErrorFrame] = None
        self._running = True

    def subscribe(self, callback: FrameCallback) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: FrameCallback) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def status(self) -> EngineStatusFrame:
        return self._status

    @property
    def last_error(self) -> Optional[EngineErrorFrame]:
        return self._last_error

    def stream_metadata(self) -> Dict[str, object]:
        return dict(self._stream_meta)

    def connect(self, uri: Optional[str] = None) -> bool:
        if self._sdr is not None:
            return True
        if uri is not None:
            self.cfg.uri = uri
        try:
            self._sdr = PlutoSdr(self.cfg)
        except Exception as exc:
            self._sdr = None
            self._last_error = EngineErrorFrame(
                ts_monotonic_ns=self._now_ns(),
                error_code="sdr_connect_failed",
                message=str(exc) or "Failed to connect to SDR",
                recoverable=True,
            )
            self._emit(self._last_error)
            self._update_status(connected=False, message="connection failed")
            return False

        self._proc.update_fft_size(self.cfg.fft_size, self.cfg.window)
        self._worker = SpectrumWorker(
            self._sdr,
            self._proc,
            self.cfg,
            frame_cb=self._emit,
            error_cb=self._handle_worker_error,
            cfg_lock=self._cfg_lock,
        )
        self._worker.update_stream_metadata(self._stream_meta)
        self._worker.start()
        self._update_status(connected=True, message="connected")
        return True

    def disconnect(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker.join(timeout=1.0)
            self._worker = None
        if self._sdr is not None:
            try:
                self._sdr.close()
            except Exception:
                pass
            self._sdr = None
        self._update_status(connected=False, message="disconnected")

    def reconnect(self) -> bool:
        uri = self.cfg.uri
        self.disconnect()
        return self.connect(uri=uri)

    def set_running(self, enabled: bool) -> None:
        self._running = enabled
        state = "running" if enabled else "stopped"
        self._update_status(connected=self._status.connected, message=state)

    def apply_config(self, **updates: object) -> None:
        worker_updates: Dict[str, object] = {}
        stream_updates: Dict[str, object] = {}

        with self._cfg_lock:
            for key, value in updates.items():
                if key in self._stream_meta:
                    stream_updates[key] = value
                    continue
                if key == "span_hz":
                    span_hz = int(value)
                    self.cfg.sample_rate_hz = span_hz
                    worker_updates["span_hz"] = span_hz
                    continue
                if key == "sample_rate_hz":
                    span_hz = int(value)
                    self.cfg.sample_rate_hz = span_hz
                    worker_updates["span_hz"] = span_hz
                    continue
                if key == "uri":
                    self.cfg.uri = str(value)
                    continue
                if hasattr(self.cfg, key):
                    setattr(self.cfg, key, value)
                    worker_updates[key] = value
                    continue

            if stream_updates:
                self._stream_meta.update(stream_updates)

        if self._worker is not None:
            if worker_updates:
                self._worker.queue_config(worker_updates)
            if stream_updates:
                self._worker.update_stream_metadata(stream_updates)

    def _handle_worker_error(self, message: str) -> None:
        self._last_error = EngineErrorFrame(
            ts_monotonic_ns=self._now_ns(),
            error_code="worker_error",
            message=message or "Worker error",
            recoverable=True,
        )
        self._emit(self._last_error)
        self.disconnect()

    def _update_status(self, connected: bool, message: Optional[str] = None) -> None:
        self._status = EngineStatusFrame(
            ts_monotonic_ns=self._now_ns(),
            connected=connected,
            uri=self.cfg.uri if connected else None,
            device_name="unknown",
            center_hz=float(self.cfg.center_hz),
            span_hz=float(self.cfg.sample_rate_hz),
            sample_rate_hz=float(self.cfg.sample_rate_hz),
            rf_bw_hz=float(self.cfg.rf_bw_hz),
            gain_mode=str(self.cfg.gain_mode),
            gain_db=float(self.cfg.gain_db),
            fft_size=int(self.cfg.fft_size),
            window=str(self.cfg.window),
            rbw_hz=float(self.cfg.rbw_hz),
            vbw_hz=float(self.cfg.vbw_hz),
            update_hz_target=self._target_update_hz(),
            update_hz_actual=0.0,
            frame_processing_ms_avg=0.0,
            frames_dropped=0,
            spectrogram_enabled=bool(self._stream_meta["spectrogram_enabled"]),
            spectrogram_rate=float(self._stream_meta["spectrogram_rate"]),
            spectrogram_time_span_s=float(self._stream_meta["spectrogram_time_span_s"]),
            message=message,
        )
        self._emit(self._status)

    def _emit(self, frame: EngineFrame) -> None:
        for callback in list(self._subscribers):
            try:
                callback(frame)
            except Exception:
                continue

    @staticmethod
    def _now_ns() -> int:
        return int(time.monotonic() * 1e9)

    def _target_update_hz(self) -> float:
        return 1000.0 / max(1.0, float(self.cfg.update_ms))
