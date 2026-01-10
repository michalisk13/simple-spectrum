"""WebSocket handlers for streaming frames."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np

from pluto_spectrum_analyzer.engine import Engine
from pluto_spectrum_analyzer.display import apply_spectrogram_display, apply_spectrum_display
from pluto_spectrum_analyzer.protocol import (
    BINARY_KIND_SPECTROGRAM,
    BINARY_KIND_SPECTRUM,
    EngineErrorFrame,
    EngineFrame,
    EngineMarkerFrame,
    EngineSpectrogramFrame,
    EngineSpectrumFrame,
    EngineStatusFrame,
    engine_error_to_wire,
    engine_markers_to_wire,
    engine_spectrogram_meta_to_wire,
    engine_spectrum_meta_to_wire,
    engine_status_to_wire,
    make_payload_header,
)


router = APIRouter()


@dataclass
class _ClientSession:
    websocket: WebSocket
    queue: asyncio.Queue[EngineFrame]
    session_id: uuid.UUID
    seq: int = 0

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq


class _StreamHub:
    """Fan out engine frames to multiple WebSocket clients."""

    def __init__(self, engine: Engine, loop: asyncio.AbstractEventLoop) -> None:
        self._engine = engine
        self._loop = loop
        self._clients: list[_ClientSession] = []
        # Subscribe once so engine frames are broadcast to all clients.
        self._engine.subscribe(self.publish)

    def register(self, session: _ClientSession) -> None:
        self._clients.append(session)

    def unregister(self, session: _ClientSession) -> None:
        if session in self._clients:
            self._clients.remove(session)

    def publish(self, frame: EngineFrame) -> None:
        # Engine callbacks run on worker threads, so hop back to the event loop.
        self._loop.call_soon_threadsafe(self._enqueue_frame, frame)

    def _enqueue_frame(self, frame: EngineFrame) -> None:
        for session in list(self._clients):
            try:
                session.queue.put_nowait(frame)
            except asyncio.QueueFull:
                # Drop frames if a client is slow to prevent blocking others.
                # Increment a drop counter so status metrics surface backpressure.
                self._engine.record_frame_dropped()
                continue


def _get_hub(websocket: WebSocket) -> _StreamHub:
    app = websocket.app
    hub = getattr(app.state, "ws_hub", None)
    if hub is None:
        hub = _StreamHub(app.state.engine, asyncio.get_running_loop())
        app.state.ws_hub = hub
    return hub


async def _send_status(session: _ClientSession, engine: Engine) -> None:
    status = engine.status()
    payload = engine_status_to_wire(status, seq=session.next_seq(), session_id=session.session_id)
    await session.websocket.send_json(payload)


async def _send_spectrum(session: _ClientSession, frame: EngineSpectrumFrame, engine: Engine) -> None:
    # Measure time spent preparing + sending spectrum payloads.
    start_time = time.perf_counter()
    display_frame = apply_spectrum_display(frame, engine.display_config())
    payload_id = uuid.uuid4()
    meta = engine_spectrum_meta_to_wire(
        display_frame,
        seq=session.next_seq(),
        session_id=session.session_id,
        payload_id=payload_id,
    )
    await session.websocket.send_json(meta)

    payload = display_frame.y.astype("<f4", copy=False).tobytes()
    header = make_payload_header(BINARY_KIND_SPECTRUM, payload_id, display_frame.y.size)
    await session.websocket.send_bytes(header + payload)
    # Record per-frame processing duration for rolling averages.
    processing_ms = (time.perf_counter() - start_time) * 1000.0
    engine.record_frame_processed("spectrum", processing_ms)


async def _send_spectrogram(session: _ClientSession, frame: EngineSpectrogramFrame, engine: Engine) -> None:
    # Measure time spent preparing + sending spectrogram payloads.
    start_time = time.perf_counter()
    display_frame, quantized, dtype = apply_spectrogram_display(frame, engine.display_config())
    payload_id = uuid.uuid4()
    meta = engine_spectrogram_meta_to_wire(
        display_frame,
        seq=session.next_seq(),
        session_id=session.session_id,
        payload_id=payload_id,
        quantized=quantized,
        dtype=dtype,
    )
    await session.websocket.send_json(meta)

    if quantized:
        payload = display_frame.row_db.astype(np.uint8, copy=False).tobytes()
    else:
        payload = display_frame.row_db.astype("<f4", copy=False).tobytes()
    header = make_payload_header(BINARY_KIND_SPECTROGRAM, payload_id, display_frame.row_db.size)
    await session.websocket.send_bytes(header + payload)
    # Record per-frame processing duration for rolling averages.
    processing_ms = (time.perf_counter() - start_time) * 1000.0
    engine.record_frame_processed("spectrogram", processing_ms)


async def _send_frame(session: _ClientSession, frame: EngineFrame, engine: Engine) -> None:
    if isinstance(frame, EngineStatusFrame):
        payload = engine_status_to_wire(frame, seq=session.next_seq(), session_id=session.session_id)
        await session.websocket.send_json(payload)
        return

    if isinstance(frame, EngineSpectrumFrame):
        await _send_spectrum(session, frame, engine)
        return

    if isinstance(frame, EngineSpectrogramFrame):
        await _send_spectrogram(session, frame, engine)
        return

    if isinstance(frame, EngineMarkerFrame):
        payload = engine_markers_to_wire(frame, seq=session.next_seq(), session_id=session.session_id)
        await session.websocket.send_json(payload)
        return

    if isinstance(frame, EngineErrorFrame):
        payload = engine_error_to_wire(frame, seq=session.next_seq(), session_id=session.session_id)
        await session.websocket.send_json(payload)


@router.websocket("/ws/stream")
async def stream(websocket: WebSocket) -> None:
    await websocket.accept()
    session = _ClientSession(
        websocket=websocket,
        queue=asyncio.Queue(maxsize=64),
        session_id=uuid.uuid4(),
    )
    engine: Engine = websocket.app.state.engine

    # Send status immediately before joining the broadcast stream.
    await _send_status(session, engine)
    hub = _get_hub(websocket)
    hub.register(session)

    try:
        while True:
            frame = await session.queue.get()
            await _send_frame(session, frame, engine)
    except WebSocketDisconnect:
        # Client disconnected; cleanup happens in finally.
        pass
    finally:
        hub.unregister(session)
