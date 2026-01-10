"""REST endpoints for the spectrum analyzer server."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request

from pluto_spectrum_analyzer.dsp.processor import SpectrumProcessor
from pluto_spectrum_analyzer.engine import Engine
from pluto_spectrum_analyzer.protocol import EngineErrorFrame


router = APIRouter()


def _engine(request: Request) -> Engine:
    return request.app.state.engine


def _serialize_error(error: EngineErrorFrame | None) -> dict[str, Any] | None:
    if error is None:
        return None
    return asdict(error)


def _serialize_status(engine: Engine) -> dict[str, Any]:
    return asdict(engine.status())


def _serialize_config(engine: Engine) -> dict[str, Any]:
    return {"config": asdict(engine.cfg), "stream": engine.stream_metadata()}


def _apply_preset(engine: Engine, preset_name: str, measure_detector: str | None) -> None:
    if preset_name == "Fast View":
        span_hz = 10_000_000
        fft_size = 2048
        update_hz = 15.0
        window = "Hann"
        detector = "RMS"
        overlap = 0.5
        vbw_mode = "Auto"
        vbw_hz = engine.cfg.vbw_hz
    elif preset_name == "Wide Scan":
        span_hz = 20_000_000
        fft_size = 8192
        update_hz = 10.0
        window = "Blackman Harris"
        detector = "RMS"
        overlap = 0.5
        vbw_mode = "Auto"
        vbw_hz = engine.cfg.vbw_hz
    elif preset_name == "Measure":
        span_hz = 5_000_000
        fft_size = 16384
        update_hz = 5.0
        window = "Blackman Harris"
        detector = measure_detector or engine.cfg.detector
        overlap = 0.5
        vbw_mode = "Manual"
        vbw_hz = 2000.0
    else:
        raise HTTPException(status_code=404, detail="Unknown preset")

    update_ms = int(max(50, min(2000, 1000.0 / update_hz)))
    temp_proc = SpectrumProcessor(fft_size, window)
    rbw_hz = temp_proc.rbw_hz(float(span_hz))
    engine.apply_config(
        sample_rate_hz=span_hz,
        rf_bw_hz=span_hz,
        fft_size=fft_size,
        update_ms=update_ms,
        window=window,
        detector=detector,
        overlap=overlap,
        vbw_mode=vbw_mode,
        vbw_hz=vbw_hz,
        rbw_mode="Manual",
        rbw_hz=rbw_hz,
    )


@router.get("/api/status")
def get_status(request: Request) -> dict[str, Any]:
    engine = _engine(request)
    return {
        "status": _serialize_status(engine),
        "error": _serialize_error(engine.last_error),
    }


@router.get("/api/config")
def get_config(request: Request) -> dict[str, Any]:
    return _serialize_config(_engine(request))


@router.post("/api/config")
def update_config(request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Config payload must be a JSON object")
    engine = _engine(request)
    engine.apply_config(**payload)
    return _serialize_config(engine)


@router.post("/api/sdr/connect")
def connect_sdr(request: Request, payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    engine = _engine(request)
    uri = None
    if payload:
        uri = payload.get("uri")
    if engine.status().connected:
        return {"ok": True, "status": _serialize_status(engine)}
    ok = engine.connect(uri=uri)
    return {
        "ok": ok,
        "status": _serialize_status(engine),
        "error": _serialize_error(engine.last_error),
    }


@router.post("/api/sdr/disconnect")
def disconnect_sdr(request: Request) -> dict[str, Any]:
    engine = _engine(request)
    engine.disconnect()
    return {"ok": True, "status": _serialize_status(engine)}


@router.post("/api/sdr/reconnect")
def reconnect_sdr(request: Request) -> dict[str, Any]:
    engine = _engine(request)
    ok = engine.reconnect()
    return {
        "ok": ok,
        "status": _serialize_status(engine),
        "error": _serialize_error(engine.last_error),
    }


@router.post("/api/sdr/test")
def test_sdr(request: Request, payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    engine = _engine(request)
    uri = None
    if payload:
        uri = payload.get("uri")
    if engine.status().connected:
        return {"ok": True, "status": _serialize_status(engine), "message": "already connected"}
    ok = engine.connect(uri=uri)
    if ok:
        engine.disconnect()
    return {
        "ok": ok,
        "status": _serialize_status(engine),
        "error": _serialize_error(engine.last_error),
    }


@router.get("/api/presets")
def list_presets() -> dict[str, Any]:
    return {"presets": ["Fast View", "Wide Scan", "Measure"]}


@router.post("/api/presets/apply")
def apply_preset(
    request: Request,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    preset_name = payload.get("name") if isinstance(payload, dict) else None
    if not preset_name:
        raise HTTPException(status_code=400, detail="Preset name is required")
    engine = _engine(request)
    measure_detector = payload.get("measure_detector")
    _apply_preset(engine, str(preset_name), measure_detector)
    return {"ok": True, "config": _serialize_config(engine)}
