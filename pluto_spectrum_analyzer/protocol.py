"""Frame schemas and transport helpers for spectrum analyzer streaming protocol.

Engine frames are internal and not wire format.
Wire format frames are dict objects built via helpers and validated against the
Protocol Contract v1.0 JSON schema. The binary payload header format is fixed
and matches Protocol Contract v1.0.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
import uuid
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

PROTO_VERSION = "1.0"
FRAME_TYPES = {
    "status",
    "spectrum_meta",
    "spectrogram_meta",
    "markers",
    "config_ack",
    "error",
}

BINARY_MAGIC = b"SPAY"
BINARY_HEADER_VERSION = 1
BINARY_KIND_SPECTRUM = 1
BINARY_KIND_SPECTROGRAM = 2
BINARY_HEADER_STRUCT = struct.Struct("<4sHH16sII")


def protocol_json_schema() -> dict[str, Any]:
    """Return the Protocol Contract v1.0 JSON schema for metadata frames."""

    base_fields = {
        "proto_version": {"const": PROTO_VERSION},
        "type": {"enum": sorted(FRAME_TYPES)},
        "ts_monotonic_ns": {"type": "integer", "minimum": 0},
        "seq": {"type": "integer", "minimum": 0},
        "session_id": {"type": "string", "format": "uuid"},
    }

    spectrum_meta_fields = {
        "payload_id": {"type": "string", "format": "uuid"},
        "freq_start_hz": {"type": "number"},
        "freq_stop_hz": {"type": "number"},
        "n_bins": {"type": "integer", "minimum": 1},
        "y_units": {"const": "dBFS"},
        "detector": {"type": "string"},
        "trace_mode": {"type": "string"},
        "rbw_hz": {"type": "number"},
        "vbw_hz": {"type": "number"},
        "fft_size": {"type": "integer", "minimum": 1},
        "window": {"type": "string"},
        "averaging_alpha": {"type": ["number", "null"]},
        "dtype": {"const": "f32"},
        "endianness": {"const": "LE"},
    }

    spectrogram_meta_fields = {
        "payload_id": {"type": "string", "format": "uuid"},
        "freq_start_hz": {"type": "number"},
        "freq_stop_hz": {"type": "number"},
        "n_cols": {"type": "integer", "minimum": 1},
        "row_ts_monotonic_ns": {"type": "integer", "minimum": 0},
        "db_min": {"type": "number"},
        "db_max": {"type": "number"},
        "colormap": {"type": "string"},
        "quantized": {"type": "boolean"},
        "dtype": {"enum": ["u8", "f32"]},
        "endianness": {"type": ["string", "null"]},
    }

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Spectrum Protocol v1.0 Metadata Frames",
        "type": "object",
        "oneOf": [
            {
                "title": "Status Frame",
                "type": "object",
                "properties": {
                    **base_fields,
                    "type": {"const": "status"},
                    "connected": {"type": "boolean"},
                    "uri": {"type": ["string", "null"]},
                    "device_name": {"type": "string"},
                    "center_hz": {"type": "number"},
                    "span_hz": {"type": "number"},
                    "sample_rate_hz": {"type": "number"},
                    "rf_bw_hz": {"type": "number"},
                    "gain_mode": {"type": "string"},
                    "gain_db": {"type": "number"},
                    "fft_size": {"type": "integer", "minimum": 1},
                    "window": {"type": "string"},
                    "rbw_hz": {"type": "number"},
                    "vbw_hz": {"type": "number"},
                    "update_hz_target": {"type": "number"},
                    "update_hz_actual": {"type": "number"},
                    "frame_processing_ms_avg": {"type": "number"},
                    "frames_dropped": {"type": "integer", "minimum": 0},
                    "spectrogram_enabled": {"type": "boolean"},
                    "spectrogram_rate": {"type": "number"},
                    "spectrogram_time_span_s": {"type": "number"},
                    "message": {"type": ["string", "null"]},
                },
                "required": [
                    "proto_version",
                    "type",
                    "ts_monotonic_ns",
                    "seq",
                    "session_id",
                    "connected",
                    "uri",
                    "device_name",
                    "center_hz",
                    "span_hz",
                    "sample_rate_hz",
                    "rf_bw_hz",
                    "gain_mode",
                    "gain_db",
                    "fft_size",
                    "window",
                    "rbw_hz",
                    "vbw_hz",
                    "update_hz_target",
                    "update_hz_actual",
                    "frame_processing_ms_avg",
                    "frames_dropped",
                    "spectrogram_enabled",
                    "spectrogram_rate",
                    "spectrogram_time_span_s",
                ],
                "additionalProperties": False,
            },
            {
                "title": "Spectrum Meta Frame",
                "type": "object",
                "properties": {
                    **base_fields,
                    "type": {"const": "spectrum_meta"},
                    **spectrum_meta_fields,
                },
                "required": [
                    "proto_version",
                    "type",
                    "ts_monotonic_ns",
                    "seq",
                    "session_id",
                    "payload_id",
                    "freq_start_hz",
                    "freq_stop_hz",
                    "n_bins",
                    "y_units",
                    "detector",
                    "trace_mode",
                    "rbw_hz",
                    "vbw_hz",
                    "fft_size",
                    "window",
                    "averaging_alpha",
                    "dtype",
                    "endianness",
                ],
                "additionalProperties": False,
            },
            {
                "title": "Spectrogram Meta Frame",
                "type": "object",
                "oneOf": [
                    {
                        "properties": {
                            **base_fields,
                            "type": {"const": "spectrogram_meta"},
                            **spectrogram_meta_fields,
                            "quantized": {"const": True},
                            "dtype": {"const": "u8"},
                            "endianness": {"type": ["null"]},
                        },
                        "required": [
                            "proto_version",
                            "type",
                            "ts_monotonic_ns",
                            "seq",
                            "session_id",
                            "payload_id",
                            "freq_start_hz",
                            "freq_stop_hz",
                            "n_cols",
                            "row_ts_monotonic_ns",
                            "db_min",
                            "db_max",
                            "colormap",
                            "quantized",
                            "dtype",
                        ],
                        "additionalProperties": False,
                    },
                    {
                        "properties": {
                            **base_fields,
                            "type": {"const": "spectrogram_meta"},
                            **spectrogram_meta_fields,
                            "quantized": {"const": False},
                            "dtype": {"const": "f32"},
                            "endianness": {"const": "LE"},
                        },
                        "required": [
                            "proto_version",
                            "type",
                            "ts_monotonic_ns",
                            "seq",
                            "session_id",
                            "payload_id",
                            "freq_start_hz",
                            "freq_stop_hz",
                            "n_cols",
                            "row_ts_monotonic_ns",
                            "db_min",
                            "db_max",
                            "colormap",
                            "quantized",
                            "dtype",
                            "endianness",
                        ],
                        "additionalProperties": False,
                    },
                ],
                "additionalProperties": False,
            },
            {
                "title": "Markers Frame",
                "type": "object",
                "properties": {
                    **base_fields,
                    "type": {"const": "markers"},
                    "markers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "freq_hz": {"type": "number"},
                                "amp_dbfs": {"type": "number"},
                                "label": {"type": "string"},
                            },
                            "required": ["id", "freq_hz", "amp_dbfs", "label"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["proto_version", "type", "ts_monotonic_ns", "seq", "session_id", "markers"],
                "additionalProperties": False,
            },
            {
                "title": "Config Ack Frame",
                "type": "object",
                "properties": {
                    **base_fields,
                    "type": {"const": "config_ack"},
                    "applied": {"type": "object"},
                },
                "required": ["proto_version", "type", "ts_monotonic_ns", "seq", "session_id", "applied"],
                "additionalProperties": False,
            },
            {
                "title": "Error Frame",
                "type": "object",
                "properties": {
                    **base_fields,
                    "type": {"const": "error"},
                    "error_code": {"type": "string"},
                    "message": {"type": "string"},
                    "details": {"type": ["object", "null"]},
                    "recoverable": {"type": "boolean"},
                },
                "required": [
                    "proto_version",
                    "type",
                    "ts_monotonic_ns",
                    "seq",
                    "session_id",
                    "error_code",
                    "message",
                    "recoverable",
                ],
                "additionalProperties": False,
            },
        ],
    }


def make_payload_header(kind: int, payload_id: uuid.UUID, element_count: int) -> bytes:
    """Create the 32-byte SPAY header for binary payloads."""

    payload_bytes = payload_id.bytes
    return BINARY_HEADER_STRUCT.pack(
        BINARY_MAGIC,
        BINARY_HEADER_VERSION,
        int(kind),
        payload_bytes,
        int(element_count),
        0,
    )


def parse_payload_header(raw: bytes) -> dict[str, Any]:
    """Parse a 32-byte SPAY header into a dict."""

    if len(raw) != BINARY_HEADER_STRUCT.size:
        raise ValueError("Invalid SPAY header length")
    magic, version, kind, payload_bytes, count, reserved = BINARY_HEADER_STRUCT.unpack(raw)
    if magic != BINARY_MAGIC:
        raise ValueError("Invalid SPAY magic")
    if version != BINARY_HEADER_VERSION:
        raise ValueError("Invalid SPAY version")
    if kind not in {BINARY_KIND_SPECTRUM, BINARY_KIND_SPECTROGRAM}:
        raise ValueError("Invalid SPAY kind")
    if reserved != 0:
        raise ValueError("Invalid SPAY reserved field")
    return {
        "magic": magic,
        "version": int(version),
        "kind": int(kind),
        "payload_id": str(uuid.UUID(bytes=payload_bytes)),
        "element_count": int(count),
        "reserved": int(reserved),
    }


def make_frame_base(
    *,
    frame_type: str,
    ts_monotonic_ns: int,
    seq: int,
    session_id: uuid.UUID,
) -> dict[str, Any]:
    """Build shared metadata fields for protocol frames."""

    if frame_type not in FRAME_TYPES:
        raise ValueError(f"Unsupported frame type: {frame_type}")
    return {
        "proto_version": PROTO_VERSION,
        "type": frame_type,
        "ts_monotonic_ns": int(ts_monotonic_ns),
        "seq": int(seq),
        "session_id": str(session_id),
    }


def make_spectrum_meta(
    *,
    base: Mapping[str, Any],
    payload_id: uuid.UUID,
    freq_start_hz: float,
    freq_stop_hz: float,
    n_bins: int,
    detector: str,
    trace_mode: str,
    rbw_hz: float,
    vbw_hz: float,
    fft_size: int,
    window: str,
    averaging_alpha: Optional[float],
) -> dict[str, Any]:
    meta = dict(base)
    meta.update(
        {
            "payload_id": str(payload_id),
            "freq_start_hz": float(freq_start_hz),
            "freq_stop_hz": float(freq_stop_hz),
            "n_bins": int(n_bins),
            "y_units": "dBFS",
            "detector": str(detector),
            "trace_mode": str(trace_mode),
            "rbw_hz": float(rbw_hz),
            "vbw_hz": float(vbw_hz),
            "fft_size": int(fft_size),
            "window": str(window),
            "averaging_alpha": averaging_alpha,
            "dtype": "f32",
            "endianness": "LE",
        }
    )
    return meta


def make_spectrogram_meta(
    *,
    base: Mapping[str, Any],
    payload_id: uuid.UUID,
    freq_start_hz: float,
    freq_stop_hz: float,
    n_cols: int,
    row_ts_monotonic_ns: int,
    db_min: float,
    db_max: float,
    colormap: str,
    quantized: bool,
    dtype: str,
) -> dict[str, Any]:
    if quantized:
        if dtype != "u8":
            raise ValueError("Quantized spectrograms must use dtype u8")
        endianness: Optional[str] = None
    else:
        if dtype != "f32":
            raise ValueError("Non-quantized spectrograms must use dtype f32")
        endianness = "LE"
    meta = dict(base)
    meta.update(
        {
            "payload_id": str(payload_id),
            "freq_start_hz": float(freq_start_hz),
            "freq_stop_hz": float(freq_stop_hz),
            "n_cols": int(n_cols),
            "row_ts_monotonic_ns": int(row_ts_monotonic_ns),
            "db_min": float(db_min),
            "db_max": float(db_max),
            "colormap": str(colormap),
            "quantized": bool(quantized),
            "dtype": dtype,
            "endianness": endianness,
        }
    )
    return meta


@dataclass(frozen=True)
class EngineStatusFrame:
    """Internal connection and runtime status."""

    ts_monotonic_ns: int
    connected: bool
    uri: Optional[str]
    device_name: str
    center_hz: float
    span_hz: float
    sample_rate_hz: float
    rf_bw_hz: float
    gain_mode: str
    gain_db: float
    fft_size: int
    window: str
    rbw_hz: float
    vbw_hz: float
    update_hz_target: float
    update_hz_actual: float
    frame_processing_ms_avg: float
    frames_dropped: int
    spectrogram_enabled: bool
    spectrogram_rate: float
    spectrogram_time_span_s: float
    message: Optional[str] = None
    ts_unix_ms: Optional[int] = None


@dataclass(frozen=True)
class EngineSpectrumFrame:
    """Internal spectrum payload."""

    ts_monotonic_ns: int
    freq_start_hz: float
    freq_stop_hz: float
    n_bins: int
    y: np.ndarray
    y_units: str
    rbw_hz: float
    vbw_hz: float
    fft_size: int
    avg_mode: str
    avg_count: int
    detector: str
    trace_mode: str
    averaging_alpha: Optional[float]
    window: str
    sample_rate_hz: float
    lo_hz: float
    rf_bw_hz: float
    gain_db: float
    fft_count: int
    enbw_bins: float
    ts_unix_ms: Optional[int] = None


@dataclass(frozen=True)
class EngineSpectrogramFrame:
    """Internal spectrogram slice payload."""

    ts_monotonic_ns: int
    row_ts_monotonic_ns: int
    row_db: np.ndarray
    col_count: int
    db_min: float
    db_max: float
    colormap: str
    time_span_s: float
    slice_rate_hz: float
    ts_unix_ms: Optional[int] = None


@dataclass(frozen=True)
class EngineMarker:
    """Internal marker data used for spectrum annotations."""

    id: str
    freq_hz: float
    amp_dbfs: float
    label: str


@dataclass(frozen=True)
class EngineMarkerFrame:
    """Internal marker update frame."""

    ts_monotonic_ns: int
    markers: Sequence[EngineMarker]
    ts_unix_ms: Optional[int] = None


@dataclass(frozen=True)
class EngineErrorFrame:
    """Internal error notifications."""

    ts_monotonic_ns: int
    error_code: str
    message: str
    details: Optional[Mapping[str, Any]] = None
    recoverable: bool = False
    ts_unix_ms: Optional[int] = None


EngineFrame = Union[
    EngineStatusFrame,
    EngineSpectrumFrame,
    EngineSpectrogramFrame,
    EngineMarkerFrame,
    EngineErrorFrame,
]


def engine_status_to_wire(
    frame: EngineStatusFrame,
    *,
    seq: int,
    session_id: uuid.UUID,
) -> dict[str, Any]:
    base = make_frame_base(
        frame_type="status",
        ts_monotonic_ns=frame.ts_monotonic_ns,
        seq=seq,
        session_id=session_id,
    )
    base.update(
        {
            "connected": frame.connected,
            "uri": frame.uri,
            "device_name": frame.device_name,
            "center_hz": frame.center_hz,
            "span_hz": frame.span_hz,
            "sample_rate_hz": frame.sample_rate_hz,
            "rf_bw_hz": frame.rf_bw_hz,
            "gain_mode": frame.gain_mode,
            "gain_db": frame.gain_db,
            "fft_size": frame.fft_size,
            "window": frame.window,
            "rbw_hz": frame.rbw_hz,
            "vbw_hz": frame.vbw_hz,
            "update_hz_target": frame.update_hz_target,
            "update_hz_actual": frame.update_hz_actual,
            "frame_processing_ms_avg": frame.frame_processing_ms_avg,
            "frames_dropped": frame.frames_dropped,
            "spectrogram_enabled": frame.spectrogram_enabled,
            "spectrogram_rate": frame.spectrogram_rate,
            "spectrogram_time_span_s": frame.spectrogram_time_span_s,
            "message": frame.message,
        }
    )
    return base


def engine_spectrum_meta_to_wire(
    frame: EngineSpectrumFrame,
    *,
    seq: int,
    session_id: uuid.UUID,
    payload_id: uuid.UUID,
) -> dict[str, Any]:
    base = make_frame_base(
        frame_type="spectrum_meta",
        ts_monotonic_ns=frame.ts_monotonic_ns,
        seq=seq,
        session_id=session_id,
    )
    return make_spectrum_meta(
        base=base,
        payload_id=payload_id,
        freq_start_hz=frame.freq_start_hz,
        freq_stop_hz=frame.freq_stop_hz,
        n_bins=frame.n_bins,
        detector=frame.detector,
        trace_mode=frame.trace_mode,
        rbw_hz=frame.rbw_hz,
        vbw_hz=frame.vbw_hz,
        fft_size=frame.fft_size,
        window=frame.window,
        averaging_alpha=frame.averaging_alpha,
    )


def engine_spectrogram_meta_to_wire(
    frame: EngineSpectrogramFrame,
    *,
    seq: int,
    session_id: uuid.UUID,
    payload_id: uuid.UUID,
    quantized: bool,
    dtype: str,
) -> dict[str, Any]:
    base = make_frame_base(
        frame_type="spectrogram_meta",
        ts_monotonic_ns=frame.ts_monotonic_ns,
        seq=seq,
        session_id=session_id,
    )
    return make_spectrogram_meta(
        base=base,
        payload_id=payload_id,
        freq_start_hz=frame.freq_start_hz,
        freq_stop_hz=frame.freq_stop_hz,
        n_cols=frame.col_count,
        row_ts_monotonic_ns=frame.row_ts_monotonic_ns,
        db_min=frame.db_min,
        db_max=frame.db_max,
        colormap=frame.colormap,
        quantized=quantized,
        dtype=dtype,
    )


def engine_markers_to_wire(
    frame: EngineMarkerFrame,
    *,
    seq: int,
    session_id: uuid.UUID,
) -> dict[str, Any]:
    base = make_frame_base(
        frame_type="markers",
        ts_monotonic_ns=frame.ts_monotonic_ns,
        seq=seq,
        session_id=session_id,
    )
    base["markers"] = [
        {
            "id": marker.id,
            "freq_hz": marker.freq_hz,
            "amp_dbfs": marker.amp_dbfs,
            "label": marker.label,
        }
        for marker in frame.markers
    ]
    return base


def engine_error_to_wire(
    frame: EngineErrorFrame,
    *,
    seq: int,
    session_id: uuid.UUID,
) -> dict[str, Any]:
    base = make_frame_base(
        frame_type="error",
        ts_monotonic_ns=frame.ts_monotonic_ns,
        seq=seq,
        session_id=session_id,
    )
    base.update(
        {
            "error_code": frame.error_code,
            "message": frame.message,
            "details": frame.details,
            "recoverable": frame.recoverable,
        }
    )
    return base
