import uuid

import jsonschema

from pluto_spectrum_analyzer import protocol


def test_payload_header_vectors() -> None:
    vectors = [
        (
            protocol.BINARY_KIND_SPECTRUM,
            uuid.UUID("00112233-4455-6677-8899-aabbccddeeff"),
            1024,
            "535041590100010000112233445566778899aabbccddeeff0004000000000000",
        ),
        (
            protocol.BINARY_KIND_SPECTROGRAM,
            uuid.UUID("12345678-1234-5678-1234-567812345678"),
            4096,
            "5350415901000200123456781234567812345678123456780010000000000000",
        ),
    ]
    for kind, payload_id, count, expected_hex in vectors:
        raw = protocol.make_payload_header(kind, payload_id, count)
        assert raw.hex() == expected_hex


def test_schema_accepts_valid_frames() -> None:
    schema = protocol.protocol_json_schema()
    session_id = str(uuid.uuid4())

    status_frame = {
        "proto_version": protocol.PROTO_VERSION,
        "type": "status",
        "ts_monotonic_ns": 123,
        "seq": 1,
        "session_id": session_id,
        "connected": True,
        "uri": "ip:192.168.2.1",
        "device_name": "PlutoSDR",
        "center_hz": 2_400_000_000,
        "span_hz": 20_000_000,
        "sample_rate_hz": 20_000_000,
        "rf_bw_hz": 20_000_000,
        "gain_mode": "manual",
        "gain_db": 40.0,
        "fft_size": 8192,
        "window": "Blackman Harris",
        "rbw_hz": 5000.0,
        "vbw_hz": 3000.0,
        "update_hz_target": 10.0,
        "update_hz_actual": 9.5,
        "frame_processing_ms_avg": 5.0,
        "frames_dropped": 0,
        "spectrogram_enabled": True,
        "spectrogram_rate": 15.0,
        "spectrogram_time_span_s": 10.0,
        "message": "ok",
    }

    spectrum_meta = {
        "proto_version": protocol.PROTO_VERSION,
        "type": "spectrum_meta",
        "ts_monotonic_ns": 124,
        "seq": 2,
        "session_id": session_id,
        "payload_id": str(uuid.uuid4()),
        "freq_start_hz": 2_390_000_000.0,
        "freq_stop_hz": 2_410_000_000.0,
        "n_bins": 4096,
        "y_units": "dBFS",
        "detector": "RMS",
        "trace_mode": "Clear Write",
        "rbw_hz": 5000.0,
        "vbw_hz": 3000.0,
        "fft_size": 8192,
        "window": "Blackman Harris",
        "averaging_alpha": None,
        "dtype": "f32",
        "endianness": "LE",
    }

    spectrogram_meta = {
        "proto_version": protocol.PROTO_VERSION,
        "type": "spectrogram_meta",
        "ts_monotonic_ns": 125,
        "seq": 3,
        "session_id": session_id,
        "payload_id": str(uuid.uuid4()),
        "freq_start_hz": 2_390_000_000.0,
        "freq_stop_hz": 2_410_000_000.0,
        "n_cols": 1024,
        "row_ts_monotonic_ns": 124,
        "db_min": -120.0,
        "db_max": 0.0,
        "colormap": "viridis",
        "quantized": False,
        "dtype": "f32",
        "endianness": "LE",
    }

    jsonschema.validate(status_frame, schema)
    jsonschema.validate(spectrum_meta, schema)
    jsonschema.validate(spectrogram_meta, schema)


def test_schema_rejects_invalid_frames() -> None:
    schema = protocol.protocol_json_schema()
    session_id = str(uuid.uuid4())

    bad_quantized = {
        "proto_version": protocol.PROTO_VERSION,
        "type": "spectrogram_meta",
        "ts_monotonic_ns": 125,
        "seq": 3,
        "session_id": session_id,
        "payload_id": str(uuid.uuid4()),
        "freq_start_hz": 2_390_000_000.0,
        "freq_stop_hz": 2_410_000_000.0,
        "n_cols": 1024,
        "row_ts_monotonic_ns": 124,
        "db_min": -120.0,
        "db_max": 0.0,
        "colormap": "viridis",
        "quantized": True,
        "dtype": "f32",
    }

    bad_unquantized = {
        "proto_version": protocol.PROTO_VERSION,
        "type": "spectrogram_meta",
        "ts_monotonic_ns": 125,
        "seq": 3,
        "session_id": session_id,
        "payload_id": str(uuid.uuid4()),
        "freq_start_hz": 2_390_000_000.0,
        "freq_stop_hz": 2_410_000_000.0,
        "n_cols": 1024,
        "row_ts_monotonic_ns": 124,
        "db_min": -120.0,
        "db_max": 0.0,
        "colormap": "viridis",
        "quantized": False,
        "dtype": "u8",
        "endianness": "LE",
    }

    missing_status_field = {
        "proto_version": protocol.PROTO_VERSION,
        "type": "status",
        "ts_monotonic_ns": 123,
        "seq": 1,
        "session_id": session_id,
        "connected": True,
        "uri": "ip:192.168.2.1",
        "device_name": "PlutoSDR",
    }

    for payload in (bad_quantized, bad_unquantized, missing_status_field):
        try:
            jsonschema.validate(payload, schema)
        except jsonschema.ValidationError:
            continue
        raise AssertionError("Expected schema validation failure")
