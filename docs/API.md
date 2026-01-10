# API reference

This document summarizes the REST and WebSocket interfaces available in the
headless FastAPI server.

## Base URL
By default, the server listens on `http://127.0.0.1:8000` when launched with
Uvicorn. Adjust host/port as needed.

## REST endpoints

### GET `/api/status`
Returns the current status frame and the most recent error (if any).

**Response**
```json
{
  "status": { "connected": false, "center_hz": 100000000, "spectrum_fps": 0.0 },
  "error": null
}
```

### GET `/api/config`
Returns the current configuration and stream metadata (spectrogram settings).

**Response**
```json
{
  "config": { "center_hz": 100000000, "fft_size": 8192 },
  "stream": { "spectrogram_enabled": false, "spectrogram_rate": 15.0 }
}
```

### POST `/api/config`
Apply partial configuration updates. The payload is a JSON object containing any
subset of config or stream metadata keys.

**Request**
```json
{ "center_hz": 101000000, "spectrogram_enabled": true }
```

**Response**
```json
{
  "config": { "center_hz": 101000000, "fft_size": 8192 },
  "stream": { "spectrogram_enabled": true, "spectrogram_rate": 15.0 }
}
```

### POST `/api/sdr/connect`
Connect to the SDR. Optionally pass a custom URI.

**Request**
```json
{ "uri": "ip:192.168.2.1" }
```

**Response**
```json
{ "ok": true, "status": { "connected": true }, "error": null }
```

### POST `/api/sdr/disconnect`
Disconnect from the SDR.

**Response**
```json
{ "ok": true, "status": { "connected": false } }
```

### POST `/api/sdr/reconnect`
Disconnect and reconnect in one step.

**Response**
```json
{ "ok": true, "status": { "connected": true }, "error": null }
```

### POST `/api/sdr/test`
Attempt to connect and immediately disconnect (useful for validation). Optionally
pass a URI.

**Request**
```json
{ "uri": "ip:192.168.2.1" }
```

**Response**
```json
{ "ok": true, "status": { "connected": false }, "error": null }
```

### GET `/api/presets`
List available presets.

**Response**
```json
{ "presets": ["Fast View", "Wide Scan", "Measure"] }
```

### POST `/api/presets/apply`
Apply a preset. The Measure preset supports an optional `measure_detector`.

**Request**
```json
{ "name": "Measure", "measure_detector": "Peak" }
```

**Response**
```json
{ "ok": true, "config": { "config": {}, "stream": {} } }
```

## WebSocket streaming

### GET `ws://<host>:<port>/ws/stream`
The WebSocket stream sends:

1. An immediate `status` JSON frame on connection.
2. Spectrum metadata JSON frames followed by binary payloads.
3. Spectrogram metadata JSON frames followed by binary payloads.
4. Marker and error JSON frames as needed.

Binary payloads use the Protocol Contract v1.0 payload header and are keyed to
metadata frames by `payload_id`. See `pluto_spectrum_analyzer/protocol.py` for
wire details.

## Related docs
- Transition plan and phase definitions: `docs/PLAN.md`
- WebSocket quick tests: `docs/phase3-tesd.md`, `docs/phase4-test.md`
