# Phase 3 WebSocket streaming quick test

Minimal smoke checks for the Phase 3 WebSocket stream.

## Prereqs

1. Start the FastAPI server (example):
   ```bash
   python3 -m uvicorn pluto_spectrum_analyzer.server.app:create_app --factory --host 0.0.0.0 --port 8000
   ```
   If you see a 404 when connecting to `/ws/stream`, verify you started the server with the factory command above or use:
   ```bash
   python3 -m uvicorn pluto_spectrum_analyzer.server.app:app --host 0.0.0.0 --port 8000
   ```
2. Install the Python WebSocket client if you do not have it:
   ```bash
   python3 -m pip install websockets
   ```

## Minimal client script

Save as `phase3-test.py`, then run `python3 phase3-test.py`.

### Connect the SDR before testing (optional)

The WebSocket stream reports whatever the engine status is at connection time.
If you have an SDR attached, you still need to explicitly tell the backend to
connect before running the script. Otherwise the first status frame will show
`connected=false`.

```bash
curl -s -X POST http://localhost:8000/api/sdr/connect | jq
```

If you use a different server address, replace the URL or set `BASE_URL` in the
script below.

```python
import asyncio
import json
import uuid

import websockets

BASE_URL = "ws://localhost:8000/ws/stream"
MAX_FRAMES = 6


def _print_status(status: dict) -> None:
    print(
        "status:",
        f"seq={status['seq']}",
        f"connected={status['connected']}",
        f"message={status.get('message')!r}",
        f"center_hz={status.get('center_hz')}",
        f"span_hz={status.get('span_hz')}",
        f"fft_size={status.get('fft_size')}",
        f"spectrum_fps={status.get('spectrum_fps')}",
        f"spectrogram_fps={status.get('spectrogram_fps')}",
        f"processing_ms={status.get('processing_ms')}",
    )


def _print_meta(meta: dict) -> None:
    payload_id = meta.get("payload_id")
    payload_info = f"payload_id={payload_id}" if payload_id else "payload_id=None"
    print(
        "meta:",
        f"type={meta['type']}",
        f"seq={meta['seq']}",
        payload_info,
        f"ts_monotonic_ns={meta.get('ts_monotonic_ns')}",
    )


def _assert_payload(meta: dict, payload: bytes) -> None:
    assert payload[:4] == b"SPAY", "binary payload must include SPAY header"
    header_payload_id = str(uuid.UUID(bytes=payload[8:24]))
    assert header_payload_id == meta["payload_id"], "SPAY header payload_id mismatch"


async def main() -> None:
    async with websockets.connect(BASE_URL, max_size=None) as ws:
        # 1) Status frame should arrive immediately.
        status = json.loads(await ws.recv())
        assert status["type"] == "status", "first frame must be status"
        _print_status(status)

        last_seq = status["seq"]
        payload_ids = set()

        # 2) Read the next few frames to validate sequencing and payload order.
        for _ in range(MAX_FRAMES):
            raw = await ws.recv()
            if isinstance(raw, (bytes, bytearray)):
                raise AssertionError("unexpected binary frame without metadata")
            meta = json.loads(raw)
            _print_meta(meta)
            assert meta["seq"] > last_seq, "seq must increment"
            last_seq = meta["seq"]

            payload_id = meta.get("payload_id")
            if not payload_id:
                continue

            assert payload_id not in payload_ids, "payload_id must be unique"
            payload_ids.add(payload_id)

            # Binary payload must follow the metadata message.
            payload = await ws.recv()
            assert isinstance(payload, (bytes, bytearray)), "expected binary payload after meta"
            _assert_payload(meta, payload)


asyncio.run(main())
```

## Notes

- Set a different WebSocket URL by changing `BASE_URL` in the script.
- If the SDR is **disconnected**:
  - The first message is still a `status` frame with `connected=false` and
    `message="disconnected"` (or `"connection failed"` if the SDR connect attempt failed).
  - Spectrum/spectrogram frames are not emitted because the worker is not running.
  - The script will block waiting for metadata after the initial status; that is expected.
- If the SDR is **connected**:
  - The first message is a `status` frame with `connected=true` and `message="connected"`.
  - You should then see alternating `spectrum_meta` and `spectrogram_meta` JSON frames, each
    immediately followed by a binary payload that starts with the `SPAY` header.
