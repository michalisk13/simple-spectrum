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

```python
import asyncio
import json
import uuid

import websockets

BASE_URL = "ws://localhost:8000/ws/stream"


async def main() -> None:
    async with websockets.connect(BASE_URL) as ws:
        # 1) Status frame should arrive immediately.
        status = json.loads(await ws.recv())
        print("status:", status["type"], status["seq"])

        last_seq = status["seq"]
        payload_ids = set()

        # 2) Read the next few frames to validate sequencing and payload order.
        for _ in range(4):
            meta = json.loads(await ws.recv())
            print("meta:", meta["type"], meta["seq"])
            assert meta["seq"] > last_seq, "seq must increment"
            last_seq = meta["seq"]
            payload_id = meta.get("payload_id")
            if payload_id:
                assert payload_id not in payload_ids, "payload_id must be unique"
                payload_ids.add(payload_id)
                # Binary payload must follow the metadata message.
                payload = await ws.recv()
                assert isinstance(payload, (bytes, bytearray)), "expected binary payload after meta"
                assert payload[:4] == b"SPAY", "binary payload must include SPAY header"
                header_payload_id = str(uuid.UUID(bytes=payload[8:24]))
                assert header_payload_id == payload_id, "SPAY header payload_id mismatch"


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
