# Phase 4 display pipeline quick test

Minimal smoke checks for display decimation and optional spectrogram quantization.

## Prereqs

1. Start the FastAPI server (example):
   ```bash
   python3 -m uvicorn pluto_spectrum_analyzer.server.app:create_app --factory --host 0.0.0.0 --port 8000
   ```
2. Install the Python WebSocket client if you do not have it:
   ```bash
   python3 -m pip install websockets
   ```

## Configure display caps (optional)

```bash
curl -s -X POST http://localhost:8000/api/config \
  -H 'Content-Type: application/json' \
  -d '{"max_spectrum_bins": 1024, "max_spectrogram_cols": 512, "spectrogram_quantize": true}' | jq
```

## Minimal client script

Save as `phase4-test.py`, then run `python3 phase4-test.py`.

```python
import asyncio
import json
import uuid

import websockets

BASE_URL = "ws://localhost:8000/ws/stream"
MAX_SPECTRUM_BINS = 1024
MAX_SPECTROGRAM_COLS = 512
EXPECT_QUANTIZED = True
MAX_FRAMES = 8
META_TIMEOUT_S = 3


def _parse_spay(payload: bytes) -> dict:
    if payload[:4] != b"SPAY":
        raise AssertionError("binary payload must include SPAY header")
    payload_id = str(uuid.UUID(bytes=payload[8:24]))
    element_count = int.from_bytes(payload[24:28], "little", signed=False)
    return {"payload_id": payload_id, "element_count": element_count}


async def main() -> None:
    async with websockets.connect(BASE_URL, max_size=None) as ws:
        status = json.loads(await ws.recv())
        assert status["type"] == "status", "first frame must be status"
        print("status:", status["connected"], status.get("message"))

        seen_spectrum = False
        seen_spectrogram = False

        for _ in range(MAX_FRAMES):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=META_TIMEOUT_S)
            except asyncio.TimeoutError:
                break
            if isinstance(raw, (bytes, bytearray)):
                raise AssertionError("unexpected binary frame without metadata")
            meta = json.loads(raw)
            if meta["type"] == "spectrum_meta":
                seen_spectrum = True
                assert meta["n_bins"] <= MAX_SPECTRUM_BINS, "spectrum bins should be capped"
            elif meta["type"] == "spectrogram_meta":
                seen_spectrogram = True
                assert meta["n_cols"] <= MAX_SPECTROGRAM_COLS, "spectrogram cols should be capped"
                assert meta["quantized"] is EXPECT_QUANTIZED, "quantized flag mismatch"
                if EXPECT_QUANTIZED:
                    assert meta["dtype"] == "u8", "expected u8 spectrogram payload"
                else:
                    assert meta["dtype"] == "f32", "expected f32 spectrogram payload"
            else:
                continue

            payload = await ws.recv()
            assert isinstance(payload, (bytes, bytearray)), "expected binary payload after meta"
            info = _parse_spay(payload)
            assert info["payload_id"] == meta["payload_id"], "SPAY header payload_id mismatch"
            if meta["type"] == "spectrogram_meta" and EXPECT_QUANTIZED:
                expected_len = 32 + meta["n_cols"]
            else:
                expected_len = 32 + meta["n_bins"] * 4 if meta["type"] == "spectrum_meta" else 32 + meta["n_cols"] * 4
            assert len(payload) == expected_len, "payload size mismatch"

        if status["connected"]:
            assert seen_spectrum, "expected at least one spectrum_meta frame"
            assert seen_spectrogram, "expected at least one spectrogram_meta frame"
        else:
            assert not seen_spectrum, "unexpected spectrum_meta while disconnected"
            assert not seen_spectrogram, "unexpected spectrogram_meta while disconnected"


asyncio.run(main())
```

## Notes

- If the SDR is **disconnected**, the only message sent is the initial `status` frame and no spectrum/spectrogram metadata or payloads are emitted. The script will exit after the timeout and print the status line (this is expected behavior).
- If the SDR is **connected**, you should see `spectrum_meta` and `spectrogram_meta` frames before the timeout and the assertions will confirm the capped sizes and quantization flags.
- Set different caps by changing `MAX_SPECTRUM_BINS`/`MAX_SPECTROGRAM_COLS` (and matching the `/api/config` call).
