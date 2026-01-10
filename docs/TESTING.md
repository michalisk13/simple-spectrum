## Testing guide

### Quick checks
* Run unit tests (protocol schema and payload header vectors):

```bash
python -m pytest
```

### Hardware-dependent checks (manual)
These require a connected SDR and are not suitable for CI:

1. **Connect/disconnect flow**
   * Launch the PyQt app.
   * Connect to a valid SDR URI.
   * Disconnect and reconnect.
   * Verify no crashes and status updates are correct.

2. **Spectrum rendering**
   * Confirm trace updates without stalling.
   * Adjust FFT size, RBW/VBW, detector, and window.

3. **Spectrogram rendering**
   * Enable spectrogram panel and adjust rate/time-span.
   * Verify waterfall updates and colormap selection works.

### Phase-by-phase QA scenarios

#### Phase 1: Engine and protocol stabilization
* Engine starts with no SDR and reports a disconnected status.
* Config updates do not crash when SDR is absent.
* Engine frames use monotonic nanoseconds and no Qt types.
* PyQt still renders spectrum and spectrogram correctly.

#### Phase 2: REST API (FastAPI)
* GET /api/status returns a complete StatusFrame.
* POST /api/config accepts partial updates and returns updated config.
* POST /api/sdr/connect handles invalid URIs gracefully.
* Server stays up when SDR is absent.

#### Phase 3: WebSocket streaming
* Connection sends immediate StatusFrame.
* Spectrum meta JSON precedes binary payload.
* Payload IDs are unique, seq increments per message.
* Multiple clients can connect without crashing the server.
* Quick script available in `docs/phase3-tesd.md`.

#### Phase 4: Decimation and quantization
* Spectrum output capped to max bins while preserving endpoints.
* Spectrogram columns capped, optional quantization to u8.
* UI remains responsive at large FFT sizes.

#### Phase 5: Web UI scaffold
* `npm run dev` starts the app and loads layout with placeholders.
* Status bar reflects `/api/status` changes.

#### Phase 6: Web UI streaming
* Spectrum trace updates continuously without stutter.
* Spectrogram updates smoothly and uses canvas/WebGL rendering.
* Config changes propagate through REST and update the display.
