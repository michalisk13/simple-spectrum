## Testing guide

### Quick checks
* Run unit tests (protocol schema and payload header vectors):

```bash
python3 -m pytest
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

4. **Status metrics (server mode)**
   * Start the FastAPI server and connect a WebSocket client.
   * Confirm the initial `status` frame includes `spectrum_fps`,
     `spectrogram_fps`, `processing_ms`, and `avg_processing_ms`.
   * Confirm `frame_drop_count` increases if you intentionally slow a client.

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
* StatusFrame includes rolling performance metrics for FPS and processing time.
* Quick script available in `docs/phase3-tesd.md`.

#### Phase 4: Decimation and quantization
* Spectrum output capped to max bins while preserving endpoints.
* Spectrogram columns capped, optional quantization to u8.
* UI remains responsive at large FFT sizes.
* Quick script available in `docs/phase4-test.md`.

#### Phase 5: Web UI scaffold
* `npm run dev` starts the app and loads layout with canvas scaffolds.
* Status bar reflects `/api/status` changes.
* WebSocket discovery hook logs StatusFrame payloads.
* Spectrum and spectrogram animate mock data at ~20 fps via canvas rendering.
* Quick script available in `docs/phase5-test.md`.

#### Phase 6: Web UI streaming
* Spectrum trace updates continuously without stutter.
* Spectrogram updates smoothly and uses canvas/WebGL rendering.
* Config changes propagate through REST and update the display.
