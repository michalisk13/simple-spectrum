## Transition plan

This plan keeps the existing PyQt application intact while introducing a
headless backend and a modern web UI, all without streaming raw IQ.

### Goals
* Keep SDR/DSP logic in Python.
* Add web UI without rewriting DSP pipeline.
* Keep PyQt working during transition.
* Support high-bandwidth SDRs by streaming display artifacts only.

### Non-negotiable constraints
* Never stream raw IQ to browsers.
* Stream only spectrum traces, spectrogram rows, and metadata.
* Preserve existing functionality, calibration, and saved state.
* Keep PyQt UI until web UI is feature complete.

---

## Phase 1: Engine & protocol foundation (complete)

**Step 1: Engine module**
* Add a headless Engine that manages SDR and worker lifecycle.
* Engine exposes connect/disconnect/reconnect/status and subscriptions.

**Step 2: Protocol module**
* Define Protocol Contract v1.0 JSON schema and binary header helpers.
* Define internal Engine frames and wire serialization helpers.

**Step 3: Worker emits frames**
* Worker continues DSP and emits engine spectrum/spectrogram frames.

**Step 4: PyQt consumes Engine frames**
* UI uses Engine for all SDR and config actions.
* UI updates from Engine frames only.

**Definition of done**
* Engine runs headless without Qt.
* PyQt UX unchanged from user perspective.
* Engine and protocol are stable and versioned.

---

## Phase 2: FastAPI backend (complete)

**Step 5: Add server package**
* Create `pluto_spectrum_analyzer/server/` with app factory, routes, WS.

**Step 6: REST endpoints**
* GET /api/status, /api/config.
* POST /api/config, /api/sdr/connect|disconnect|reconnect|test.
* Preset endpoints for parity with PyQt presets.

**Definition of done**
* All responses are JSON serializable.
* Errors are clear and server never crashes when SDR is absent.

---

## Phase 3: WebSocket streaming (complete)

**Step 7: WebSocket stream**
* WS /ws/stream sends StatusFrame immediately.
* Spectrum meta JSON precedes binary payload (SPAY header + payload).
* Spectrogram meta JSON precedes binary payload.

**Definition of done**
* Multiple clients receive frames concurrently.
* Disconnect/reconnect does not crash server.

---

## Phase 4: High-throughput display pipeline (complete)

**Step 8: Decimation and quantization**
* Add Engine display pipeline for spectrum and spectrogram decimation.
* Optional spectrogram quantization to u8.
* Ensure monotonic X and endpoint preservation.

**Step 9: Performance metrics**
* Track update rates, processing time, frames dropped.
* Surface metrics in StatusFrame.

**Definition of done**
* Stream bandwidth stays stable at large FFT sizes.
* UI remains responsive.

---

## Phase 5: Web UI scaffold (next)

**Step 10: Frontend project**
* React + TypeScript + Vite, Mantine or MUI, Tabler/Lucide icons.

**Step 11: Layout**
* Sidebar settings, spectrum plot, spectrogram panel, status bar.

**Definition of done**
* UI loads and can fetch /api/status.

---

## Phase 6: Web UI streaming

**Step 12: Streaming client**
* Connect to WS, decode meta + binary payloads.
* Render spectrum and spectrogram via Canvas/WebGL.
* Apply config changes via REST.

**Step 13: Compatibility**
* Keep PyQt as reference until feature parity.
* All DSP/config changes land in Engine first.

---

## QA scenarios per phase

See `docs/TESTING.md` for detailed QA scenarios and manual checks.
