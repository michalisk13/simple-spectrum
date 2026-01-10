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

## Next incremental Web UI integration goals

### Task A: Frontend layout split and structure

**Scope**
* Split `frontend/src/App.tsx` into modular layout components.
* Implement AppShell-based layout structure with dedicated panels.
* Preserve the overall layout hierarchy while preparing for splitters.

**Files to touch**
* `frontend/src/App.tsx`
* `frontend/src/components/LayoutShell.tsx`
* `frontend/src/components/LeftSidebar.tsx`
* `frontend/src/components/StatusBar.tsx`
* `frontend/src/components/SpectrumPanel.tsx`
* `frontend/src/components/SpectrogramPanel.tsx`
* `frontend/src/components/layout/*.ts` (if needed for splitters or shared layout utilities)
* `frontend/src/styles/*` (if needed for layout styling)

**Done conditions**
* App loads with no console errors.
* Sidebar collapses and expands smoothly.
* Layout matches the Qt version: sidebar left, plots right, status top.
* Provide screenshot at 1366×768 and 1920×1080.

**Manual QA steps**
* Run `npm run dev` and load the app in the browser.
* Confirm sidebar toggle works and does not shift the main panel unexpectedly.
* Verify status bar remains fixed at the top and panels remain stacked.
* Capture screenshots at 1366×768 and 1920×1080.

**Mandatory frontend stack and styling requirements**
* Stack (mandatory): React + TypeScript + Vite.
* UI framework: Mantine (mandatory, do not use Tabler React as framework).
* Icons: Tabler Icons (SVG).
* Disallowed: heavy admin templates, SVG based charting, Plotly SVG mode, DOM rendering per FFT bin, Socket.IO.
* Rendering requirement: Spectrum and spectrogram must be custom Canvas or WebGL components.
* Modern UI requirements: match Qt layout but cleaner, more modern, and more user friendly.
* Layout requirements: left collapsible sidebar with icon section headers, top status bar, right area with spectrum above spectrogram, resizable panels via splitters.
* Visual style: dark theme near-black background, slightly lighter panels, subtle borders, rounded corners, clear hover/active states.
* Typography: single font family with clear hierarchy for title, section headers, labels, helper text.
* Buttons: icon + label where appropriate, proper hover/pressed states, no default browser look.
* UX requirements: disconnected mode is first class with clear badge, visible connect button, UI usable without SDR.

### Task B: API client wrapper

**Scope**
* Add a typed API client module to wrap REST calls.
* Wire StatusBar to show connection state from `/api/status`.
* Surface API errors through non-blocking UI notifications.

**Files to touch**
* `frontend/src/api/client.ts`
* `frontend/src/api/types.ts`
* `frontend/src/components/StatusBar.tsx`
* `frontend/src/components/notifications/*` (if needed for shared notification helpers)

**Done conditions**
* TypeScript strict mode passes.
* API errors are shown as non-blocking UI notifications.
* StatusBar shows Connected or Disconnected state from `/api/status`.

**Manual QA steps**
* Run `npm run lint` or `npm run typecheck` (as configured) and confirm strict mode passes.
* Stop the backend and confirm API errors show a non-blocking notification.
* Start the backend and confirm StatusBar shows Connected/Disconnected based on `/api/status`.

**Mandatory frontend stack and styling requirements**
* Stack (mandatory): React + TypeScript + Vite.
* UI framework: Mantine (mandatory, do not use Tabler React as framework).
* Icons: Tabler Icons (SVG).
* Disallowed: heavy admin templates, SVG based charting, Plotly SVG mode, DOM rendering per FFT bin, Socket.IO.
* Rendering requirement: Spectrum and spectrogram must be custom Canvas or WebGL components.
* Modern UI requirements: match Qt layout but cleaner, more modern, and more user friendly.
* Layout requirements: left collapsible sidebar with icon section headers, top status bar, right area with spectrum above spectrogram, resizable panels via splitters.
* Visual style: dark theme near-black background, slightly lighter panels, subtle borders, rounded corners, clear hover/active states.
* Typography: single font family with clear hierarchy for title, section headers, labels, helper text.
* Buttons: icon + label where appropriate, proper hover/pressed states, no default browser look.
* UX requirements: disconnected mode is first class with clear badge, visible connect button, UI usable without SDR.

### Task C: WebSocket discovery hooks (no rendering yet)

**Scope**
* Add a WebSocket client module prepared for `/ws/stream`.
* Reference the endpoint implemented in `pluto_spectrum_analyzer/server/ws.py`.
* Implement reconnect logic and StatusFrame logging for validation.

**Files to touch**
* `frontend/src/ws/client.ts`
* `frontend/src/ws/types.ts`
* `frontend/src/components/StatusBar.tsx` (if needed for connection indicator)
* `frontend/src/hooks/useWebSocket.ts`

**Done conditions**
* WebSocket connects and logs receipt of StatusFrame.
* Reconnect logic works after server restart.
* No UI freeze if WS is disconnected.

**Manual QA steps**
* Start the backend and web UI, then verify the console logs StatusFrame receipt.
* Restart the backend and confirm the WebSocket reconnects.
* Stop the backend and ensure the UI remains responsive with no freezes.

**Mandatory frontend stack and styling requirements**
* Stack (mandatory): React + TypeScript + Vite.
* UI framework: Mantine (mandatory, do not use Tabler React as framework).
* Icons: Tabler Icons (SVG).
* Disallowed: heavy admin templates, SVG based charting, Plotly SVG mode, DOM rendering per FFT bin, Socket.IO.
* Rendering requirement: Spectrum and spectrogram must be custom Canvas or WebGL components.
* Modern UI requirements: match Qt layout but cleaner, more modern, and more user friendly.
* Layout requirements: left collapsible sidebar with icon section headers, top status bar, right area with spectrum above spectrogram, resizable panels via splitters.
* Visual style: dark theme near-black background, slightly lighter panels, subtle borders, rounded corners, clear hover/active states.
* Typography: single font family with clear hierarchy for title, section headers, labels, helper text.
* Buttons: icon + label where appropriate, proper hover/pressed states, no default browser look.
* UX requirements: disconnected mode is first class with clear badge, visible connect button, UI usable without SDR.

### Task D: Replace plot placeholders with real rendering scaffolds

**Scope**
* Implement a Canvas or WebGL-based spectrum plot scaffold that renders a polyline from an array.
* Implement a Canvas-based spectrogram buffer scaffold that can accept rows.
* Render mock data at 20 fps to validate performance.

**Files to touch**
* `frontend/src/components/SpectrumPanel.tsx`
* `frontend/src/components/SpectrogramPanel.tsx`
* `frontend/src/components/plots/SpectrumCanvas.tsx`
* `frontend/src/components/plots/SpectrogramCanvas.tsx`
* `frontend/src/hooks/useAnimationFrame.ts`
* `frontend/src/utils/mockData.ts`

**Done conditions**
* Rendering uses Canvas or WebGL only.
* No SVG based charting libraries are used.
* A mock trace and mock spectrogram rows render smoothly at 20 fps in the browser.

**Manual QA steps**
* Run the web UI and verify the spectrum and spectrogram render mock data.
* Inspect the DOM to confirm rendering is Canvas/WebGL, not SVG.
* Observe animation for at least 30 seconds and confirm smooth 20 fps updates.

**Mandatory frontend stack and styling requirements**
* Stack (mandatory): React + TypeScript + Vite.
* UI framework: Mantine (mandatory, do not use Tabler React as framework).
* Icons: Tabler Icons (SVG).
* Disallowed: heavy admin templates, SVG based charting, Plotly SVG mode, DOM rendering per FFT bin, Socket.IO.
* Rendering requirement: Spectrum and spectrogram must be custom Canvas or WebGL components.
* Modern UI requirements: match Qt layout but cleaner, more modern, and more user friendly.
* Layout requirements: left collapsible sidebar with icon section headers, top status bar, right area with spectrum above spectrogram, resizable panels via splitters.
* Visual style: dark theme near-black background, slightly lighter panels, subtle borders, rounded corners, clear hover/active states.
* Typography: single font family with clear hierarchy for title, section headers, labels, helper text.
* Buttons: icon + label where appropriate, proper hover/pressed states, no default browser look.
* UX requirements: disconnected mode is first class with clear badge, visible connect button, UI usable without SDR.

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
