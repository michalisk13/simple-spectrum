# Phase 5 Web UI (Task D) Test Plan

## Purpose
Validate that the frontend canvas scaffolds render mock spectrum traces and
spectrogram rows at 20 fps without SVG usage or UI stutter.

## Prerequisites
- Backend running locally on `http://localhost:8000` (optional for this task).
- Frontend running locally on `http://localhost:5173`.

## Environment setup
1. Start the backend server (optional for mock rendering validation):
   ```bash
   python3 -m uvicorn pluto_spectrum_analyzer.server.app:app --reload --host 0.0.0.0 --port 8000
   ```
2. Start the frontend dev server:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Test cases
### 1) Spectrum canvas renders mock trace
1. Open the frontend in a browser.
2. Inspect the Spectrum panel.
3. Confirm a glowing polyline is animating.
4. Inspect the DOM in DevTools and verify the plot is a `<canvas>` element.

**Expected result**
- The spectrum trace animates smoothly (target ~20 fps).
- Rendering uses canvas only (no SVG elements for the plot).

### 2) Spectrogram canvas renders mock rows
1. Inspect the Spectrogram panel.
2. Confirm the waterfall updates with scrolling color bands.
3. Inspect the DOM in DevTools and verify the plot is a `<canvas>` element.

**Expected result**
- The spectrogram scrolls smoothly without stutter.
- Rendering uses canvas only (no SVG elements for the plot).

### 3) 20 fps stability check
1. Leave the UI running for at least 30 seconds.
2. Observe both plots during the interval.

**Expected result**
- Animation remains smooth at a steady cadence.
- No visual tearing or layout jitter in the panels.

## Notes
- The mock rendering runs on a 20 fps animation loop and does not require a live
  backend connection.
- If the backend is offline, the status bar may show a disconnected badge; the
  mock plots should continue to animate.
