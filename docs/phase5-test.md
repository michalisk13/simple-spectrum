# Phase 5 Web UI (Task C) Test Plan

## Purpose
Validate that the frontend WebSocket discovery hooks connect to the backend
`/ws/stream` endpoint, log the incoming status frames, and recover from
backend restarts without freezing the UI.

## Prerequisites
- Backend running locally on `http://localhost:8000`.
- Frontend running locally on `http://localhost:5173`.
- Optional: set `VITE_WS_BASE_URL` if the backend is hosted elsewhere.

## Environment setup
1. Start the backend server:
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
### 1) Initial WebSocket connect + StatusFrame logging
1. Open the frontend in a browser.
2. Open the browser DevTools console.
3. Confirm a log entry appears similar to:
   ```
   Received StatusFrame { frame_type: "status", ... }
   ```

**Expected result**
- Console logs at least one StatusFrame soon after page load.
- UI remains responsive while logs appear.

### 2) Backend restart reconnect
1. With the frontend running, stop the backend server.
2. Restart the backend server.
3. Monitor the console for a new StatusFrame log after restart.

**Expected result**
- WebSocket reconnects automatically.
- A new StatusFrame is logged after the backend is back online.
- No UI lockups or crashes.

### 3) Backend shutdown behavior
1. Stop the backend server while keeping the frontend running.
2. Continue interacting with the UI (sidebar toggle, panel resize).

**Expected result**
- UI remains responsive with no freezes.
- Connection attempts continue in the background until the backend returns.

## Notes
- If the backend is hosted remotely, set `VITE_WS_BASE_URL` to the base URL
  (for example, `ws://hostname:8000`) before running `npm run dev`.
