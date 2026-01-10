# Frontend Development

## Overview
The frontend lives in `frontend/` and is built with Vite + React + TypeScript. During development it proxies API
requests to the FastAPI backend.

For REST calls, the UI uses a typed wrapper at `frontend/src/api/client.ts` and
shows failures with Mantine notifications via
`frontend/src/components/notifications/notify.tsx`.
WebSocket discovery lives in `frontend/src/hooks/useWebSocket.ts` with typed
helpers in `frontend/src/ws/`.

## Run the backend
```bash
python -m uvicorn pluto_spectrum_analyzer.server.app:app --reload --host 0.0.0.0 --port 8000
```

## Run the frontend
```bash
cd frontend
npm install
npm run dev
```

## Defaults
- Frontend dev server: http://localhost:5173
- Backend API: http://localhost:8000 (proxied via `/api`)
- Backend WebSocket: ws://localhost:8000/ws/stream

If you want to hit the API directly, update `frontend/vite.config.ts` or use the backend base URL in your
requests.
If the WebSocket backend is hosted elsewhere, set `VITE_WS_BASE_URL` before
running `npm run dev` (for example, `VITE_WS_BASE_URL=ws://hostname:8000`).
