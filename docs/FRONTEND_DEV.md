# Frontend Development

## Overview
The frontend lives in `frontend/` and is built with Vite + React + TypeScript. During development it proxies API
requests to the FastAPI backend.

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

If you want to hit the API directly, update `frontend/vite.config.ts` or use the backend base URL in your
requests.
