# WebSocket Testing Guide: Pluto Spectrum Analyzer

This document provides an overview of the real-time data architecture and clear, practical instructions for verifying that the WebSocket stream is functioning correctly in the Pluto Spectrum Analyzer project.

## 1. Key Insights & Architecture

The Pluto Spectrum Analyzer is designed for high-throughput, real-time data visualization. Understanding the following architectural choices will make WebSocket testing and debugging much easier:

- **Decoupled Data Flow**  
  Standard REST endpoints (via `/api`) are used for configuration, control, and low-frequency operations. High-frequency spectrum data is streamed exclusively over a dedicated WebSocket endpoint at `/ws/stream`.

- **20 FPS Animation Loop**  
  To avoid excessive React re-renders, the frontend does not update the UI on every WebSocket message. Instead, `useWebSocket.ts` maintains an internal buffer, and `useAnimationFrame.ts` pulls from that buffer at a fixed 20 FPS to drive spectrum and spectrogram rendering.

- **Typed Communication**  
  WebSocket payloads are defined in `frontend/src/ws/`. A mismatch between backend message formats and frontend TypeScript interfaces can result in silent failures or console warnings.

- **Proxy and Networking Model**  
  REST calls are proxied through Vite during development, but WebSockets typically connect directly to the backend. The WebSocket base URL is controlled via the `VITE_WS_BASE_URL` environment variable.

---

## 2. Default Endpoints

- Frontend Dev Server: `http://localhost:5173`
- Backend REST API: `http://localhost:8000` (proxied via `/api`)
- Backend WebSocket: `ws://localhost:8000/ws/stream`

---

## 3. Testing Methods

### Method A: Browser Developer Tools (Fastest Feedback)

This is the quickest way to confirm that the frontend is receiving real-time data.

1. Start the backend:
   ```bash
   python -m uvicorn pluto_spectrum_analyzer.server.app:app --reload --host 0.0.0.0 --port 8000
   ```

2. Start the frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. Open `http://localhost:5173` in your browser.
4. Open **Developer Tools (F12)** and go to the **Network** tab.
5. Filter by **WS** (WebSockets).
6. Click on the `stream` connection.
7. Open the **Messages** or **Frames** tab.

**Expected Results**
- Success: A continuous stream of incoming frames (binary or JSON).
- Failure: No messages, or the connection closes immediately after a `101 Switching Protocols` response.

---

### Method B: Command Line Test with `wscat`

This method isolates the backend and confirms that it is emitting data independently of the frontend.

1. Install `wscat`:
   ```bash
   npm install -g wscat
   ```

2. Connect to the WebSocket:
   ```bash
   wscat -c ws://localhost:8000/ws/stream
   ```

**Expected Results**
- A steady stream of raw messages printed to the terminal.
- If the connection fails, check backend logs for errors.

---

### Method C: Programmatic Test with Python

Use this script to validate connectivity, message size, and basic stream health.

```python
import asyncio
import websockets

async def check_spectrum_stream():
    url = "ws://localhost:8000/ws/stream"
    print(f"Attempting to connect to {url}...")
    try:
        async with websockets.connect(url) as ws:
            print("Connected successfully")
            for i in range(10):
                message = await ws.recv()
                print(f"Packet {i + 1}: {len(message)} bytes received")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(check_spectrum_stream())
```

---

## 4. Troubleshooting Guide

| Issue | Possible Cause | Resolution |
|------|---------------|------------|
| Connection refused | Backend not running | Verify `uvicorn` is running on port 8000 |
| 404 Not Found | Incorrect WebSocket path | Ensure backend defines `/ws/stream` |
| No incoming data | Generator or hardware inactive | Check backend data source or mock generator |
| UI not updating | Animation loop not firing | Inspect `useAnimationFrame.ts` |
| Works in CLI but not UI | Wrong WS base URL | Set `VITE_WS_BASE_URL` correctly |

---

## 5. Environment Configuration

If the backend is running on a different host or inside a container, create a `.env` file in the `frontend/` directory:

```env
VITE_WS_BASE_URL=ws://<HOSTNAME_OR_IP>:8000
```

Restart the frontend after changing environment variables.

---

## 6. Recommended Next Step

For frontend-only testing, consider adding a mock spectrum data generator on the backend. This allows you to validate rendering performance and 20 FPS behavior without requiring live RF hardware.
