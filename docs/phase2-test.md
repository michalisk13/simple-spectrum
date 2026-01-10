# Phase 2 REST API quick test (curl)

Minimal smoke checks for the Phase 2 REST endpoints.

## Prereqs

1. Start the FastAPI server (example):
   ```bash
   python3 -m uvicorn pluto_spectrum_analyzer.server.app:create_app --factory --host 0.0.0.0 --port 8000
   ```

## Minimal curl script

Save as `phase2-test.sh`, then run `bash phase2-test.sh`.

```bash
#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://localhost:8000}

printf "\n1) GET /api/status (expect connected=false when SDR absent)\n"
curl -s "$BASE_URL/api/status" | jq

printf "\n2) POST /api/sdr/connect (expect ok=false with error if SDR missing)\n"
curl -s -X POST "$BASE_URL/api/sdr/connect" | jq

printf "\n3) POST /api/config (partial update)\n"
curl -s -X POST "$BASE_URL/api/config" \
  -H 'Content-Type: application/json' \
  -d '{"center_hz": 2450000000}' | jq
```

## Notes

- If you want a different server address, set `BASE_URL` before running the script.
- `jq` is optional but makes the JSON readable.
