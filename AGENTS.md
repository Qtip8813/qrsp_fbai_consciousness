# Base44 Dev Environment

## Project Overview
QRSP-FBAI EchoSync API — a Python FastAPI backend wrapping a quantum-evolutionary
computing engine. No frontend; the API is served directly on port 3000.

## Running the App
```bash
docker compose -f docker-compose.base44.yml up -d
```
- **Port:** 3000 (mapped from container port 3000)
- **Entry point:** `echosync_api:app` via uvicorn with `--reload`
- **Health check:** `GET /health`
- **Interactive docs:** `GET /docs` (Swagger UI) — root `/` redirects here

## Key Details
- Dependencies install on container startup from `requirements.txt` (plus `matplotlib`
  and `pytz` needed by `qrsp_fbai_consciousness.py` but absent from requirements.txt).
- The QRSP engine adapter gracefully falls back to a Kuramoto simulation if
  `qrsp_fbai_consciousness.py` cannot be imported (missing deps or import path issues).
- State is in-memory (no database, no Redis required for dev).
- No external secrets or credentials are needed.

## Sandbox Overrides
- `PORT=3000` is set via compose `environment:` (app reads `os.environ.get("PORT", 8000)`).
- `BASE44_PREVIEW_MODE` and `BASE44_*` host vars are passed through for sandbox compatibility.
- CORS already allows `http://localhost:3000`; no additional sandbox CORS changes needed.
