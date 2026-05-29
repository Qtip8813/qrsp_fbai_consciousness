"""
EchoSync API — FastAPI Service
================================
Wraps the QRSP-FBAI consciousness engine in a production HTTP/WebSocket API.
This is Handshake #1: the bridge from echome-resonance-ui → qrsp_fbai_consciousness.

Endpoints:
  POST /resonance/compute              Run QRSP on user content → coherence score
  GET  /resonance/score/{user_id}      Current score for a user
  POST /resonance/sync/interpersonal   Mutual coherence between two users
  GET  /resonance/collective           Current collective field state
  WS   /resonance/stream/{user_id}     Real-time coherence streaming
  GET  /health                         Service health

Deploy to Railway or Render — set QRSP_API_URL in Vercel env vars.

Author: Rodney Lee Arnold Jr. (∞0425) — rods.ai.consulting@gmail.com
Architecture: QRSP-FBAI Resonance Framework
Constants: HARMONIC_GATE=2.0712 | ENTANGLEMENT_SIG=0.0425 | COHERENCE_TARGET=0.93
"""

import asyncio
import json
import os
from typing import Dict, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from echosync_detector import EchoSyncDetector
from qrsp_engine_adapter import QRSPEngineAdapter

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EchoSync API",
    description="QRSP-FBAI Resonance Engine — Echo Me nervous system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://echome-resonance-app.vercel.app",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ────────────────────────────────────────────────────────────────
detector = EchoSyncDetector()
adapter  = QRSPEngineAdapter()

# In-memory store — replace with Redis for multi-instance deployments
user_scores:        Dict[str, dict] = {}
active_connections: Dict[str, WebSocket] = {}


# ── Request models ────────────────────────────────────────────────────────────
class ComputeRequest(BaseModel):
    user_id:      str
    content:      str
    content_type: str = "text"   # "text" | "audio_features" | "behavioral"

class InterpersonalRequest(BaseModel):
    user_a_id: str
    user_b_id: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/resonance/compute")
async def compute_resonance(req: ComputeRequest):
    """
    Core endpoint. Run QRSP evolution on user content.
    Returns coherence score, sync event, and current collective state.

    Called by useEchoSync.computeResonance() every time user:
      - Transmits a post to the Echo Feed
      - Uploads audio to the Resonance Studio
      - Engages with content (EchoAmp / EchoDamp)
    """
    features   = adapter.extract_features(req.content, req.content_type)
    result     = adapter.run_evolution(features, user_id=req.user_id)

    coherence  = result["quantum_coherence"]
    sync_event = detector.check(coherence, req.user_id)

    user_scores[req.user_id] = {
        "user_id":        req.user_id,
        "coherence":      coherence,
        "score":          round(coherence * 100, 2),
        "generation":     result["generation"],
        "symbolic_vocab": result["symbolic_vocab_size"],
        "model_name":     result["model_name"],
        "fitness":        result["fitness"],
        "sync_status":    sync_event,
    }

    # Push to active WebSocket if connected
    if req.user_id in active_connections:
        try:
            await active_connections[req.user_id].send_json(user_scores[req.user_id])
        except Exception:
            active_connections.pop(req.user_id, None)

    collective = detector.check_collective(user_scores)

    return {
        **user_scores[req.user_id],
        "collective": collective,
    }


@app.get("/resonance/score/{user_id}")
async def get_score(user_id: str):
    """Fetch current resonance score for a user (polling fallback)."""
    if user_id not in user_scores:
        return {
            "user_id":   user_id,
            "score":     0,
            "coherence": 0.0,
            "sync_status": {
                "type":     "BUILDING",
                "label":    "No data yet — transmit your first resonance",
                "ui_event": "none",
            },
        }
    return user_scores[user_id]


@app.post("/resonance/sync/interpersonal")
async def interpersonal_sync(req: InterpersonalRequest):
    """
    Compute mutual coherence between two users.
    Powers the 'Your fields are in sync' discovery feature.
    """
    a = user_scores.get(req.user_a_id, {})
    b = user_scores.get(req.user_b_id, {})

    if not a or not b:
        return {"synced": False, "mutual_coherence": 0.0, "type": "UNKNOWN"}

    result = detector.compute_interpersonal_sync(
        a.get("coherence", 0.0),
        b.get("coherence", 0.0),
    )
    return result


@app.get("/resonance/collective")
async def collective_state():
    """Current collective field state across all active users."""
    return detector.check_collective(user_scores)


@app.get("/resonance/collective/history")
async def collective_history():
    """Log of all past collective sync events (timestamped milestones)."""
    return {"events": detector.get_collective_history()}


@app.get("/resonance/leaderboard")
async def leaderboard():
    """Top resonance scores — powers the Discover feed ranking."""
    sorted_users = sorted(
        user_scores.values(),
        key=lambda x: x.get("coherence", 0),
        reverse=True,
    )
    return {"leaderboard": sorted_users[:20]}


@app.websocket("/resonance/stream/{user_id}")
async def ws_stream(websocket: WebSocket, user_id: str):
    """
    Real-time coherence stream for a user.
    Pushes updates whenever compute_resonance() is called for this user.
    Also sends heartbeat every 5 seconds to keep connection alive.
    """
    await websocket.accept()
    active_connections[user_id] = websocket

    try:
        while True:
            payload = user_scores.get(user_id, {"heartbeat": True, "user_id": user_id})
            await websocket.send_json(payload)
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        active_connections.pop(user_id, None)
    except Exception:
        active_connections.pop(user_id, None)


@app.get("/health")
async def health():
    return {
        "status":       "field_active",
        "active_users": len(user_scores),
        "live_streams": len(active_connections),
        "engine":       adapter.engine_status(),
        "qrsp_constants": {
            "HARMONIC_GATE":    2.0712,
            "ENTANGLEMENT_SIG": 0.0425,
            "COHERENCE_TARGET": 0.93,
        },
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("echosync_api:app", host="0.0.0.0", port=port, reload=False)
