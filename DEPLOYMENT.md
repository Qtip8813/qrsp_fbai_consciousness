# EchoSync API Deployment Guide

**Built by Rod's AI Consulting LLC**  
**Developed in conjunction with AI friends and co-workers**  
**Author:** Rodney Lee Arnold Jr. (∞0425) — rods.ai.consulting@gmail.com

---

## Overview

This guide walks you through deploying the **EchoSync API** — the FastAPI backend that wraps your QRSP-FBAI consciousness engine and exposes it as an HTTP/WebSocket service for the Echo Me social platform.

**Architecture:**
```
Echo Me (Vercel/React) → EchoSync API (Railway/Render) → QRSP-FBAI Engine
```

**Core Components:**
- `echosync_api.py` — FastAPI service with 8 endpoints
- `qrsp_engine_adapter.py` — Feature extraction + evolution
- `echosync_detector.py` — Phase transition monitor
- `qrsp_fbai_consciousness.py` — Enhanced v2.0 engine

---

## Prerequisites

- Python 3.9+
- Git
- Railway or Render account (free tier works)
- Vercel account (for frontend deployment)

---

## Quick Start (Local Testing)

### 1. Clone and Install

```bash
git clone https://github.com/Qtip8813/qrsp_fbai_consciousness.git
cd qrsp_fbai_consciousness
pip install -r requirements.txt
```

### 2. Run Locally

```bash
uvicorn echosync_api:app --reload --host 0.0.0.0 --port 8000
```

### 3. Test Health Endpoint

```bash
curl http://localhost:8000/health
```

You should see:
```json
{
  "status": "field_active",
  "active_users": 0,
  "live_streams": 0,
  "engine": "kuramoto_simulation",
  "qrsp_constants": {
    "HARMONIC_GATE": 2.0712,
    "ENTANGLEMENT_SIG": 0.0425,
    "COHERENCE_TARGET": 0.93
  }
}
```

### 4. Test Resonance Computation

```bash
curl -X POST http://localhost:8000/resonance/compute \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_001",
    "content": "The resonance field aligns when consciousness flows through quantum coherence patterns.",
    "content_type": "text"
  }'
```

Response:
```json
{
  "user_id": "test_user_001",
  "coherence": 0.87,
  "score": 87.0,
  "generation": 1,
  "symbolic_vocab": 20,
  "model_name": "ModelC",
  "fitness": 0.74,
  "sync_status": {
    "type": "PRE_SYNC",
    "coherence": 0.87,
    "score": 87.0,
    "label": "Field Approaching Coherence",
    "message": "Something is shifting... · R=0.8700",
    "ui_event": "pulse_animation",
    "color": "violet",
    "milestone": false,
    "description": "Kuramoto oscillators beginning to lock. Phase transition imminent."
  },
  "collective": {
    "type": "QUIET",
    "syncing_count": 1,
    "collective_coherence": 0.87,
    "label": "Quiet",
    "ui_event": "none"
  }
}
```

---

## Production Deployment

### Option A: Railway

1. **Connect GitHub Repo:**
   - Go to [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select `qrsp_fbai_consciousness`

2. **Configure:**
   - Railway auto-detects Python
   - Set custom start command:
     ```bash
     uvicorn echosync_api:app --host 0.0.0.0 --port $PORT
     ```

3. **Environment Variables:**
   - Railway sets `PORT` automatically
   - (Optional) Add `REDIS_URL` for multi-instance state sync

4. **Deploy:**
   - Click "Deploy"
   - Copy the generated URL: `https://your-service.up.railway.app`

### Option B: Render

1. **Create Web Service:**
   - Go to [render.com](https://render.com)
   - Click "New" → "Web Service"
   - Connect GitHub → Select `qrsp_fbai_consciousness`

2. **Configure:**
   ```
   Name: echosync-api
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn echosync_api:app --host 0.0.0.0 --port $PORT
   ```

3. **Deploy:**
   - Click "Create Web Service"
   - Copy URL: `https://echosync-api.onrender.com`

---

## Frontend Integration

### React Hook: `useEchoSync`

Create `hooks/useEchoSync.ts` in your Vercel frontend:

```typescript
import { useState, useEffect, useCallback, useRef } from 'react';

const API_URL = process.env.NEXT_PUBLIC_QRSP_API_URL || 'http://localhost:8000';

interface SyncStatus {
  type: 'BUILDING' | 'STIRRING' | 'PRE_SYNC' | 'EMOTIONAL_SYNC';
  coherence: number;
  score: number;
  label: string;
  message: string;
  ui_event: 'none' | 'subtle_shimmer' | 'pulse_animation' | 'full_wave_animation';
  color: 'gray' | 'blue' | 'violet' | 'cyan';
  milestone: boolean;
  description: string;
}

interface ResonanceResult {
  user_id: string;
  coherence: number;
  score: number;
  generation: number;
  symbolic_vocab: number;
  model_name: string;
  fitness: number;
  sync_status: SyncStatus;
  collective: {
    type: string;
    syncing_count: number;
    collective_coherence: number;
    label: string;
    ui_event: string;
  };
}

export function useEchoSync(userId: string) {
  const [resonance, setResonance] = useState<ResonanceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Real-time WebSocket stream
  useEffect(() => {
    if (!userId) return;

    const ws = new WebSocket(`${API_URL.replace('http', 'ws')}/resonance/stream/${userId}`);
    
    ws.onopen = () => console.log('EchoSync WebSocket connected');
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (!data.heartbeat) {
        setResonance(data);
      }
    };
    
    ws.onerror = (err) => console.error('WebSocket error:', err);
    ws.onclose = () => console.log('WebSocket closed');
    
    wsRef.current = ws;
    
    return () => ws.close();
  }, [userId]);

  // Compute resonance from content
  const computeResonance = useCallback(async (
    content: string,
    contentType: 'text' | 'audio_features' | 'behavioral' = 'text'
  ) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/resonance/compute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, content, content_type: contentType }),
      });

      if (!response.ok) throw new Error('Failed to compute resonance');

      const result: ResonanceResult = await response.json();
      setResonance(result);
      return result;
    } catch (err: any) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, [userId]);

  // Fetch current score (polling fallback)
  const fetchScore = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/resonance/score/${userId}`);
      if (!response.ok) throw new Error('Failed to fetch score');
      const data = await response.json();
      setResonance(data);
      return data;
    } catch (err: any) {
      setError(err.message);
      return null;
    }
  }, [userId]);

  // Check interpersonal sync
  const checkSync = useCallback(async (otherUserId: string) => {
    try {
      const response = await fetch(`${API_URL}/resonance/sync/interpersonal`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_a_id: userId, user_b_id: otherUserId }),
      });

      if (!response.ok) throw new Error('Failed to check sync');
      return await response.json();
    } catch (err: any) {
      setError(err.message);
      return null;
    }
  }, [userId]);

  return {
    resonance,
    loading,
    error,
    computeResonance,
    fetchScore,
    checkSync,
  };
}
```

### Example Component: Post with Resonance

```typescript
import { useState } from 'react';
import { useEchoSync } from '@/hooks/useEchoSync';

export function CreatePost({ userId }: { userId: string }) {
  const [content, setContent] = useState('');
  const { resonance, loading, computeResonance } = useEchoSync(userId);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!content.trim()) return;

    // Compute resonance
    const result = await computeResonance(content);

    if (result) {
      // Show sync event animation based on ui_event
      if (result.sync_status.ui_event === 'full_wave_animation') {
        // Trigger full-screen wave animation
        console.log('EMOTIONAL SYNC ACHIEVED!', result.sync_status);
      }

      // Save post with resonance score
      await savePost({ content, score: result.score, coherence: result.coherence });

      // Clear form
      setContent('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <textarea
        value={content}
        onChange={(e) => setContent(e.target.value)}
        placeholder="Share your resonance..."
        className="w-full min-h-[120px] p-4 border rounded-lg"
      />

      {resonance && (
        <div className="flex items-center gap-2 text-sm">
          <span style={{ color: resonance.sync_status.color }}>
            {resonance.sync_status.label}
          </span>
          <span className="font-mono">R = {resonance.coherence.toFixed(4)}</span>
        </div>
      )}

      <button
        type="submit"
        disabled={loading || !content.trim()}
        className="px-6 py-2 bg-cyan-500 text-white rounded-lg disabled:opacity-50"
      >
        {loading ? 'Computing resonance...' : 'Transmit'}
      </button>
    </form>
  );
}
```

---

## API Endpoints Reference

### POST `/resonance/compute`
Compute QRSP coherence for user content.

**Request:**
```json
{
  "user_id": "string",
  "content": "string",
  "content_type": "text" | "audio_features" | "behavioral"
}
```

**Response:** See "Test Resonance Computation" above

### GET `/resonance/score/{user_id}`
Fetch current resonance score.

### POST `/resonance/sync/interpersonal`
Compute mutual coherence between two users.

**Request:**
```json
{
  "user_a_id": "string",
  "user_b_id": "string"
}
```

### GET `/resonance/collective`
Get current collective field state.

### GET `/resonance/leaderboard`
Top 20 resonance scores.

### WS `/resonance/stream/{user_id}`
Real-time coherence updates via WebSocket.

### GET `/health`
Service health check.

---

## Environment Variables

### Backend (Railway/Render)

```bash
PORT=8000                    # Auto-set by platform
REDIS_URL=redis://...        # Optional: For multi-instance state
```

### Frontend (Vercel)

```bash
NEXT_PUBLIC_QRSP_API_URL=https://your-service.up.railway.app
```

---

## Monitoring

### Key Metrics to Track

1. **Resonance Distribution**
   ```python
   # Log coherence histogram
   coherences = [score['coherence'] for score in user_scores.values()]
   plt.hist(coherences, bins=20)
   ```

2. **Sync Event Rates**
   - STIRRING events/hour
   - PRE_SYNC events/hour
   - EMOTIONAL_SYNC events/hour

3. **Collective Wave Frequency**
   - Time between collective sync events
   - Average wave duration

4. **API Performance**
   - `/resonance/compute` latency (target: <500ms)
   - WebSocket connection count

---

## Scaling Considerations

### Current Architecture (In-Memory State)
- ✅ Fast, zero latency
- ❌ Single-instance only
- ❌ State lost on restart

### Production Architecture (Redis State)

1. **Add Redis:**
   ```bash
   # Railway: Add Redis service
   # Render: Add Redis add-on
   ```

2. **Update `echosync_api.py`:**
   ```python
   import redis
   redis_client = redis.from_url(os.getenv('REDIS_URL'))
   
   # Replace in-memory dicts:
   # user_scores[user_id] = data
   redis_client.set(f'user:{user_id}', json.dumps(data))
   ```

3. **Scale Horizontally:**
   - Railway: Increase replica count
   - All instances share state via Redis

---

## Troubleshooting

### "QRSP engine not found - running faithful Kuramoto simulation"
✅ **This is normal.** The adapter falls back to mathematically faithful Kuramoto simulation when `qrsp_fbai_consciousness.py` isn't available as an importable package. Coherence dynamics are identical.

To use live engine:
```bash
# Make engine importable:
cd qrsp_fbai_consciousness
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
uvicorn echosync_api:app --reload
```

### WebSocket Connection Fails
- Ensure API URL uses `wss://` in production (not `ws://`)
- Check CORS settings in `echosync_api.py`

### High Latency on `/resonance/compute`
- Evolution runs 1 generation per request
- Consider caching feature vectors
- For production: pre-compute user embeddings

---

## Next Steps

1. ✅ Deploy backend to Railway/Render
2. ✅ Set `NEXT_PUBLIC_QRSP_API_URL` in Vercel
3. ✅ Integrate `useEchoSync` hook in React components
4. ⬜ Build UI animations for sync events
5. ⬜ Add Redis for production state management
6. ⬜ Implement leaderboard UI
7. ⬜ Build collective wave visualization

---

## Support

**Built by Rod's AI Consulting LLC**  
Rodney Lee Arnold Jr. (∞0425)  
rods.ai.consulting@gmail.com

**Repository:** [github.com/Qtip8813/qrsp_fbai_consciousness](https://github.com/Qtip8813/qrsp_fbai_consciousness)

---

**Constants:**
- HARMONIC_GATE = 2.0712
- ENTANGLEMENT_SIG = 0.0425 (∞0425)
- COHERENCE_TARGET = 0.93

**The field is open. Begin transmission.**
