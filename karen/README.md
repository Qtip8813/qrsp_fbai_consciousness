# KAREN — Independent Observability MVP

KAREN is a separate evidence service, not an Agent Hub worker. The MVP provides an authenticated ingestion boundary, a separate read-only query identity, SQLite event storage, SHA-256 hash chaining, vital snapshots, bounded queries, and basic redaction.

## Run

Python 3.10+; standard library only.

PowerShell:

```powershell
$env:KAREN_INGEST_TOKEN = "replace-with-long-random-ingest-secret"
$env:KAREN_READ_TOKEN = "a-different-long-random-read-secret"
$env:KAREN_HOME = "D:\KAREN\data"
python -m karen.server
```

The default bind is `127.0.0.1:8765`. Do not expose this development HTTP server directly to the internet. For remote deployment, use a private network and TLS reverse proxy, firewall rules, secret manager, and service-level authentication.

## Event ingestion

Only the telemetry publisher receives `KAREN_INGEST_TOKEN`.

```http
POST /v1/events
Authorization: Bearer <ingest-token>
Content-Type: application/json
```

```json
{
  "agent_id": "AGT-000001",
  "deployment_id": "DEPLOY-2026-001",
  "run_id": "workflow-123",
  "sequence": 1,
  "event_type": "deployment.started",
  "source_id": "agent-hub",
  "payload": {"agent_version": "1.0.0"}
}
```

Sequence numbers start at 1 and must be contiguous per agent/deployment stream. Replaying the same event ID is idempotent. The service redacts fields whose names resemble credentials before persistence.

## Vital snapshots

`POST /v1/snapshots` accepts only the predefined triggers in `server.py`. Snapshot payloads are hashed and stored separately from events.

## KAREN read-only queries

KAREN receives only `KAREN_READ_TOKEN`; it cannot ingest or mutate records.

- `GET /health`
- `GET /v1/events?agent_id=AGT-000001&deployment_id=DEPLOY-2026-001&limit=100`
- `GET /v1/snapshots?agent_id=AGT-000001`
- `GET /v1/verify?agent_id=AGT-000001&deployment_id=DEPLOY-2026-001`

All query routes require the read token. No update/delete route exists.

## Tests

From repository root:

```bash
python -m unittest -v karen.test_server
```

## Important limitations

- This is an MVP evidence receiver, not a hardened production server.
- Hash chains detect record alteration but are not immutable against a privileged database administrator who can rewrite the entire chain. Add separately stored signed checkpoints and independent backups.
- The MVP does not yet connect to Agent Hub. A publisher adapter must be added at Agent Hub lifecycle, tool, model, memory, authorization, and workflow event boundaries.
- Sequence allocation must be durable at the publisher and serialized per agent/deployment.
- Redaction is key-name based and is not a substitute for data minimization or payload allowlists.
- TLS, network ACLs, secret rotation, retention enforcement, backup verification, rate limiting, and operational monitoring remain deployment requirements.
- Visual recording is deliberately not enabled yet. Add it only with explicit application-scoped authorization and a redaction pipeline.
