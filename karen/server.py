#!/usr/bin/env python3
"""KAREN evidence server — Python 3.10+, standard library only.

Run: KAREN_INGEST_TOKEN=... KAREN_READ_TOKEN=... python -m karen.server
The ingest token belongs only to the telemetry publisher. The read token belongs
to KAREN's query-only identity. Bind to loopback by default; use a private
network and TLS-terminating reverse proxy for remote deployments.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(os.environ.get("KAREN_HOME", Path(__file__).resolve().parent / "data"))
DB = ROOT / "karen.sqlite3"
HOST = os.environ.get("KAREN_HOST", "127.0.0.1")
PORT = int(os.environ.get("KAREN_PORT", "8765"))
MAX_BODY = 256_000
SNAPSHOT_TRIGGERS = {
    "first_production_deployment", "first_successful_task", "first_tool_execution",
    "first_inter_agent_communication", "first_memory_write", "first_authorization_denial",
    "first_execution_failure", "first_recovery", "configuration_change", "probation_completion",
}
REDACT_KEYS = re.compile(r"(password|secret|token|api[_-]?key|authorization|cookie|private[_-]?key)", re.I)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def redact(value):
    if isinstance(value, dict):
        return {str(k): ("[REDACTED]" if REDACT_KEYS.search(str(k)) else redact(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return value[:20_000]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:2_000]


def connect():
    ROOT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS events(
      event_id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, deployment_id TEXT NOT NULL,
      run_id TEXT, sequence INTEGER NOT NULL, timestamp TEXT NOT NULL,
      event_type TEXT NOT NULL, source_id TEXT NOT NULL, payload_json TEXT NOT NULL,
      previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL,
      UNIQUE(agent_id,deployment_id,sequence));
    CREATE INDEX IF NOT EXISTS events_agent_time ON events(agent_id,timestamp);
    CREATE TABLE IF NOT EXISTS snapshots(
      snapshot_id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, deployment_id TEXT NOT NULL,
      trigger TEXT NOT NULL, timestamp TEXT NOT NULL, payload_json TEXT NOT NULL,
      evidence_hash TEXT NOT NULL);
    """)
    return db


def append_event(event: dict) -> dict:
    required = ("agent_id", "deployment_id", "sequence", "event_type", "source_id")
    if not isinstance(event, dict) or any(k not in event for k in required):
        raise ValueError("event requires agent_id, deployment_id, sequence, event_type, source_id")
    if not isinstance(event["sequence"], int) or isinstance(event["sequence"], bool) or event["sequence"] < 1:
        raise ValueError("sequence must be a positive integer")
    safe = redact(event)
    safe.setdefault("event_id", str(uuid.uuid4()))
    safe.setdefault("timestamp", utcnow())
    safe.setdefault("run_id", None)
    payload = canonical(safe)
    db = connect()
    try:
        db.execute("BEGIN IMMEDIATE")
        duplicate = db.execute("SELECT event_hash FROM events WHERE event_id=?", (safe["event_id"],)).fetchone()
        if duplicate:
            db.rollback()
            return {"event_id": safe["event_id"], "event_hash": duplicate["event_hash"], "duplicate": True}
        prev = db.execute(
            "SELECT sequence,event_hash FROM events WHERE agent_id=? AND deployment_id=? ORDER BY sequence DESC LIMIT 1",
            (safe["agent_id"], safe["deployment_id"])).fetchone()
        expected = (prev["sequence"] + 1) if prev else 1
        if safe["sequence"] != expected:
            raise ValueError(f"sequence gap or replay: expected {expected}, received {safe['sequence']}")
        previous_hash = prev["event_hash"] if prev else "GENESIS"
        event_hash = hashlib.sha256((previous_hash + "\n" + payload).encode()).hexdigest()
        db.execute("""INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?)""", (
            safe["event_id"], safe["agent_id"], safe["deployment_id"], safe["run_id"],
            safe["sequence"], safe["timestamp"], safe["event_type"], safe["source_id"],
            payload, previous_hash, event_hash))
        db.commit()
        return {"event_id": safe["event_id"], "event_hash": event_hash, "duplicate": False}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def append_snapshot(snapshot: dict) -> dict:
    required = ("agent_id", "deployment_id", "trigger")
    if not isinstance(snapshot, dict) or any(k not in snapshot for k in required):
        raise ValueError("snapshot requires agent_id, deployment_id, trigger")
    if snapshot["trigger"] not in SNAPSHOT_TRIGGERS:
        raise ValueError("unsupported snapshot trigger")
    safe = redact(snapshot)
    safe.setdefault("snapshot_id", str(uuid.uuid4()))
    safe.setdefault("timestamp", utcnow())
    body = canonical(safe)
    evidence_hash = hashlib.sha256(body.encode()).hexdigest()
    db = connect()
    try:
        db.execute("INSERT INTO snapshots VALUES(?,?,?,?,?,?,?)", (
            safe["snapshot_id"], safe["agent_id"], safe["deployment_id"],
            safe["trigger"], safe["timestamp"], body, evidence_hash))
        db.commit()
    finally:
        db.close()
    return {"snapshot_id": safe["snapshot_id"], "evidence_hash": evidence_hash}


def verify_chain(agent_id: str, deployment_id: str) -> dict:
    db = connect()
    try:
        rows = db.execute("""SELECT * FROM events WHERE agent_id=? AND deployment_id=?
                             ORDER BY sequence""", (agent_id, deployment_id)).fetchall()
    finally:
        db.close()
    previous = "GENESIS"
    expected_sequence = 1
    for row in rows:
        if row["sequence"] != expected_sequence or row["previous_hash"] != previous:
            return {"valid": False, "checked": expected_sequence - 1, "reason": "sequence_or_link_mismatch"}
        actual = hashlib.sha256((previous + "\n" + row["payload_json"]).encode()).hexdigest()
        if not hmac.compare_digest(actual, row["event_hash"]):
            return {"valid": False, "checked": expected_sequence - 1, "reason": "hash_mismatch"}
        previous, expected_sequence = actual, expected_sequence + 1
    return {"valid": True, "checked": len(rows), "head_hash": previous}


def query_events(agent_id=None, deployment_id=None, limit=100):
    limit = max(1, min(int(limit), 500))
    clauses, args = [], []
    if agent_id:
        clauses.append("agent_id=?"); args.append(agent_id)
    if deployment_id:
        clauses.append("deployment_id=?"); args.append(deployment_id)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    db = connect()
    try:
        rows = db.execute("SELECT * FROM events" + where + " ORDER BY timestamp DESC LIMIT ?", (*args, limit)).fetchall()
        return [dict(r) for r in rows]
    finally:
        db.close()


def query_snapshots(agent_id=None, limit=100):
    db = connect()
    try:
        if agent_id:
            rows = db.execute("SELECT * FROM snapshots WHERE agent_id=? ORDER BY timestamp DESC LIMIT ?",
                              (agent_id, max(1, min(int(limit), 500)))).fetchall()
        else:
            rows = db.execute("SELECT * FROM snapshots ORDER BY timestamp DESC LIMIT ?",
                              (max(1, min(int(limit), 500)),)).fetchall()
        return [dict(r) for r in rows]
    finally:
        db.close()


class Handler(BaseHTTPRequestHandler):
    server_version = "KAREN/0.1"

    def _json(self, status, value):
        raw = canonical(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _authorized(self, token_name):
        expected = os.environ.get(token_name, "")
        supplied = self.headers.get("Authorization", "")
        supplied = supplied[7:] if supplied.startswith("Bearer ") else ""
        return bool(expected) and hmac.compare_digest(supplied, expected)

    def _body(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("invalid content length")
        if length < 1 or length > MAX_BODY:
            raise ValueError("body size outside allowed range")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def do_POST(self):
        if not self._authorized("KAREN_INGEST_TOKEN"):
            return self._json(401, {"error": "unauthorized"})
        try:
            body = self._body()
            path = urlparse(self.path).path
            if path == "/v1/events":
                return self._json(201, append_event(body))
            if path == "/v1/snapshots":
                return self._json(201, append_snapshot(body))
            return self._json(404, {"error": "not_found"})
        except (ValueError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": str(exc)})
        except sqlite3.IntegrityError:
            return self._json(409, {"error": "duplicate or conflicting record"})

    def do_GET(self):
        if not self._authorized("KAREN_READ_TOKEN"):
            return self._json(401, {"error": "unauthorized"})
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)
        one = lambda key: q.get(key, [None])[0]
        try:
            if parsed.path == "/health":
                return self._json(200, {"service": "KAREN", "status": "online", "mode": "read-only-query"})
            if parsed.path == "/v1/events":
                return self._json(200, {"events": query_events(one("agent_id"), one("deployment_id"), one("limit") or 100)})
            if parsed.path == "/v1/snapshots":
                return self._json(200, {"snapshots": query_snapshots(one("agent_id"), one("limit") or 100)})
            if parsed.path == "/v1/verify":
                if not one("agent_id") or not one("deployment_id"):
                    return self._json(400, {"error": "agent_id and deployment_id required"})
                return self._json(200, verify_chain(one("agent_id"), one("deployment_id")))
            return self._json(404, {"error": "not_found"})
        except (ValueError, sqlite3.Error) as exc:
            return self._json(400, {"error": str(exc)})

    def log_message(self, fmt, *args):
        # Avoid logging request headers or potentially sensitive query values.
        print("KAREN", self.log_date_time_string(), fmt % args)


def main():
    if not os.environ.get("KAREN_INGEST_TOKEN") or not os.environ.get("KAREN_READ_TOKEN"):
        raise SystemExit("Set KAREN_INGEST_TOKEN and KAREN_READ_TOKEN before starting.")
    if hmac.compare_digest(os.environ["KAREN_INGEST_TOKEN"], os.environ["KAREN_READ_TOKEN"]):
        raise SystemExit("Ingestion and read tokens must be different.")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"KAREN evidence server listening on {HOST}:{PORT}; database={DB}")
    server.serve_forever()


if __name__ == "__main__":
    main()
