import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import server


class KarenEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"KAREN_HOME": self.tmp.name})
        self.env.start()
        server.ROOT = Path(self.tmp.name)
        server.DB = server.ROOT / "test.sqlite3"

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def event(self, sequence, **extra):
        return {"agent_id": "AGT-1", "deployment_id": "DEP-1",
                "sequence": sequence, "event_type": "tool.completed",
                "source_id": "agent-hub", **extra}

    def test_hash_chain_and_verification(self):
        server.append_event(self.event(1))
        server.append_event(self.event(2))
        self.assertTrue(server.verify_chain("AGT-1", "DEP-1")["valid"])
        self.assertEqual(server.verify_chain("AGT-1", "DEP-1")["checked"], 2)

    def test_sequence_gap_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 1"):
            server.append_event(self.event(2))

    def test_secret_fields_redacted(self):
        server.append_event(self.event(1, payload={"api_key": "do-not-store", "result": "ok"}))
        record = server.query_events("AGT-1")[0]
        self.assertNotIn("do-not-store", record["payload_json"])
        self.assertIn("[REDACTED]", record["payload_json"])

    def test_duplicate_event_id_is_idempotent(self):
        e = self.event(1, event_id="fixed-event")
        first = server.append_event(e)
        second = server.append_event(e)
        self.assertEqual(first["event_hash"], second["event_hash"])
        self.assertTrue(second["duplicate"])

    def test_snapshot_trigger_validation(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            server.append_snapshot({"agent_id": "A", "deployment_id": "D", "trigger": "random"})
        saved = server.append_snapshot({"agent_id": "A", "deployment_id": "D",
                                        "trigger": "first_production_deployment"})
        self.assertIn("evidence_hash", saved)

    def test_snapshot_stored(self):
        server.append_snapshot({"agent_id": "A", "deployment_id": "D",
                                "trigger": "first_production_deployment"})
        self.assertEqual(len(server.query_snapshots("A")), 1)


if __name__ == "__main__":
    unittest.main()
