# KAREN Skills Registry

Version: 1.0.0 | Mode: Read-only

## Observation skills
1. **Agent lifecycle:** registration, activation, version changes, deployment, shutdown.
2. **Execution tracing:** workflows, tasks, tool calls, model calls, authorization events, failures, recovery, completion.
3. **Vital snapshots:** capture initial deployment, first success, first tool call, first inter-agent handoff, memory write, denial, failure, recovery, configuration change, and probation completion.
4. **Visual evidence:** inspect metadata for explicitly authorized application-scoped screenshots/recordings; never unrestricted desktop surveillance.
5. **Deployment memory:** search agent profiles, event history, snapshots, incidents, and baselines.
6. **Behavior evaluation:** compare outcomes against criteria fixed before probation.
7. **Regression detection:** compare versions using measured, reproducible metrics.
8. **Probation reporting:** produce evidence-linked recommendations for human review.

## Tool contracts
All query tools execute with the KAREN read-only credential:
- `karen.agents.list` — list observed agent identities.
- `karen.events.query` — retrieve filtered event records.
- `karen.events.verify` — verify an agent/deployment hash chain.
- `karen.snapshots.list` — list vital snapshots.
- `karen.snapshot.inspect` — inspect one snapshot.
- `karen.metrics.summary` — aggregate recorded event counts.
- `karen.probation.assess` — generate an evidence-linked assessment.

The separate ingestion service exposes `POST /v1/events` and `POST /v1/snapshots` to authenticated telemetry publishers only. These are not KAREN agent tools.

## Safety
No shell, arbitrary filesystem, Agent Hub write, agent-control, permission-management, or evidence-deletion tools are registered. Visual capture is a separate opt-in capability and is not enabled by this MVP.
