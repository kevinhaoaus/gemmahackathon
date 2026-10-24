"""SwarmMind wire protocol (v1). See docs/PROTOCOL.md for the full spec.

Two channels:
  * UDP heartbeat (port 50000): one JSON object per datagram, broadcast + unicast, every 200 ms.
  * Peer bus (WebSocket /ws/peer): JSON envelopes, full mesh, each node pushes its own
    messages to every alive peer. No relaying, so there are no loops.

Every bus message is an Envelope:
  {"v":1, "type":"job.token", "src":"laptop", "term":7, "mid":"laptop:1234", "ts":1729.5, "body":{...}}

`mid` (src:counter) dedupes at the transport layer; app layers dedupe by their own keys
(job_id, seq), (task_id), (job_id for reports).
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

PROTOCOL_VERSION = 1

# ---- message types (namespaced "<domain>.<verb>"; new domains = new use cases, e.g. "arm.*")
HEARTBEAT = "hb"                  # UDP only
SYNC_REQ = "sync.req"             # ask a peer for its full state (jobs, reports, tasks, plans)
SYNC_RESP = "sync.resp"
JOB_NEW = "job.new"               # leader -> all: question, vision context, model, image
JOB_TOKEN = "job.token"           # leader -> all: {job_id, seq, token, node_id, term}
JOB_HANDOFF = "job.handoff"       # new leader -> all: {job_id, from_node, to_node, at_seq, gap_ms}
JOB_DONE = "job.done"             # leader -> all: {job_id, final_seq, ...}
REPORT_NEW = "report.new"         # leader -> all: completed-job report (store-and-forward)
REPORT_SENT = "report.sent"       # leader -> all: {job_ids:[...]} flushed to uplink
TASK_SUBMIT = "task.submit"       # any -> leader: generic capability task (robot arm etc.)
TASK_ASSIGN = "task.assign"       # leader -> all (addressed to `assignee`)
TASK_PROGRESS = "task.progress"   # assignee -> all
TASK_RESULT = "task.result"       # assignee -> all
TASK_CANCEL = "task.cancel"       # leader -> all: stop a task now (e.g. arm e-stop)
TOPIC_PUB = "topic.pub"           # any -> all: latest-value stream sample (IMU, telemetry, teleop)
PLAN_NEW = "plan.new"             # leader -> all: robot-cell plan created (full plan state)
PLAN_UPDATE = "plan.update"       # leader -> all: plan state changed (full plan state, newer rev)
EVENT = "event"                   # free-form log line for dashboards

# message types that are only valid from the current leader at the current term
LEADER_ONLY = {JOB_NEW, JOB_TOKEN, JOB_HANDOFF, JOB_DONE, REPORT_NEW, REPORT_SENT, TASK_ASSIGN,
               TASK_CANCEL, PLAN_NEW, PLAN_UPDATE}

_counter = itertools.count(1)


@dataclass
class Envelope:
    type: str
    src: str
    term: int
    body: dict[str, Any] = field(default_factory=dict)
    mid: str = ""
    ts: float = 0.0
    v: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if not self.mid:
            self.mid = f"{self.src}:{time.time_ns()}:{next(_counter)}"
        if not self.ts:
            self.ts = time.time()

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "Envelope":
        if not isinstance(obj, dict) or "type" not in obj or "src" not in obj:
            raise ValueError("not an envelope")
        return cls(type=str(obj["type"]), src=str(obj["src"]), term=int(obj.get("term", 0)),
                   body=dict(obj.get("body") or {}), mid=str(obj.get("mid", "")),
                   ts=float(obj.get("ts", 0.0)), v=int(obj.get("v", PROTOCOL_VERSION)))

    @classmethod
    def from_json(cls, data: str | bytes) -> "Envelope":
        return cls.from_obj(json.loads(data))


@dataclass
class Heartbeat:
    node_id: str
    rank: int
    ip: str
    http_port: int
    model: str
    vision_mode: str
    healthy: bool
    leader_id: str | None
    term: int
    active_job_id: str | None
    seq: int
    capabilities: list[str] = field(default_factory=list)
    tps: float = 0.0              # tokens/sec of the last/current stream on this node
    boot_id: str = ""             # changes on every process start (detects restarts)
    v: int = PROTOCOL_VERSION
    type: str = HEARTBEAT

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "Heartbeat":
        obj = json.loads(data)
        if obj.get("type") != HEARTBEAT:
            raise ValueError("not a heartbeat")
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in obj.items() if k in fields})


# Default system prompt for operator questions (/ask) at the robot cell.
SYSTEM_PROMPT = ("You are the planning assistant for a robotic assembly cell where a large robot arm "
                 "assembles an SO-101 arm from parts located by AprilTags. Describe what you see in the "
                 "cell, flag hazards to people or equipment, and give one concrete next step. "
                 "You support the human operator; you do not replace their decisions.")

# Planner job (kind "plan"): Gemma decides WHAT, geometry decides WHERE (docs/ROBOT_CELL.md).
PLAN_SYSTEM_PROMPT = (
    "You are the step planner for a robotic assembly cell. Turn the operator's goal into a plan "
    "for the robot arm. Output ONLY one JSON object, with no prose and no code fences, in exactly "
    "this schema: "
    '{"steps": [{"op": "pick", "object": "<name>"}, {"op": "place", "target": "<name>"}, '
    '{"op": "verify", "check": "<what to confirm in the camera image>"}, {"op": "home"}]}. '
    "Rules: use only the ops pick, place, verify and home. Use only the object names from the "
    "list of available objects, spelled exactly. Never output coordinates, positions, angles or "
    "any numbers: the cell computes all geometry itself. Pair every pick with a place: after a "
    "pick, the next pick or place op must be a place. Add a verify step after each place. "
    "Use at most 12 steps.")

# Verify job (kind "verify"): a yes/no check of one condition from the camera image.
VERIFY_SYSTEM_PROMPT = (
    "You are the visual inspector for a robotic assembly cell. Look at the camera image and "
    "judge only the given check. Answer ONLY with one JSON object, no prose and no code fences: "
    '{"ok": true or false, "reason": "<one short sentence>"}.')
