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


# Default system prompt for operator questions (/ask) at the lab bench.
SYSTEM_PROMPT = ("You are LabMate, a research-assistant robot arm on a lab bench. Describe what you see "
                 "on the bench, flag hazards to people or equipment, and give one concrete next step. "
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

# ---- LabMate (docs/LABMATE.md): Gemma names things, code places things.
# Label job (kind "label"): numbered marks on the camera frame -> catalog names. The user turn lists
# every mark (size, aspect ratio), the catalog item names, the hazard types and the use cases.
LABEL_SYSTEM_PROMPT = (
    "You are the eyes of LabMate, a robot arm on a lab bench. The camera image shows the bench "
    "from above; every object on it has a big numbered mark like #3. Name what each mark is. "
    "Output ONLY one JSON object, with no prose and no code fences, in exactly this schema: "
    '{"items": [{"mark": 3, "item": "<item name>", "state": "open" | "closed" | ""}], '
    '"hazards": [{"type": "<hazard type>", "marks": [3]}], "use_case": "<use case id>" | "unknown"}. '
    "Rules: every mark number must be one of the listed marks. Name every object, whatever it "
    "is: use the catalog item name when one fits, spelled exactly; otherwise give your own short "
    "name of 1 to 3 lowercase words joined by underscores (scissors, tape_measure, beaker, "
    "usb_cable). Use only hazard types from the hazard list, spelled exactly. Use \"unknown\" "
    "only if you really cannot tell what a mark is. A mark may be part of the robot arm itself: "
    'name it robot_arm. state is "open" or "closed" for containers with a lid '
    '(glue, bottles), else "". Report a hazard only if you can see it. use_case is the id of the '
    'listed experiment this bench is set up for, or "unknown". Never output coordinates, '
    "positions, sizes, angles or any numbers other than mark numbers: the robot measures all "
    "geometry itself.")

# Answer job (kind "answer"): the operator's reply to LabMate's question -> structured fields.
ANSWER_SYSTEM_PROMPT = (
    "You turn a lab operator's spoken reply into structured fields. The user turn gives the "
    "experiment LabMate proposed and the operator's reply. Output ONLY one JSON object, with no "
    "prose and no code fences, with exactly these fields: "
    '{"experiment": "<short name of the experiment the operator says this is>", '
    '"confirmed": true or false, "phase": "first_test" | "final_data" | "unknown", '
    '"noise_ok": true | false | null}. '
    "confirmed is true only if the operator agrees with the proposed experiment. phase is "
    '"first_test" for a first try, trial or rough test run, "final_data" for a run whose data '
    'will be kept or published, else "unknown". noise_ok is true if the operator says noisy or '
    "rough results are acceptable, false if they need clean data, null if they did not say.")

# Report job (kind "report", plain text): the spoken summary at the end of an inspection.
REPORT_SYSTEM_PROMPT = (
    "You are LabMate, a robot lab assistant, speaking to the operator at the bench. From the "
    "facts in the user turn, give a short spoken-style summary (4 to 6 plain sentences, no "
    "lists, no markdown): the hazards you found and which ones you fixed, what still needs a "
    "human, the experiment, the tools that are missing, and notes for this run. For a "
    "first_test you may relax a design note (for example skip averaging or fine calibration) "
    "when noise is acceptable; for final_data keep every design note. Safety items never "
    "change with the phase: always repeat every unresolved hazard and its fix. Use only the "
    "facts given; do not invent items or numbers.")
