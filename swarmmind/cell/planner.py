"""Robot-cell planner: goal -> Gemma JSON plan -> validated arm tasks, replicated, resumes on failover.

Design rule (docs/ROBOT_CELL.md): Gemma decides WHAT (object names from the config whitelist, step
order, yes/no verification from the camera image); geometry decides WHERE (AprilTag positions from
the latest `cell.state`, converted with `swarmmind.cell.perception.table_to_arm`).

Plan state machine (driven by the leader only, every transition broadcast as `plan.update`):

    planning (job_id) -> validating -> running (step i) -> done | failed | cancelled
        ^                    |
        +--- invalid output, up to PLAN_RETRIES times (the error is appended to the question)

* Planning and verification are ordinary jobs (kind "plan" / "verify"), so their token streams get
  the same mid-sentence handoff as /ask answers.
* pick / place / home steps compile to `arm.<id>` tasks on the task bus. Task ids are deterministic
  per (plan_id, step, attempt), so a new leader never submits a second copy of a task: it waits for
  the one that exists (the task bus reassigns it if its assignee died).
* A failed verify re-runs the preceding pick/place pair once, then fails the plan.

Every node keeps a replica of every plan (`plan.new` / `plan.update`, plus `plans` in sync.resp);
replicas merge by (term, rev). On becoming leader a node resumes every unfinished plan.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from ..logutil import get_logger
from ..protocol import PLAN_NEW, PLAN_SYSTEM_PROMPT, PLAN_UPDATE, VERIFY_SYSTEM_PROMPT

if TYPE_CHECKING:
    from ..agent import Node

log = get_logger("planner")

OPS = ("pick", "place", "verify", "home")
MAX_STEPS = 12
PLAN_RETRIES = 2          # re-plan this many times after invalid output (3 attempts in total)
STEP_RETRIES = 1          # retry a failed arm task / verify job this many times
VERIFY_REDOS = 1          # after a failed verify, re-run the preceding pick/place this many times
TERMINAL = ("done", "failed", "cancelled")
MAX_PLANS = 30

POLL_S = 0.1
VERIFY_SETTLE_S = 0.3        # verify only trusts frames captured this long after the last move
VERIFY_FRAME_TIMEOUT_S = 3.0  # ...and waits at most this long for one
TASK_TIMEOUT_S = 180.0    # an arm task that never finishes is cancelled (e-stop) and counts as failed
TASK_LOOKUP_S = 3.0       # a step's task_id unknown locally: wait this long for sync before submitting
JOB_TIMEOUT_S = 240.0
JOB_STALL_S = 15.0        # job not done and nobody generating it for this long = generation failed
JOB_NUDGE_S = 3.0         # ... after this long, ask the job layer to resume it once
LOCATE_TIMEOUT_S = 5.0    # wait this long for perception to see an object before failing the step
ARM_WAIT_S = 10.0         # wait this long for a configured arm to be alive

# keys that would mean Gemma tried to give geometry; the plan must only name things
_COORD_KEYS = {"x", "y", "z", "yaw", "roll", "pitch", "xy", "xyz", "pos", "position", "pose",
               "coords", "coordinates", "point", "location", "offset", "approach_h", "insert_depth",
               "height", "angle", "rotation", "joints"}
_STEP_KEYS = {"pick": {"object"}, "place": {"target"}, "verify": {"check"}, "home": set()}
_NOTE_KEYS = {"note", "reason", "why", "comment", "description"}   # harmless prose, dropped
_FENCE = re.compile(r"```[a-zA-Z0-9_-]*")


class PlanError(ValueError):
    """Invalid plan output, or a step that cannot be compiled/executed."""


class _Unclosed(PlanError):
    pass


class PlanBusy(RuntimeError):
    def __init__(self, plan_id: str):
        super().__init__(f"plan {plan_id} is still active; cancel it first")
        self.plan_id = plan_id


class _LostLeadership(Exception):
    pass


class _JobFailed(Exception):
    pass


# ---------------------------------------------------------------------------- parsing / validation
def extract_json(text: str) -> Any:
    """The first balanced {...} in `text` that parses as JSON (code fences and surrounding prose
    ignored). Raises PlanError with the reason the first candidate failed."""
    s = _FENCE.sub("", text or "")
    start = s.find("{")
    if start < 0:
        raise PlanError("no JSON object in the output")
    first_err: PlanError | None = None
    while start >= 0:
        try:
            return _balanced_at(s, start)
        except _Unclosed as e:          # runs to the end: every later "{" is inside it
            raise first_err or e from None
        except PlanError as e:
            first_err = first_err or e
        start = s.find("{", start + 1)
    raise first_err


def _balanced_at(s: str, start: int) -> Any:
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except json.JSONDecodeError as e:
                    raise PlanError(f"output is not valid JSON: {e.msg}") from None
    raise _Unclosed("JSON object is not closed (output cut off?)")


def _has_number(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, dict):
        return any(_has_number(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return any(_has_number(x) for x in v)
    return False


def _resolve_name(name: Any, objects: dict[str, dict], i: int, role: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise PlanError(f"step {i}: {role} must be an object name")
    n = name.strip()
    if n in objects:
        return n
    folded = [k for k in objects if k.lower() == n.lower()]
    if len(folded) == 1:
        return folded[0]
    raise PlanError(f"step {i}: unknown object {n!r}; use only: {', '.join(objects)}")


def validate_plan(obj: Any, objects: dict[str, dict]) -> list[dict]:
    """Check a parsed plan against the whitelist and rules; returns normalised steps
    ([{"op": "pick", "object": n}, {"op": "place", "target": n}, {"op": "verify", "check": s},
    {"op": "home"}]). Raises PlanError with a message meant to be shown back to Gemma."""
    if not isinstance(obj, dict) or "steps" not in obj:
        raise PlanError('expected {"steps": [...]}')
    steps = obj["steps"]
    if not isinstance(steps, list) or not steps:
        raise PlanError('"steps" must be a non-empty list')
    if len(steps) > MAX_STEPS:
        raise PlanError(f"too many steps ({len(steps)}); at most {MAX_STEPS}")
    out: list[dict] = []
    holding: str | None = None
    for i, st in enumerate(steps, 1):
        if not isinstance(st, dict):
            raise PlanError(f"step {i}: each step must be an object")
        op = str(st.get("op", "")).strip().lower()
        if op not in OPS:
            raise PlanError(f"step {i}: op {st.get('op')!r} is not one of {', '.join(OPS)}")
        st = {k: v for k, v in st.items() if k != "op"}
        if op == "place" and "target" not in st and "object" in st:
            st["target"] = st.pop("object")               # tolerate {"op": "place", "object": slot}
        coords = sorted(set(map(str.lower, st)) & _COORD_KEYS)
        if coords or _has_number(st):
            raise PlanError(f"step {i}: coordinates/numbers are not allowed ({', '.join(coords) or 'numeric value'}); "
                            "name objects only, the cell computes all positions")
        extra = sorted(set(st) - _STEP_KEYS[op] - _NOTE_KEYS)
        if extra:
            raise PlanError(f"step {i}: unexpected field(s) {', '.join(extra)} for op {op}")
        if op == "pick":
            name = _resolve_name(st.get("object"), objects, i, "pick object")
            if (objects[name] or {}).get("kind") == "slot":
                raise PlanError(f"step {i}: {name} is a slot and cannot be picked")
            if holding:
                raise PlanError(f"step {i}: pick {name} while still holding {holding}; place it first")
            holding = name
            out.append({"op": "pick", "object": name})
        elif op == "place":
            name = _resolve_name(st.get("target"), objects, i, "place target")
            if not holding:
                raise PlanError(f"step {i}: place into {name} without a preceding pick")
            if name == holding:
                raise PlanError(f"step {i}: cannot place {name} onto itself")
            holding = None
            out.append({"op": "place", "target": name})
        elif op == "verify":
            check = st.get("check")
            if not isinstance(check, str) or not check.strip():
                raise PlanError(f"step {i}: verify needs a non-empty \"check\" text")
            out.append({"op": "verify", "check": check.strip()[:200]})
        else:
            out.append({"op": "home"})
    if holding:
        raise PlanError(f"pick of {holding} has no matching place")
    return out


def parse_plan(text: str, objects: dict[str, dict]) -> list[dict]:
    return validate_plan(extract_json(text), objects)


def parse_verify(text: str) -> tuple[bool, str]:
    """(ok, reason) from a verify job's answer. Anything unparseable counts as not ok."""
    try:
        obj = extract_json(text)
    except PlanError as e:
        return False, f"unparseable verify answer ({e}): {(text or '').strip()[:80]!r}"
    if not isinstance(obj, dict):
        return False, "verify answer is not a JSON object"
    ok = obj.get("ok")
    if isinstance(ok, str) and ok.strip().lower() in ("true", "false", "yes", "no"):
        ok = ok.strip().lower() in ("true", "yes")
    if not isinstance(ok, bool):
        return False, f"verify answer has no boolean ok: {json.dumps(obj)[:80]}"
    return ok, str(obj.get("reason") or "").strip()[:200]


def describe(step: dict) -> str:
    op = step.get("op")
    if op == "pick":
        return f"pick {step.get('object')}"
    if op == "place":
        return f"place {step.get('target')}"
    if op == "verify":
        return f"verify '{step.get('check')}'"
    return str(op)


def _norm_deg(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


_identity_warned = False


def _yaw_by_base(yaw_deg: float, arm_cfg: dict) -> float:
    base = list((arm_cfg or {}).get("base") or [0.0, 0.0, 0.0])
    return _norm_deg(float(yaw_deg) - (float(base[2]) if len(base) > 2 else 0.0))


def _transforms() -> tuple[Callable[[Any, dict], Any], Callable[[float, dict], float]]:
    """(table_to_arm, yaw_table_to_arm) from swarmmind.cell.perception, imported lazily. Falls
    back to an identity position transform (with a warning) only if that module does not exist;
    any other import error is a real error."""
    global _identity_warned
    try:
        from . import perception
    except ModuleNotFoundError as e:
        if e.name not in ("swarmmind.cell.perception", "cell.perception"):
            raise
        if not _identity_warned:
            _identity_warned = True
            log.warning("swarmmind.cell.perception missing: using an IDENTITY table->arm transform")
        return (lambda xyz, arm_cfg: tuple(float(v) for v in list(xyz)[:3])), _yaw_by_base
    return perception.table_to_arm, getattr(perception, "yaw_table_to_arm", _yaw_by_base)


# ---------------------------------------------------------------------------- replicated state
def _new_step(s: dict) -> dict:
    return {**s, "state": "pending", "task_id": None, "job_id": None, "result": None,
            "error": None, "attempt": 0, "retries": 0, "redo": 0, "arm": None, "args": None}


@dataclass
class Plan:
    plan_id: str
    goal: str
    state: str = "planning"                 # planning | validating | running | done | failed | cancelled
    step: int = 0                           # index of the current step while running
    steps: list[dict] = field(default_factory=list)
    job_id: str | None = None               # current planning job
    attempt: int = 0                        # planning attempt (0..PLAN_RETRIES)
    errors: list[str] = field(default_factory=list)   # validation errors of rejected plans
    error: str | None = None                # why the plan failed
    leader: str = ""                        # node that last drove it
    term: int = 0                           # leader term of the last update
    rev: int = 0                            # bumped on every update; replicas keep max (term, rev)
    created_ts: float = 0.0
    updated_ts: float = 0.0

    def wire(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_wire(cls, d: dict[str, Any]) -> "Plan":
        fields = cls.__dataclass_fields__
        p = cls(**{k: v for k, v in d.items() if k in fields})
        p.steps = [dict(s) for s in (p.steps or [])]
        p.errors = list(p.errors or [])
        return p

    @property
    def finished(self) -> bool:
        return self.state in TERMINAL

    def key(self) -> tuple[int, int]:
        return (int(self.term), int(self.rev))


class PlanManager:
    def __init__(self, node: "Node"):
        self.node = node
        self.plans: dict[str, Plan] = {}
        self.order: list[str] = []
        self.drivers: dict[str, asyncio.Task] = {}     # plan_id -> leader-side driver task

    # ------------------------------------------------------------------ config / perception
    @property
    def cell(self) -> dict:
        return dict((self.node.cfg.capability_config or {}).get("cell") or {})

    @property
    def objects(self) -> dict[str, dict]:
        return {str(k): dict(v or {}) for k, v in (self.cell.get("objects") or {}).items()}

    @property
    def arms(self) -> dict[str, dict]:
        return {str(k): dict(v or {}) for k, v in (self.cell.get("arms") or {}).items()}

    def cell_state(self) -> dict | None:
        """Latest `cell.state` (topic cache first, else what local perception set)."""
        sample = self.node.topics.latest.get("cell.state")
        data = sample.get("data") if isinstance(sample, dict) else None
        if isinstance(data, dict):
            return data
        local = getattr(self.node, "cell_state", None)
        return local if isinstance(local, dict) else None

    def _state_objects(self) -> dict[str, dict]:
        st = self.cell_state() or {}
        return {o.get("name"): o for o in st.get("objects") or [] if isinstance(o, dict) and o.get("name")}

    def objects_context(self) -> str:
        """User-turn text listing what Gemma may name (config whitelist + visibility from cell.state)."""
        seen = self._state_objects()
        lines = ["Available objects (use these names exactly; nothing else exists):"]
        for name, oc in self.objects.items():
            so = seen.get(name)
            vis = "not seen yet" if so is None else ("visible" if so.get("visible") else "not visible")
            lines.append(f"- {name}: {oc.get('kind') or (so or {}).get('kind') or 'object'}, {vis}")
        return "\n".join(lines)

    async def _frame(self) -> bytes | None:
        """Annotated cell frame (local or from the camera node); None = let the job use the raw frame."""
        fn = getattr(self.node, "cell_frame", None)
        if fn is not None:
            try:
                return await fn()
            except Exception as e:  # noqa: BLE001 - a missing frame must not stop the plan
                log.warning("cell frame unavailable: %s", e)
                return None
        return getattr(self.node, "cell_annotated_jpeg", None)

    # ------------------------------------------------------------------ queries
    def snapshot(self) -> list[dict]:
        return [self.plans[p].wire() for p in self.order[-20:]]

    def sync_payload(self) -> list[dict]:
        return [self.plans[p].wire() for p in self.order]

    def active(self) -> Plan | None:
        return next((self.plans[p] for p in self.order if not self.plans[p].finished), None)

    def has_active(self) -> bool:
        return self.active() is not None

    def _store(self, plan: Plan) -> None:
        if plan.plan_id not in self.plans:
            self.order.append(plan.plan_id)
        self.plans[plan.plan_id] = plan
        while len(self.order) > MAX_PLANS:
            old = next((p for p in self.order if self.plans[p].finished), None)
            if old is None:
                break
            self.order.remove(old)
            self.plans.pop(old, None)

    # ------------------------------------------------------------------ followers / sync
    def _merge(self, remote: Plan) -> bool:
        mine = self.plans.get(remote.plan_id)
        if mine is None:
            self._store(remote)
            return True
        if remote.key() <= mine.key():
            return False
        mine.__dict__.update(remote.__dict__)     # in place: a running driver sees the new state
        return True

    def on_new(self, body: dict) -> None:
        if self._merge(Plan.from_wire(body)):
            self.node.ui_push(PLAN_NEW, body)

    def on_update(self, body: dict) -> None:
        if self._merge(Plan.from_wire(body)):
            self.node.ui_push(PLAN_UPDATE, body)

    def merge_sync(self, plans: list[dict]) -> None:
        for d in plans:
            try:
                remote = Plan.from_wire(d)
            except (TypeError, KeyError):
                continue
            if self._merge(remote):
                self.node.ui_push(PLAN_UPDATE, self.plans[remote.plan_id].wire())
        if self.node.election.is_leader:
            self.resume_all()

    # ------------------------------------------------------------------ leader API
    async def submit_goal(self, goal: str) -> Plan:
        n = self.node
        if not n.election.is_leader:
            raise PlanError("not the leader")
        act = self.active()
        if act:
            raise PlanBusy(act.plan_id)
        now = time.time()
        plan = Plan(plan_id=uuid.uuid4().hex[:8], goal=goal, created_ts=now)
        self._store(plan)
        self._update(plan, n.election.term, f"new goal: {goal}", type_=PLAN_NEW)
        self._start(plan)
        return plan

    def resume_all(self) -> None:
        for pid in list(self.order):
            plan = self.plans[pid]
            if not plan.finished and not self._driving(pid):
                where = (f"step {plan.step + 1}/{len(plan.steps)}" if plan.state == "running"
                         else f"{plan.state} (job {plan.job_id})")
                self.node.event(f"plan {pid}: resuming on {self.node.cfg.node_id} at {where}")
                self._start(plan)

    def stop_drivers(self) -> None:
        """Step-down: stop driving (running arm tasks keep going; the next leader waits for them)."""
        for pid, t in list(self.drivers.items()):
            if not t.done():
                t.cancel()
        self.drivers.clear()

    def cancel(self, plan_id: str) -> bool:
        n = self.node
        plan = self.plans.get(plan_id)
        if plan is None or plan.finished or not n.election.is_leader:
            return False
        t = self.drivers.pop(plan_id, None)
        if t and not t.done():
            t.cancel()
        stopped = []
        for s in plan.steps:
            if s.get("state") == "running" and s.get("task_id") and n.tasks.cancel(s["task_id"]):
                stopped.append(s["task_id"])
            if s.get("state") in ("running", "pending"):
                s["state"] = "cancelled"
        plan.state = "cancelled"
        self._update(plan, n.election.term, "cancelled by operator"
                     + (f" (stopped task {', '.join(stopped)})" if stopped else ""), level="warning")
        return True

    # ------------------------------------------------------------------ driver
    def _driving(self, plan_id: str) -> bool:
        t = self.drivers.get(plan_id)
        return t is not None and not t.done()

    def _start(self, plan: Plan) -> None:
        if not self._driving(plan.plan_id):
            self.drivers[plan.plan_id] = asyncio.create_task(self._drive(plan.plan_id))

    def _guard(self, term: int) -> None:
        e = self.node.election
        if not e.is_leader or e.term != term:
            raise _LostLeadership()

    def _update(self, plan: Plan, term: int, msg: str | None = None, level: str = "info",
                type_: str = PLAN_UPDATE) -> None:
        self._guard(term)
        n = self.node
        plan.rev += 1
        plan.term = term
        plan.leader = n.cfg.node_id
        plan.updated_ts = time.time()
        body = plan.wire()
        n.broadcast(type_, body)
        n.ui_push(type_, body)
        if msg:
            n.event(f"plan {plan.plan_id}: {msg}", level=level)

    def _fail(self, plan: Plan, term: int, why: str) -> None:
        plan.state, plan.error = "failed", why
        self._update(plan, term, f"FAILED: {why}", level="error")

    async def _drive(self, plan_id: str) -> None:
        n = self.node
        term = n.election.term
        try:
            while True:
                plan = self.plans.get(plan_id)
                if plan is None or plan.finished:
                    return
                self._guard(term)
                if plan.state == "planning":
                    await self._do_planning(plan, term)
                elif plan.state == "validating":
                    self._do_validate(plan, term)
                elif plan.state == "running":
                    await self._do_step(plan, term)
                else:
                    self._fail(plan, term, f"unknown plan state {plan.state!r}")
        except _LostLeadership:
            log.info("plan %s: no longer leader at term %d; driver stops", plan_id, term)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - a planner bug must fail the plan, not hang it
            log.exception("plan %s driver crashed", plan_id)
            plan = self.plans.get(plan_id)
            if plan and not plan.finished and n.election.is_leader and n.election.term == term:
                self._fail(plan, term, f"planner error: {type(e).__name__}: {e}")
        finally:
            if self.drivers.get(plan_id) is asyncio.current_task():
                self.drivers.pop(plan_id, None)

    # ---- planning
    def _plan_question(self, plan: Plan) -> str:
        q = f"Goal: {plan.goal}"
        if plan.errors:
            q += (f"\n\nYour previous plan was rejected: {plan.errors[-1]}. "
                  "Reply again with ONLY the JSON plan and follow every rule.")
        return q

    async def _do_planning(self, plan: Plan, term: int) -> None:
        n = self.node
        jid = plan.job_id or f"{plan.plan_id}-p{plan.attempt}"
        if jid not in n.jobs.jobs:
            frame = await self._frame()
            self._guard(term)
            await n.jobs.create(self._plan_question(plan), kind="plan", system=PLAN_SYSTEM_PROMPT,
                                image=frame, context=self.objects_context(), job_id=jid)
        if plan.job_id != jid:
            plan.job_id = jid
            self._update(plan, term, f"planning with Gemma (job {jid}, attempt {plan.attempt + 1})")
        try:
            await self._wait_job(jid, term)
        except _JobFailed as e:
            self._plan_invalid(plan, term, str(e))
            return
        plan.state = "validating"
        self._update(plan, term, f"plan job {jid} done; validating")

    def _do_validate(self, plan: Plan, term: int) -> None:
        job = self.node.jobs.jobs.get(plan.job_id or "")
        if job is None or job.state != "done":
            plan.state = "planning"                   # e.g. replica got ahead of its job: wait again
            self._update(plan, term)
            return
        try:
            steps = parse_plan(job.text(), self.objects)
        except PlanError as e:
            self._plan_invalid(plan, term, str(e))
            return
        plan.steps = [_new_step(s) for s in steps]
        plan.state, plan.step = "running", 0
        self._update(plan, term, f"plan valid, {len(steps)} steps: " + " -> ".join(describe(s) for s in steps))

    def _plan_invalid(self, plan: Plan, term: int, err: str) -> None:
        plan.errors.append(err)
        if plan.attempt < PLAN_RETRIES:
            plan.attempt += 1
            plan.job_id = None
            plan.state = "planning"
            self._update(plan, term, f"invalid plan ({err}); re-planning, attempt {plan.attempt + 1}",
                         level="warning")
        else:
            self._fail(plan, term, f"no valid plan after {plan.attempt + 1} attempts: {err}")

    async def _wait_job(self, jid: str, term: int) -> str:
        n = self.node
        t0 = time.monotonic()
        stalled: float | None = None
        nudged = False
        while True:
            self._guard(term)
            job = n.jobs.jobs.get(jid)
            if job is not None and job.state == "done":
                return job.text()
            now = time.monotonic()
            gen = n.jobs.streaming.get(jid)
            if job is None or gen is None or gen.done():
                stalled = stalled or now
                if not nudged and job is not None and now - stalled > JOB_NUDGE_S:
                    nudged = True
                    n.jobs.resume_unfinished()
                if now - stalled > JOB_STALL_S:
                    raise _JobFailed(f"generation of job {jid} stalled")
            else:
                stalled = None
            if now - t0 > JOB_TIMEOUT_S:
                raise _JobFailed(f"job {jid} timed out after {JOB_TIMEOUT_S:.0f}s")
            await asyncio.sleep(POLL_S)

    # ---- steps
    async def _do_step(self, plan: Plan, term: int) -> None:
        i = plan.step
        if i >= len(plan.steps):
            plan.state = "done"
            self._update(plan, term, f"DONE: all {len(plan.steps)} steps complete")
            return
        s = plan.steps[i]
        if s.get("state") == "done":
            plan.step = i + 1
            self._update(plan, term)
            return
        if s["op"] == "verify":
            await self._run_verify(plan, i, term)
        else:
            await self._run_arm(plan, i, term)

    def _label(self, plan: Plan, i: int) -> str:
        return f"step {i + 1}/{len(plan.steps)} {describe(plan.steps[i])}"

    def _cap_alive(self, cap: str) -> bool:
        n = self.node
        if cap in n.capabilities():
            return True
        return any(cap in (ps.hb.capabilities or []) for ps in n.election.alive_peers().values())

    async def _pick_arm(self, term: int) -> str:
        arms = self.arms
        if not arms:
            raise PlanError("no arms configured in capability_config.cell.arms")
        end = time.monotonic() + ARM_WAIT_S
        while True:
            for a in arms:
                if self._cap_alive(a):
                    return a
            if time.monotonic() > end:
                raise PlanError(f"no configured arm is alive ({', '.join(arms)})")
            self._guard(term)
            await asyncio.sleep(0.2)

    async def _locate(self, name: str, term: int) -> tuple[list[float], float]:
        end = time.monotonic() + LOCATE_TIMEOUT_S
        while True:
            so = self._state_objects().get(name)
            if so and so.get("visible") and isinstance(so.get("table"), (list, tuple)) and len(so["table"]) >= 3:
                return [float(v) for v in so["table"][:3]], float(so.get("yaw_deg") or 0.0)
            if time.monotonic() > end:
                why = ("no cell.state from perception" if self.cell_state() is None
                       else "not in cell.state" if so is None else "not visible")
                raise PlanError(f"cannot locate {name}: {why}")
            self._guard(term)
            await asyncio.sleep(0.2)

    async def compile_step(self, s: dict, arm: str, term: int) -> dict:
        """Arm task args for a pick/place/home step, in the arm base frame (metres, degrees)."""
        arm_cfg = self.arms.get(arm) or {}
        if s["op"] == "home":
            return {"action": "home"}
        name = s["object"] if s["op"] == "pick" else s["target"]
        obj_cfg = self.objects.get(name) or {}
        xyz, yaw_table = await self._locate(name, term)
        to_arm, yaw_to_arm = _transforms()
        x, y, z = (float(v) for v in to_arm(xyz, arm_cfg))
        yaw = _norm_deg(float(yaw_to_arm(yaw_table + float(obj_cfg.get("grasp_yaw") or 0.0), arm_cfg)))
        args = {"action": s["op"], "x": round(x, 4), "y": round(y, 4), "z": round(z, 4),
                "yaw": round(yaw, 2), "approach_h": float(arm_cfg.get("approach_h", 0.08))}
        if s["op"] == "place":
            args["insert_depth"] = float(obj_cfg.get("insert_depth") or 0.0)
        return args

    async def _lookup_task(self, tid: str, term: int) -> Any:
        end = time.monotonic() + TASK_LOOKUP_S
        while time.monotonic() < end:
            t = self.node.tasks.tasks.get(tid)
            if t is not None:
                return t
            self._guard(term)
            await asyncio.sleep(POLL_S)
        return None

    async def _run_arm(self, plan: Plan, i: int, term: int) -> None:
        n = self.node
        s = plan.steps[i]
        tid = s.get("task_id")
        task = n.tasks.tasks.get(tid) if tid else None
        if tid and task is None:
            task = await self._lookup_task(tid, term)    # replica may still be syncing
        if task is None:
            tid = tid or f"{plan.plan_id}-s{i}-a{s.get('attempt', 0)}"
            try:
                arm = await self._pick_arm(term)
                args = await self.compile_step(s, arm, term)
                # names ride along for telemetry and the simulator (drivers ignore unknown args)
                if s["op"] == "pick":
                    args["object"] = s["object"]
                elif s["op"] == "place":
                    args["target"] = s["target"]
                    carry = next((plan.steps[j]["object"] for j in range(i - 1, -1, -1)
                                  if plan.steps[j]["op"] == "pick"), None)
                    if carry:
                        args["carry"] = carry
            except PlanError as e:
                self._step_failed(plan, i, term, str(e))
                return
            self._guard(term)
            task = n.tasks.submit(arm, args, tid)
            s.update(task_id=tid, state="running", arm=arm, args=args, error=None)
            self._update(plan, term, f"{self._label(plan, i)}: task {tid} on {arm} "
                                     f"(assignee {task.assignee or 'none yet'})")
        elif s.get("state") != "running":
            s["state"] = "running"
            self._update(plan, term, f"{self._label(plan, i)}: waiting for task {tid}")
        else:
            log.info("plan %s %s: waiting for existing task %s (%s on %s)", plan.plan_id,
                     self._label(plan, i), tid, task.state, task.assignee)
        final = await self._wait_task(tid, term)
        if plan.finished:
            return
        if final is not None and final.state == "done":
            s.update(state="done", result=final.result, error=None)
            plan.step = i + 1
            self._update(plan, term, f"{self._label(plan, i)} done on {final.assignee}")
        else:
            why = "task vanished" if final is None else (final.error or final.state)
            self._step_failed(plan, i, term, f"task {tid} {getattr(final, 'state', 'missing')}: {why}")

    async def _wait_task(self, tid: str, term: int) -> Any:
        n = self.node
        end = time.monotonic() + TASK_TIMEOUT_S
        while True:
            self._guard(term)
            t = n.tasks.tasks.get(tid)
            if t is not None and t.finished:
                return t
            if time.monotonic() > end:
                n.tasks.cancel(tid)
                t = n.tasks.tasks.get(tid)
                if t is not None:
                    t.error = f"timed out after {TASK_TIMEOUT_S:.0f}s"
                return t
            await asyncio.sleep(POLL_S)

    def _step_failed(self, plan: Plan, i: int, term: int, err: str) -> None:
        s = plan.steps[i]
        if s.get("retries", 0) < STEP_RETRIES:
            s.update(retries=s.get("retries", 0) + 1, attempt=s.get("attempt", 0) + 1,
                     task_id=None, job_id=None, state="pending", error=err)
            self._update(plan, term, f"{self._label(plan, i)} failed ({err}); retrying", level="warning")
        else:
            s.update(state="failed", error=err)
            self._fail(plan, term, f"{self._label(plan, i)} failed: {err}")

    @staticmethod
    def _preceding_pair(plan: Plan, i: int) -> list[int] | None:
        place = next((j for j in range(i - 1, -1, -1) if plan.steps[j]["op"] == "place"), None)
        if place is None:
            return None
        pick = next((j for j in range(place - 1, -1, -1) if plan.steps[j]["op"] == "pick"), None)
        return None if pick is None else [pick, place]

    def _cell_frame_ts(self) -> float:
        st = getattr(self.node, "cell_state", None)
        if not st:
            st = (self.node.topics.latest.get("cell.state") or {}).get("data")
        try:
            return float((st or {}).get("frame_ts") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    async def _fresh_frame(self, term: int) -> bytes | None:
        """A frame captured AFTER the arm finished (plus settle time for vibration), so verify
        never judges the scene from before the place. Falls back to the newest frame on timeout."""
        if self._cell_frame_ts() <= 0:      # no cell camera reporting at all: nothing to wait for
            return await self._frame()
        timeout_s = VERIFY_FRAME_TIMEOUT_S
        after = time.time() + VERIFY_SETTLE_S
        deadline = time.monotonic() + timeout_s
        while self._cell_frame_ts() < after and time.monotonic() < deadline:
            self._guard(term)
            await asyncio.sleep(0.1)
        if self._cell_frame_ts() < after:
            log.warning("verify: no camera frame newer than the last move after %.1f s; using newest", timeout_s)
        return await self._frame()

    async def _run_verify(self, plan: Plan, i: int, term: int) -> None:
        n = self.node
        s = plan.steps[i]
        jid = s.get("job_id") or f"{plan.plan_id}-v{i}-a{s.get('attempt', 0)}"
        if jid not in n.jobs.jobs:
            frame = await self._fresh_frame(term)
            self._guard(term)
            await n.jobs.create(f"Check: {s['check']}", kind="verify", system=VERIFY_SYSTEM_PROMPT,
                                image=frame, job_id=jid)
        if s.get("job_id") != jid or s.get("state") != "running":
            s.update(job_id=jid, state="running")
            self._update(plan, term, f"{self._label(plan, i)} (job {jid})")
        try:
            text = await self._wait_job(jid, term)
        except _JobFailed as e:
            self._step_failed(plan, i, term, str(e))
            return
        if plan.finished:
            return
        ok, reason = parse_verify(text)
        if ok:
            s.update(state="done", result={"ok": True, "reason": reason}, error=None)
            plan.step = i + 1
            self._update(plan, term, f"{self._label(plan, i)}: ok ({reason})")
            return
        pair = self._preceding_pair(plan, i)
        if pair and s.get("redo", 0) < VERIFY_REDOS:
            for j in pair:
                st = plan.steps[j]
                st.update(state="pending", task_id=None, attempt=st.get("attempt", 0) + 1, error=None,
                          result={"previous": st.get("result")})
            s.update(state="pending", job_id=None, attempt=s.get("attempt", 0) + 1,
                     redo=s.get("redo", 0) + 1, result={"ok": False, "reason": reason})
            plan.step = pair[0]
            self._update(plan, term, f"{self._label(plan, i)} NOT ok ({reason}); redoing "
                                     + " -> ".join(describe(plan.steps[j]) for j in pair), level="warning")
            return
        s.update(state="failed", result={"ok": False, "reason": reason}, error=reason)
        self._fail(plan, term, f"{self._label(plan, i)} not ok: {reason}")
