"""Robot-cell planner (swarmmind/cell/planner.py).

Unit tests (no cluster): plan JSON extraction + validation, verify parsing, and the leader-side
state machine on a fake node (happy path, re-plan after invalid output, verify-failure redo,
resume after failover without re-running finished steps or resubmitting a running task).

The assembly-goal endpoint (POST /cell/goal) and its simulator end-to-end tests are removed for
now; LabMate (tests/test_labmate.py) is the only user of this planner.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from swarmmind.cell import planner as P  # noqa: E402
from swarmmind.cell.planner import (PlanError, PlanManager, extract_json, parse_plan,  # noqa: E402
                                    parse_verify, validate_plan)
from swarmmind.protocol import PLAN_NEW, PLAN_SYSTEM_PROMPT, PLAN_UPDATE, VERIFY_SYSTEM_PROMPT  # noqa: E402
from swarmmind.tasks import Task  # noqa: E402

OBJECTS = {
    "servo_1": {"tag": 10, "kind": "part", "offset": [0.0, 0.0, 0.015], "grasp_yaw": 0},
    "so101_slot_1": {"tag": 20, "kind": "slot", "offset": [0.07, 0.0, 0.03], "insert_depth": 0.02},
    "tray_a": {"tag": 11, "kind": "fixture", "offset": [0.0, 0.0, 0.0]},
}
CELL = {"objects": OBJECTS,
        "arms": {"arm.big": {"base": [0.30, -0.25, 90.0, 0.0], "approach_h": 0.08, "speed": 0.05}}}
GOOD = {"steps": [{"op": "pick", "object": "servo_1"}, {"op": "place", "target": "so101_slot_1"},
                  {"op": "verify", "check": "servo seated in slot"}]}


# ============================================================================ parsing / validation
def test_parse_good_plan():
    steps = parse_plan(json.dumps(GOOD), OBJECTS)
    assert steps == GOOD["steps"]


def test_parse_strips_code_fences_and_prose():
    text = "Sure! Here is the plan {as requested}:\n```json\n" + json.dumps(GOOD, indent=2) + "\n```\nDone."
    assert parse_plan(text, OBJECTS) == GOOD["steps"]
    assert extract_json('noise {"a": "}{"} trailing {') == {"a": "}{"}


def test_rejects_unknown_names():
    bad = {"steps": [{"op": "pick", "object": "servo_9"}, {"op": "place", "target": "so101_slot_1"}]}
    with pytest.raises(PlanError, match="unknown object 'servo_9'"):
        parse_plan(json.dumps(bad), OBJECTS)
    bad2 = {"steps": [{"op": "pick", "object": "servo_1"}, {"op": "place", "target": "the slot"}]}
    with pytest.raises(PlanError, match="unknown object"):
        parse_plan(json.dumps(bad2), OBJECTS)


def test_rejects_coordinates():
    for step in ({"op": "pick", "object": "servo_1", "x": 0.2, "y": 0.1},
                 {"op": "pick", "object": "servo_1", "position": [0.2, 0.1, 0.0]},
                 {"op": "pick", "object": "servo_1", "note": [1, 2]}):
        plan = {"steps": [step, {"op": "place", "target": "so101_slot_1"}]}
        with pytest.raises(PlanError, match="coordinates"):
            parse_plan(json.dumps(plan), OBJECTS)


def test_rejects_missing_or_unpaired_place():
    with pytest.raises(PlanError, match="no matching place"):
        validate_plan({"steps": [{"op": "pick", "object": "servo_1"}, {"op": "verify", "check": "held"}]}, OBJECTS)
    with pytest.raises(PlanError, match="without a preceding pick"):
        validate_plan({"steps": [{"op": "place", "target": "so101_slot_1"}]}, OBJECTS)
    with pytest.raises(PlanError, match="still holding"):
        validate_plan({"steps": [{"op": "pick", "object": "servo_1"}, {"op": "pick", "object": "tray_a"},
                                 {"op": "place", "target": "so101_slot_1"}]}, OBJECTS)


def test_rejects_bad_ops_and_shapes():
    with pytest.raises(PlanError, match="not one of"):
        validate_plan({"steps": [{"op": "weld", "object": "servo_1"}]}, OBJECTS)
    with pytest.raises(PlanError, match="at most 12"):
        validate_plan({"steps": [{"op": "home"}] * 13}, OBJECTS)
    with pytest.raises(PlanError, match="slot and cannot be picked"):
        validate_plan({"steps": [{"op": "pick", "object": "so101_slot_1"},
                                 {"op": "place", "target": "tray_a"}]}, OBJECTS)
    with pytest.raises(PlanError):
        parse_plan("I cannot help with that.", OBJECTS)
    with pytest.raises(PlanError, match="not closed"):
        parse_plan('{"steps": [{"op": "pick", "object": "servo_1"}', OBJECTS)


def test_normalises_lenient_forms():
    plan = {"steps": [{"op": "PICK", "object": "Servo_1", "reason": "it is loose"},
                      {"op": "place", "object": "so101_slot_1"}, {"op": "home"}]}
    assert validate_plan(plan, OBJECTS) == [{"op": "pick", "object": "servo_1"},
                                            {"op": "place", "target": "so101_slot_1"}, {"op": "home"}]


def test_parse_verify():
    assert parse_verify('{"ok": true, "reason": "servo appears seated"}') == (True, "servo appears seated")
    assert parse_verify('```json\n{"ok": false, "reason": "tilted"}\n```') == (False, "tilted")
    ok, why = parse_verify("looks fine to me")
    assert ok is False and "unparseable" in why
    assert parse_verify('{"ok": "maybe"}')[0] is False


def test_mock_llm_plan_and_verify_answers():
    from swarmmind.mock_llm import words_for
    user = ("Available objects (use these names exactly; nothing else exists):\n"
            "- servo_7: part, visible\n- slot_b: slot, visible\n\nGoal: insert it")
    plan = " ".join(words_for(PLAN_SYSTEM_PROMPT, user, True))
    objs = {"servo_7": {"kind": "part"}, "slot_b": {"kind": "slot"}}
    assert [s["op"] for s in parse_plan(plan, objs)] == ["pick", "place", "verify"]
    assert "servo_7" in plan and "slot_b" in plan
    assert parse_verify(" ".join(words_for(VERIFY_SYSTEM_PROMPT, "Check: x", True)))[0] is True
    assert "Reviewing" in " ".join(words_for("other prompt", "what do you see", True))


# ============================================================================ fake node harness
class FakeJob:
    def __init__(self, job_id: str, kind: str, question: str, system: str, context: str | None, text: str):
        self.job_id, self.kind, self.question, self.system, self.context = job_id, kind, question, system, context
        self._text = text
        self.state = "streaming"

    def text(self) -> str:
        return self._text


class FakeJobs:
    """Completes each job after `delay` with the next canned answer for its kind."""

    def __init__(self, answers: dict[str, list[str]], delay: float = 0.05):
        self.answers = {k: list(v) for k, v in answers.items()}
        self.delay = delay
        self.jobs: dict[str, FakeJob] = {}
        self.streaming: dict[str, asyncio.Task] = {}
        self.created: list[FakeJob] = []

    async def create(self, question, kind="ask", system=None, image=None, context=None, job_id=None):
        if job_id in self.jobs:
            return self.jobs[job_id]
        q = self.answers[kind]
        job = FakeJob(job_id, kind, question, system, context, q.pop(0) if len(q) > 1 else q[0])
        self.jobs[job_id] = job
        self.created.append(job)
        self.streaming[job_id] = asyncio.create_task(self._finish(job))
        return job

    async def _finish(self, job: FakeJob) -> None:
        await asyncio.sleep(self.delay)
        job.state = "done"
        self.streaming.pop(job.job_id, None)

    def resume_unfinished(self) -> None:
        for job in self.jobs.values():
            if job.state != "done" and job.job_id not in self.streaming:
                self.streaming[job.job_id] = asyncio.create_task(self._finish(job))


class FakeTasks:
    def __init__(self, delay: float = 0.05, auto: bool = True):
        self.tasks: dict[str, Task] = {}
        self.submitted: list[str] = []
        self.delay, self.auto = delay, auto

    def submit(self, capability, args, task_id=None):
        if task_id in self.tasks:
            return self.tasks[task_id]
        t = Task(task_id, capability, dict(args), state="assigned", assignee="armhost", attempt=1,
                 created_ts=time.time(), updated_ts=time.time())
        self.tasks[task_id] = t
        self.submitted.append(task_id)
        if self.auto:
            asyncio.get_running_loop().call_later(self.delay, self.complete, task_id)
        return t

    def complete(self, task_id, ok=True):
        t = self.tasks[task_id]
        if not t.finished:
            t.state = "done" if ok else "failed"
            t.result = {"ok": True, "action": t.args.get("action")} if ok else None
            t.error = None if ok else "gripper slipped"

    def cancel(self, task_id):
        t = self.tasks.get(task_id)
        if not t or t.finished:
            return False
        t.state = "cancelled"
        return True


STATE = {"frame_ts": 1.0, "camera": "cam", "calibrated": True, "tags": [], "reproj_err_px": 0.5,
         "objects": [{"name": "servo_1", "kind": "part", "tag": 10, "visible": True,
                      "table": [0.20, 0.15, 0.015], "yaw_deg": 0.0},
                     {"name": "so101_slot_1", "kind": "slot", "tag": 20, "visible": True,
                      "table": [0.44, 0.24, 0.03], "yaw_deg": 25.0},
                     {"name": "tray_a", "kind": "fixture", "tag": 11, "visible": False,
                      "table": None, "yaw_deg": 0.0}]}


class FakeNode:
    def __init__(self, jobs: FakeJobs, tasks: FakeTasks, node_id: str = "lead", term: int = 3):
        self.cfg = SimpleNamespace(node_id=node_id, capabilities=[], capability_config={"cell": CELL})
        self.election = SimpleNamespace(is_leader=True, term=term, alive_peers=lambda: {
            "armhost": SimpleNamespace(hb=SimpleNamespace(capabilities=["arm.big"]))})
        self.topics = SimpleNamespace(latest={"cell.state": {"topic": "cell.state", "data": STATE}})
        self.jobs, self.tasks = jobs, tasks
        self.cell_state = None
        self.cell_annotated_jpeg = b"\xff\xd8annotated"
        self.sent: list[tuple[str, dict]] = []
        self.events: list[str] = []

    def capabilities(self):
        return []

    def broadcast(self, type_, body):
        self.sent.append((type_, json.loads(json.dumps(body))))

    def ui_push(self, type_, body):
        pass

    def event(self, msg, level="info", **_):
        self.events.append(msg)

    async def cell_frame(self):
        return self.cell_annotated_jpeg


async def _until(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        await asyncio.sleep(0.02)
    return False


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(P, "POLL_S", 0.01)
    monkeypatch.setattr(P, "VERIFY_FRAME_TIMEOUT_S", 0.05)   # fake camera state never advances


# ============================================================================ state machine (fake node)
@pytest.mark.asyncio
async def test_plan_runs_to_done(fast):
    jobs = FakeJobs({"plan": [json.dumps(GOOD)], "verify": ['{"ok": true, "reason": "seated"}']})
    node = FakeNode(jobs, FakeTasks())
    pm = PlanManager(node)
    plan = await pm.submit_goal("insert the servo into the SO-101")
    assert await _until(lambda: plan.finished), plan.wire()
    assert plan.state == "done" and plan.step == 3 and plan.leader == "lead"
    assert [s["state"] for s in plan.steps] == ["done"] * 3
    pid = plan.plan_id
    assert node.tasks.submitted == [f"{pid}-s0-a0", f"{pid}-s1-a0"]          # deterministic ids
    pick, place = (node.tasks.tasks[t].args for t in node.tasks.submitted)
    assert pick["action"] == "pick" and place["action"] == "place"
    assert place["insert_depth"] == 0.02 and pick["approach_h"] == 0.08
    # table (0.20, 0.15) with the arm base at (0.30, -0.25) rotated 90 deg -> arm (0.40, 0.10)
    assert (pick["x"], pick["y"], pick["z"]) == pytest.approx((0.40, 0.10, 0.015), abs=1e-6)
    pj = jobs.created[0]
    assert pj.kind == "plan" and pj.system == PLAN_SYSTEM_PROMPT
    assert "- servo_1: part, visible" in pj.context and "- tray_a: fixture, not visible" in pj.context
    assert jobs.created[1].kind == "verify" and jobs.created[1].system == VERIFY_SYSTEM_PROMPT
    types = [t for t, _ in node.sent]
    assert types[0] == PLAN_NEW and set(types[1:]) == {PLAN_UPDATE}
    revs = [b["rev"] for _, b in node.sent]
    assert revs == sorted(revs) and len(set(revs)) == len(revs)
    body = node.sent[-1][1]
    for k in ("plan_id", "goal", "steps", "state", "step", "leader"):
        assert k in body
    assert {"op", "state", "task_id", "result"} <= set(body["steps"][0]) and body["steps"][0]["object"] == "servo_1"


@pytest.mark.asyncio
async def test_invalid_plan_is_retried_with_the_error(fast):
    bad = '{"steps": [{"op": "pick", "object": "servo_1", "x": 0.2}, {"op": "place", "target": "so101_slot_1"}]}'
    jobs = FakeJobs({"plan": ["no idea", bad, json.dumps(GOOD)], "verify": ['{"ok": true, "reason": "ok"}']})
    node = FakeNode(jobs, FakeTasks())
    pm = PlanManager(node)
    plan = await pm.submit_goal("insert the servo")
    assert await _until(lambda: plan.finished)
    assert plan.state == "done" and plan.attempt == 2 and len(plan.errors) == 2
    qs = [j.question for j in jobs.created if j.kind == "plan"]
    assert "rejected" not in qs[0] and "no JSON object" in qs[1] and "coordinates" in qs[2]


@pytest.mark.asyncio
async def test_invalid_plan_fails_after_two_retries(fast):
    jobs = FakeJobs({"plan": ['{"steps": [{"op": "pick", "object": "unobtainium"}]}'], "verify": ["{}"]})
    node = FakeNode(jobs, FakeTasks())
    plan = await PlanManager(node).submit_goal("x")
    assert await _until(lambda: plan.finished)
    assert plan.state == "failed" and "no valid plan after 3 attempts" in plan.error
    assert node.tasks.submitted == []


@pytest.mark.asyncio
async def test_failed_verify_redoes_pick_and_place_once(fast):
    jobs = FakeJobs({"plan": [json.dumps(GOOD)],
                     "verify": ['{"ok": false, "reason": "servo tilted"}', '{"ok": true, "reason": "seated"}']})
    node = FakeNode(jobs, FakeTasks())
    plan = await PlanManager(node).submit_goal("insert")
    assert await _until(lambda: plan.finished)
    pid = plan.plan_id
    assert plan.state == "done"
    assert node.tasks.submitted == [f"{pid}-s0-a0", f"{pid}-s1-a0", f"{pid}-s0-a1", f"{pid}-s1-a1"]
    assert [j.job_id for j in jobs.created if j.kind == "verify"] == [f"{pid}-v2-a0", f"{pid}-v2-a1"]

    jobs2 = FakeJobs({"plan": [json.dumps(GOOD)], "verify": ['{"ok": false, "reason": "servo tilted"}']})
    node2 = FakeNode(jobs2, FakeTasks())
    plan2 = await PlanManager(node2).submit_goal("insert")
    assert await _until(lambda: plan2.finished)
    assert plan2.state == "failed" and "servo tilted" in plan2.error
    assert len(node2.tasks.submitted) == 4                         # one redo, then give up


@pytest.mark.asyncio
async def test_failed_task_is_retried_once_then_fails(fast):
    tasks = FakeTasks(auto=False)
    jobs = FakeJobs({"plan": [json.dumps(GOOD)], "verify": ['{"ok": true, "reason": "ok"}']})
    node = FakeNode(jobs, tasks)
    plan = await PlanManager(node).submit_goal("insert")
    for n in (1, 2):
        assert await _until(lambda: len(tasks.submitted) == n)
        tasks.complete(tasks.submitted[-1], ok=False)
    assert await _until(lambda: plan.finished)
    assert plan.state == "failed" and "gripper slipped" in plan.error
    assert tasks.submitted == [f"{plan.plan_id}-s0-a0", f"{plan.plan_id}-s0-a1"]


@pytest.mark.asyncio
async def test_cancel_stops_the_running_task(fast):
    tasks = FakeTasks(auto=False)
    node = FakeNode(FakeJobs({"plan": [json.dumps(GOOD)], "verify": ["{}"]}), tasks)
    pm = PlanManager(node)
    plan = await pm.submit_goal("insert")
    assert await _until(lambda: tasks.submitted)
    assert pm.cancel(plan.plan_id)
    assert plan.state == "cancelled" and tasks.tasks[tasks.submitted[0]].state == "cancelled"
    assert not pm.has_active()


@pytest.mark.asyncio
async def test_new_leader_resumes_without_rerunning_or_resubmitting(fast):
    """Replica says: step 0 done (task a0), step 1 running (task a0 still running on the arm host).
    The new leader must wait for that task, never resubmit it, and never redo step 0."""
    jobs = FakeJobs({"plan": [json.dumps(GOOD)], "verify": ['{"ok": true, "reason": "seated"}']})
    tasks = FakeTasks(auto=False)
    old = FakeNode(FakeJobs({"plan": [json.dumps(GOOD)], "verify": ["{}"]}), FakeTasks(auto=False), "dead", 3)
    opm = PlanManager(old)
    plan = await opm.submit_goal("insert")
    assert await _until(lambda: len(old.tasks.submitted) == 1)
    old.tasks.complete(old.tasks.submitted[0])
    assert await _until(lambda: len(old.tasks.submitted) == 2)
    opm.stop_drivers()                                             # the old leader dies here
    pid = plan.plan_id

    new = FakeNode(jobs, tasks, "heir", 4)
    pm = PlanManager(new)
    new.election.is_leader = False
    for type_, body in old.sent:                                   # replicated plan.new / plan.update
        (pm.on_new if type_ == PLAN_NEW else pm.on_update)(body)
    for tid in old.tasks.submitted:                                # replicated task table
        tasks.tasks[tid] = Task(**{**old.tasks.tasks[tid].wire()})
    rep = pm.plans[pid]
    assert rep.state == "running" and rep.step == 1 and rep.steps[1]["task_id"] == f"{pid}-s1-a0"

    new.election.is_leader = True
    pm.resume_all()
    await asyncio.sleep(0.2)
    assert tasks.submitted == [] and rep.state == "running" and rep.step == 1
    tasks.complete(f"{pid}-s1-a0")                                 # the arm host finishes the place
    assert await _until(lambda: rep.finished)
    assert rep.state == "done" and rep.leader == "heir" and rep.term == 4
    assert tasks.submitted == []                                   # nothing re-run, nothing resubmitted
    assert [s["task_id"] for s in rep.steps[:2]] == [f"{pid}-s0-a0", f"{pid}-s1-a0"]
    assert any("resuming" in e for e in new.events)


@pytest.mark.asyncio
async def test_resume_submits_a_step_without_a_task(fast):
    """Replica: plan valid and running at step 0, but no task was submitted before the leader died."""
    tasks = FakeTasks()
    node = FakeNode(FakeJobs({"plan": [json.dumps(GOOD)], "verify": ['{"ok": true, "reason": "ok"}']}), tasks)
    pm = PlanManager(node)
    rep = P.Plan(plan_id="abc12345", goal="insert", state="running", step=0,
                 steps=[P._new_step(s) for s in GOOD["steps"]], term=2, rev=5)
    pm.merge_sync([rep.wire()])                                    # leader: merge_sync resumes it
    assert await _until(lambda: pm.plans["abc12345"].finished)
    assert pm.plans["abc12345"].state == "done"
    assert tasks.submitted == ["abc12345-s0-a0", "abc12345-s1-a0"]


@pytest.mark.asyncio
async def test_replica_merge_keeps_newest_term_rev():
    node = FakeNode(FakeJobs({"plan": ["{}"]}), FakeTasks())
    node.election.is_leader = False
    pm = PlanManager(node)
    base = P.Plan(plan_id="p1", goal="g", state="running", step=1, term=5, rev=7)
    pm.on_new(base.wire())
    pm.on_update({**base.wire(), "rev": 6, "step": 0})              # stale: ignored
    assert pm.plans["p1"].step == 1
    pm.merge_sync([{**base.wire(), "term": 6, "rev": 1, "step": 2}])  # newer term wins
    assert pm.plans["p1"].step == 2 and pm.has_active()


