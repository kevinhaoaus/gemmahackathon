"""LabMate brain (swarmmind/cell/labmate.py, docs/LABMATE.md).

Unit tests (no cluster): catalog loading/validation, label parsing + validation, two-pass
agreement, hazard ranking, use-case scoring / missing, answer parsing, the mock LLM's LabMate
answers, and the leader-side state machine on a fake node: inspect -> confirm -> move_object ->
verify -> question -> answer -> report; cancel during the countdown; fetch (found, not found,
stale labels); resume on a new leader (in-flight move waited on, countdown restarted,
awaiting_answer survives).

End-to-end (3-node simulator): inspect -> the open superglue goes to the safe tray -> use case ->
answer -> report; fetch screwdriver -> handover; leader killed during the move_object task.
Skipped until the blobs perception (cell.blobs) and the arm.cell move_object capability land.
"""
from __future__ import annotations

import asyncio
import glob
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from swarmmind.cell import labmate as LM  # noqa: E402
from swarmmind.cell import planner as P  # noqa: E402
from swarmmind.cell.labmate import (CatalogError, LabError, LabMate, agree, item_spec,  # noqa: E402
                                    load_catalog, parse_answer, parse_label, rank_hazards,
                                    score_use_cases)
from swarmmind.cell.planner import PlanManager  # noqa: E402
from swarmmind.config import load_config  # noqa: E402
from swarmmind.mock_llm import words_for  # noqa: E402
from swarmmind.protocol import (ANSWER_SYSTEM_PROMPT, LABEL_SYSTEM_PROMPT, PLAN_NEW,  # noqa: E402
                                REPORT_SYSTEM_PROMPT)
from swarmmind.tasks import Task  # noqa: E402

CELL = load_config(ROOT / "configs" / "sim_c.yaml").capability_config["cell"]
CAT = load_catalog(CELL)
BOUNDS = LM.table_bounds(CELL)
TRAY_XY = [0.57, 0.30]
HANDOVER_XY = [0.32, 0.05]


def blob(i, x, y, L=0.03, W=0.03, yaw=0.0):
    return {"id": i, "table_xy": [x, y], "yaw_deg": yaw, "length_m": L, "width_m": W,
            "area_m2": L * W, "stable_frames": 10}


BLOBS = [blob(3, 0.53, 0.20, 0.028, 0.028), blob(5, 0.46, 0.075, 0.18, 0.026, 8.0),
         blob(7, 0.33, 0.355, 0.20, 0.022, -4.0), blob(9, 0.095, 0.105, 0.05, 0.016, 60.0)]
TRUTH = {"3": "superglue:open", "5": "screwdriver", "7": "file", "9": "spring"}
MARKS = [b["id"] for b in BLOBS]


def label_json(items, hazards=(), use_case="unknown"):
    return json.dumps({"items": [{"mark": m, "item": n, "state": s} for m, n, s in items],
                       "hazards": [{"type": t, "marks": ms} for t, ms in hazards], "use_case": use_case})


GOOD_LABEL = label_json([(3, "superglue", "open"), (5, "screwdriver", ""), (7, "file", ""), (9, "spring", "")],
                        [("open_container", [3])], "deburring")


# ============================================================================ catalog
def test_catalog_identical_on_every_config():
    fps = {}
    for f in sorted(glob.glob(str(ROOT / "configs" / "*.yaml"))):
        cfg = load_config(f)
        cell = cfg.capability_config.get("cell")
        assert cell, f"{f} has no cell block"
        fps[Path(f).name] = load_catalog(cell).fingerprint()
        objs = cell["objects"]
        assert {"safe_tray", "handover"} <= set(objs), f
    assert len(set(fps.values())) == 1, fps
    assert set(CAT.hazards) == {"open_container", "item_near_edge", "sharp_tool_loose", "clutter_in_path",
                                "cable_near_motion", "hot_tool", "spill"}
    legion = load_config(ROOT / "configs" / "laptop.yaml")
    assert {"vision.cell", "arm.cell"} <= set(legion.capabilities)
    assert legion.capability_config["arm.cell"]["fallback_to_mock"] is False
    assert legion.capability_config["cell"]["objects"]["safe_tray"]["tag"] == 10
    sim_b = load_config(ROOT / "configs" / "sim_b.yaml")
    assert "arm.cell" in sim_b.capabilities and sim_b.capability_config["arm.cell"]["driver"] == "mock"


def test_catalog_validation_errors():
    with pytest.raises(CatalogError, match="missing"):
        load_catalog({})
    bad = json.loads(json.dumps(CELL["catalog"]))
    bad["items"]["glue gun"] = {"kind": "weapon", "height": "tall", "grasp": "pinch"}
    bad["hazards"]["spill"]["severity"] = 9
    bad["use_cases"]["x"] = {"title": "", "needs": ["unobtainium"]}
    with pytest.raises(CatalogError) as e:
        load_catalog({"catalog": bad})
    msg = str(e.value)
    for frag in ("glue gun", "kind 'weapon'", "grasp 'pinch'", "height must be a number", "severity",
                 "unknown item(s) unobtainium", "title"):
        assert frag in msg, frag


# ============================================================================ label parsing
def test_parse_label_good_and_lenient():
    p = parse_label("Here you go:\n```json\n" + GOOD_LABEL + "\n```", MARKS, CAT)
    assert [(i["mark"], i["item"], i["state"]) for i in p["items"]] == \
        [(3, "superglue", "open"), (5, "screwdriver", ""), (7, "file", ""), (9, "spring", "")]
    assert p["hazards"] == [{"type": "open_container", "marks": [3]}] and p["use_case"] == "deburring"
    lenient = json.dumps({"items": [{"mark": "#3", "item": "Super Glue", "state": "uncapped"},
                                    {"mark": 5, "item": "hammer"}, {"mark": 42, "item": "file"},
                                    {"mark": 7, "item": "file"}, {"mark": 7, "item": "ruler"}],
                          "hazards": [{"type": "Open Container", "marks": [3, 42]}, {"type": "lasers", "marks": [5]}],
                          "use_case": "rocket_science"})
    p = parse_label(lenient, MARKS, CAT)
    assert [(i["mark"], i["item"]) for i in p["items"]] == [(3, "superglue"), (5, "hammer"), (7, "file")]
    assert p["items"][0]["state"] == "open"
    assert p["hazards"] == [{"type": "open_container", "marks": [3]}] and p["use_case"] == "unknown"
    w = " | ".join(p["warnings"])
    assert "#42" in w and "lasers" in w and "rocket_science" in w and "twice" in w


def test_open_vocabulary_names():
    """Gemma may name things outside the catalog; only unusable names become unknown."""
    raw = json.dumps({"items": [{"mark": 3, "item": "Tape Measure"}, {"mark": 5, "item": "usb-cable"},
                                {"mark": 7, "item": "!!!"}, {"mark": 9, "item": "x" * 60}], "hazards": []})
    p = parse_label(raw, MARKS, CAT)
    assert [(i["mark"], i["item"]) for i in p["items"]] == \
        [(3, "tape_measure"), (5, "usb_cable"), (7, "unknown"), (9, "unknown")]
    # two passes that word the same thing differently still agree; the catalog name wins
    a = parse_label(label_json([(3, "measuring_tapes", ""), (5, "spring", "")]), MARKS, CAT)
    b = parse_label(label_json([(3, "tape_measuring", ""), (5, "springs", "")]), MARKS, CAT)
    assert [(i["mark"], i["item"]) for i in agree(a, b, CAT)["items"]] == [(3, "measuring_tapes"), (5, "spring")]
    assert item_spec(CAT, "scissors", {"length_m": 0.2, "width_m": 0.05})["grasp"] == "across"
    assert item_spec(CAT, "beaker", {"length_m": 0.06, "width_m": 0.06})["grasp"] == "center"
    assert item_spec(CAT, "robot_arm")["grasp"] == "none" and item_spec(CAT, "file") is CAT.items["file"]


def test_parse_label_rejects_coordinates_and_garbage():
    for e in ({"mark": 3, "item": "superglue", "x": 0.2}, {"mark": 3, "item": "superglue", "table_xy": [0.1, 0.2]},
              {"mark": 3, "item": "superglue", "note": [1, 2]}):
        with pytest.raises(LabError, match="coordinates"):
            parse_label(json.dumps({"items": [e], "hazards": []}), MARKS, CAT)
    with pytest.raises(LabError, match="coordinates"):
        parse_label(json.dumps({"items": [], "hazards": [{"type": "spill", "marks": [3], "position": [1, 2]}]}),
                    MARKS, CAT)
    with pytest.raises(LabError):
        parse_label("I see some tools.", MARKS, CAT)
    with pytest.raises(LabError, match="items"):
        parse_label('{"labels": []}', MARKS, CAT)
    with pytest.raises(LabError, match="mark"):
        parse_label('{"items": [{"item": "file"}]}', MARKS, CAT)


def test_two_pass_agreement():
    p1 = parse_label(GOOD_LABEL, MARKS, CAT)
    p2 = parse_label(label_json([(3, "superglue", ""), (5, "file", ""), (7, "file", ""), (9, "unknown", "")],
                                [("cable_near_motion", [7])], "deburring"), MARKS, CAT)
    a = agree(p1, p2, CAT)
    assert [(i["mark"], i["item"], i["state"]) for i in a["items"]] == [(3, "superglue", ""), (7, "file", "")]
    assert a["hazards"] == [] and a["use_case"] == "deburring"     # open seen once only: not a hazard
    a = agree(p1, parse_label(GOOD_LABEL.replace('"deburring"', '"unknown"'), MARKS, CAT), CAT)
    assert a["hazards"] == [{"type": "open_container", "marks": [3]}] and a["use_case"] == "unknown"
    # an agreed "open" state alone implies open_container
    only_state = label_json([(3, "superglue", "open")])
    assert agree(parse_label(only_state, MARKS, CAT), parse_label(only_state, MARKS, CAT), CAT)["hazards"] == \
        [{"type": "open_container", "marks": [3]}]


def test_rank_hazards_severity_then_edge():
    blobs = [blob(1, 0.30, 0.20), blob(2, 0.02, 0.20), blob(3, 0.30, 0.38), blob(4, 0.50, 0.30)]
    hz = [{"type": "open_container", "marks": [1]}, {"type": "item_near_edge", "marks": [3, 2]},
          {"type": "spill", "marks": [4]}, {"type": "cable_near_motion", "marks": [2]}]
    r = rank_hazards(hz, blobs, CAT, BOUNDS)
    assert [(h["type"], h["marks"]) for h in r] == [
        ("spill", [4]),                    # severity 4
        ("cable_near_motion", [2]),        # severity 3, 2 cm from the edge
        ("open_container", [1]),           # severity 3, 20 cm from the edge
        ("item_near_edge", [2]),           # severity 2, split per mark, nearest edge first
        ("item_near_edge", [3])]
    assert r[0]["movable"] is False and r[2]["movable"] is True and r[0]["fix"].startswith("stop and clean")
    assert r[1]["edge_m"] == pytest.approx(0.02)


def test_use_case_scoring_and_missing():
    uc = score_use_cases({"stepper_motor", "motor_driver", "power_supply"}, "unknown", CAT)
    assert uc["id"] == "stepper_test" and uc["missing"] == ["screwdriver"] and uc["confidence"] == pytest.approx(0.65)
    uc = score_use_cases({"stepper_motor", "motor_driver"}, "stepper_test", CAT)
    assert uc["id"] == "stepper_test" and set(uc["missing"]) == {"power_supply", "screwdriver"}
    assert score_use_cases({"stepper_motor"}, "unknown", CAT)["id"] == "unknown"   # 0.35 < 0.5
    uc = score_use_cases({"superglue", "screwdriver"}, "unknown", CAT)
    assert uc["id"] == "unknown" and uc["missing"] == [] and uc["confidence"] == 0.0
    # a complete stepper bench plus a file: stepper explains more of the bench
    full = {"stepper_motor", "motor_driver", "power_supply", "screwdriver", "file"}
    assert score_use_cases(full, "unknown", CAT)["id"] == "stepper_test"
    assert score_use_cases(full, "deburring", CAT)["id"] == "deburring"   # Gemma's guess breaks it
    q = LM.build_question([{"mark": 3, "item": "superglue", "state": "open"}, {"mark": 7, "item": "file", "state": ""}],
                          score_use_cases({"file", "superglue"}, "deburring", CAT))
    assert q == "I see a file and an open superglue; I think this is Filing / deburring. Is that right, and what is this run for?"


def test_parse_answer():
    a = parse_answer('{"experiment": "Filing / deburring", "confirmed": true, "phase": "first_test", "noise_ok": true}')
    assert a == {"experiment": "Filing / deburring", "confirmed": True, "phase": "first_test", "noise_ok": True}
    a = parse_answer('Sure: {"experiment": "pendulum", "confirmed": "no", "phase": "Final", "noise_ok": null}')
    assert a["confirmed"] is False and a["phase"] == "final_data" and a["noise_ok"] is None
    assert parse_answer('{"experiment": "x", "confirmed": false, "phase": "whenever"}')["phase"] == "unknown"
    for bad in ("yes it is", '{"confirmed": true}', '{"experiment": "x", "confirmed": "maybe"}'):
        with pytest.raises(LabError):
            parse_answer(bad)


def test_mock_llm_labmate_answers():
    ctx = LM.label_context(BLOBS, CAT, TRUTH)
    assert LM.SIM_TRUTH_TAG in ctx and "- #3: 2.8 x 2.8 cm, aspect 1.0" in ctx
    text = " ".join(words_for(LABEL_SYSTEM_PROMPT, ctx + "\n\nLabel every numbered mark in the image.", True))
    p = parse_label(text, MARKS, CAT)
    assert {i["mark"]: i["item"] for i in p["items"]} == {3: "superglue", 5: "screwdriver", 7: "file", 9: "spring"}
    assert p["hazards"] == [{"type": "open_container", "marks": [3]}] and p["use_case"] == "deburring"
    real = LM.label_context(BLOBS, CAT, None)                     # no sim_truth: real camera
    assert "sim_truth" not in real and "SIMULATOR" not in real
    p = parse_label(" ".join(words_for(LABEL_SYSTEM_PROMPT, real, True)), MARKS, CAT)
    assert {i["item"] for i in p["items"]} == {"unknown"} and p["use_case"] == "unknown"
    ans = " ".join(words_for(ANSWER_SYSTEM_PROMPT, "Proposed experiment: Filing / deburring (deburring)\n"
                             "LabMate asked: ...\n\nOperator's reply: yes, first test run, noise is fine", False))
    assert parse_answer(ans) == {"experiment": "Filing / deburring", "confirmed": True, "phase": "first_test",
                                 "noise_ok": True}
    rep = " ".join(words_for(REPORT_SYSTEM_PROMPT, "Hazards found (most severe first):\n- open container on #3: FIXED\n"
                             "Missing tools: none\nGive the summary.", False))
    assert "open container" in rep and "Missing tools: none" in rep


# ============================================================================ fake node harness
class FakeJob:
    def __init__(self, job_id, kind, question, system, context, text):
        self.job_id, self.kind, self.question, self.system, self.context = job_id, kind, question, system, context
        self._text = text
        self.state = "streaming"

    def text(self):
        return self._text


class FakeJobs:
    """Answers each job with the mock LLM (or `override(kind, n) -> text|None`) after `delay`."""

    def __init__(self, delay=0.02, override=None):
        self.delay, self.override = delay, override
        self.jobs: dict[str, FakeJob] = {}
        self.streaming: dict[str, asyncio.Task] = {}
        self.created: list[FakeJob] = []

    async def create(self, question, kind="ask", system=None, image=None, context=None, job_id=None):
        if job_id in self.jobs:
            return self.jobs[job_id]
        assert kind == "ask" or image is not None or kind in ("answer", "report"), "label jobs need the marked frame"
        n = sum(j.kind == kind for j in self.created)
        text = self.override(kind, n) if self.override else None
        if text is None:
            text = " ".join(words_for(system or "", "\n\n".join(p for p in (context, question) if p), bool(image)))
        job = FakeJob(job_id, kind, question, system, context, text)
        self.jobs[job_id] = job
        self.created.append(job)
        self.streaming[job_id] = asyncio.create_task(self._finish(job))
        return job

    async def _finish(self, job):
        await asyncio.sleep(self.delay)
        job.state = "done"
        self.streaming.pop(job.job_id, None)

    def resume_unfinished(self):
        for job in self.jobs.values():
            if job.state != "done" and job.job_id not in self.streaming:
                self.streaming[job.job_id] = asyncio.create_task(self._finish(job))


class FakeScene:
    """cell.state + cell.blobs; frame_ts advances like a 30 Hz camera while running."""

    def __init__(self, blobs=BLOBS, truth=TRUTH):
        self.blobs = [dict(b, table_xy=list(b["table_xy"])) for b in blobs]
        self.truth = dict(truth)
        self.ts = time.time()
        self.frozen = False          # True = the camera died: frame_ts stops advancing

    def tick(self):
        if not self.frozen:
            self.ts = time.time()

    def cell_state(self):
        objs = [{"name": "safe_tray", "kind": "fixture", "tag": 30, "visible": True, "table": TRAY_XY + [0.0],
                 "yaw_deg": 0.0},
                {"name": "handover", "kind": "fixture", "tag": 31, "visible": True,
                 "table": HANDOVER_XY + [0.0], "yaw_deg": 0.0}]
        return {"frame_ts": self.ts, "camera": "cam", "calibrated": True, "tags": [], "objects": objs}

    def blobs_state(self):
        return {"frame_ts": self.ts, "calibrated": True, "background": True,
                "blobs": [dict(b) for b in self.blobs], "sim_truth": dict(self.truth)}

    def move(self, mark, xy):
        for b in self.blobs:
            if b["id"] == mark:
                b["table_xy"] = list(xy)


class FakeTasks:
    def __init__(self, scene, delay=0.03, auto=True):
        self.scene = scene
        self.tasks: dict[str, Task] = {}
        self.submitted: list[str] = []
        self.delay, self.auto = delay, auto

    def submit(self, capability, args, task_id=None):
        if task_id in self.tasks:
            return self.tasks[task_id]
        t = Task(task_id, capability, json.loads(json.dumps(args)), state="assigned", assignee="armhost",
                 attempt=1, created_ts=time.time(), updated_ts=time.time())
        self.tasks[task_id] = t
        self.submitted.append(task_id)
        if self.auto:
            asyncio.get_running_loop().call_later(self.delay, self.complete, task_id)
        return t

    def complete(self, task_id, ok=True):
        t = self.tasks[task_id]
        if not t.finished:
            if ok:      # the simulated arm really carries the blob
                self.scene.move(t.args["blob_id"], t.args["to"]["table_xy"])
            t.state = "done" if ok else "failed"
            t.result = {"ok": True} if ok else None
            t.error = None if ok else "object not where expected"

    def cancel(self, task_id):
        t = self.tasks.get(task_id)
        if not t or t.finished:
            return False
        t.state = "cancelled"
        return True


class FakeTopics:
    def __init__(self, scene):
        self.scene = scene

    @property
    def latest(self):
        return {"cell.state": {"topic": "cell.state", "data": self.scene.cell_state()},
                "cell.blobs": {"topic": "cell.blobs", "data": self.scene.blobs_state()}}


class FakeNode:
    def __init__(self, scene, jobs=None, tasks=None, node_id="lead", term=3, confirm_s=0.05):
        cell = json.loads(json.dumps(CELL))
        cell["labmate"] = {"confirm_s": confirm_s, "pass_gap_s": 0.1, "label_max_age_s": 10.0}
        self.cfg = SimpleNamespace(node_id=node_id, capabilities=[], capability_config={"cell": cell})
        self.election = SimpleNamespace(is_leader=True, term=term, leader_id=node_id, alive_peers=lambda: {
            "armhost": SimpleNamespace(hb=SimpleNamespace(capabilities=["arm.cell"]))})
        self.scene = scene
        self.topics = FakeTopics(scene)
        self.jobs = jobs or FakeJobs()
        self.tasks = tasks or FakeTasks(scene)
        self.cell_state = None
        self.sent: list[tuple[str, dict]] = []
        self.events: list[str] = []
        self.plans = PlanManager(self)
        self.lab = LabMate(self, self.plans)

    def capabilities(self):
        return []

    def broadcast(self, type_, body):
        self.sent.append((type_, json.loads(json.dumps(body))))

    def ui_push(self, type_, body):
        pass

    def event(self, msg, level="info", **_):
        self.events.append(msg)

    async def cell_marked(self):
        return b"\xff\xd8marked"


async def _until(cond, timeout=8.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        await asyncio.sleep(0.01)
    return False


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(P, "POLL_S", 0.01)
    monkeypatch.setattr(LM, "FRESH_SETTLE_S", 0.05)
    monkeypatch.setattr(LM, "FRESH_TIMEOUT_S", 0.5)
    monkeypatch.setattr(P, "TASK_LOOKUP_S", 0.2)


async def _camera(scene, stop):
    while not stop.is_set():
        scene.tick()
        await asyncio.sleep(0.03)


@pytest_asyncio.fixture
async def scene():
    sc = FakeScene()
    stop = asyncio.Event()
    t = asyncio.create_task(_camera(sc, stop))
    yield sc
    stop.set()
    await t


# ============================================================================ state machine
@pytest.mark.asyncio
async def test_inspect_answer_report_flow(fast, scene):
    node = FakeNode(scene)
    plan = await node.lab.inspect()
    assert plan.kind == "lab" and plan.state == "labelling"
    assert await _until(lambda: plan.state == "awaiting_answer"), plan.wire()
    lab = plan.lab
    assert {(i["mark"], i["item"]) for i in lab["items"]} == {(3, "superglue"), (5, "screwdriver"), (7, "file"),
                                                              (9, "spring")}
    labels = [j for j in node.jobs.created if j.kind == "label"]
    assert len(labels) == 3 and all(j.system == LABEL_SYSTEM_PROMPT for j in labels)   # 2 passes + verify
    p0, p1 = lab["label"]["passes"]
    assert p1["frame_ts"] - p0["frame_ts"] >= 0.1                 # frames pass_gap_s apart
    assert LM.SIM_TRUTH_TAG in labels[0].context and "- #5: 18.0 x 2.6 cm, aspect 6.9" in labels[0].context

    # one movable hazard: announce -> move_object -> verify
    assert [s["op"] for s in plan.steps] == ["announce", "move", "verify"]
    assert all(s["state"] == "done" for s in plan.steps)
    (h,) = lab["hazards"]
    assert h["type"] == "open_container" and h["marks"] == [3] and h["status"] == "fixed" and h["movable"]
    pid = plan.plan_id
    assert node.tasks.submitted == [f"{pid}-s1-a0"]
    t = node.tasks.tasks[f"{pid}-s1-a0"]
    a = t.args
    assert t.capability == "arm.cell" and a["action"] == "move_object" and a["item"] == "superglue"
    assert a["blob_id"] == 3 and a["table_xy"] == [0.53, 0.2] and a["height"] == 0.06 and a["grasp"] == "center"
    assert a["to"]["table_xy"] == TRAY_XY and a["to"]["z"] == 0.06    # table frame: tray point, grip height
    assert not any(k in a for k in ("x", "y", "z", "joints"))       # never arm coordinates
    states = [b["state"] for t_, b in node.sent if b.get("plan_id") == pid]
    for st in ("labelling", "confirming", "moving", "verifying", "awaiting_answer"):
        assert st in states, st
    assert any("Moving it to the safe tray" in e for e in node.events)

    uc = lab["use_case"]
    assert uc["id"] == "deburring" and uc["title"] == "Filing / deburring" and uc["missing"] == []
    assert uc["confidence"] == 0.9
    assert lab["question"] == ("I see a file, a screwdriver, a spring and an open superglue; I think this is "
                               "Filing / deburring. Is that right, and what is this run for?")
    assert not node.plans.has_active()           # awaiting_answer does not hold the cell
    with pytest.raises(LabError, match="no LabMate question"):
        await LabMate(FakeNode(scene), PlanManager(FakeNode(scene))).answer("yes")

    await node.lab.answer("Yes, it's a first test run, noise is fine")
    assert await _until(lambda: plan.finished), plan.wire()
    assert plan.state == "done"
    assert lab["answer"]["phase"] == "first_test" and lab["answer"]["confirmed"] is True
    assert lab["answer"]["noise_ok"] is True and lab["answer"]["text"].startswith("Yes")
    rj = next(j for j in node.jobs.created if j.kind == "report")
    assert rj.system == REPORT_SYSTEM_PROMPT and "FIXED" in rj.context and "may be relaxed" in rj.context
    assert lab["report"] and "Bench check done." in lab["report"]
    aj = next(j for j in node.jobs.created if j.kind == "answer")
    assert aj.system == ANSWER_SYSTEM_PROMPT and "Filing / deburring" in aj.context
    body = node.sent[-1][1]
    for k in ("items", "hazards", "use_case", "question", "answer", "report"):
        assert k in body["lab"], k
    assert {"mark", "item", "state", "table_xy"} <= set(body["lab"]["items"][0])
    assert {"type", "marks", "severity", "movable", "fix", "status"} <= set(body["lab"]["hazards"][0])
    assert {"id", "title", "confidence", "missing"} <= set(body["lab"]["use_case"])


@pytest.mark.asyncio
async def test_disagreeing_pass_drops_labels(fast, scene):
    second = label_json([(3, "superglue", ""), (5, "file", ""), (7, "file", ""), (9, "spring", "")], [], "unknown")
    jobs = FakeJobs(override=lambda kind, n: second if kind == "label" and n == 1 else None)
    node = FakeNode(scene, jobs=jobs)
    plan = await node.lab.inspect()
    assert await _until(lambda: plan.state == "awaiting_answer"), plan.wire()
    assert {(i["mark"], i["item"]) for i in plan.lab["items"]} == {(3, "superglue"), (7, "file"), (9, "spring")}
    assert plan.lab["hazards"] == [] and plan.steps == [] and node.tasks.submitted == []   # open seen once
    assert plan.lab["use_case"]["id"] == "deburring" and plan.lab["use_case"]["confidence"] == 0.67


@pytest.mark.asyncio
async def test_bad_label_output_is_retried_with_the_error(fast, scene):
    jobs = FakeJobs(override=lambda kind, n: '{"items": [{"mark": 3, "item": "superglue", "x": 0.5}]}'
                    if kind == "label" and n == 0 else None)
    node = FakeNode(scene, jobs=jobs)
    plan = await node.lab.inspect()
    assert await _until(lambda: plan.state == "awaiting_answer"), plan.wire()
    q = [j.question for j in jobs.created if j.kind == "label"]
    assert "rejected" not in q[0] and "coordinates" in q[1]
    assert [j.job_id for j in jobs.created if j.kind == "label"][:3] == \
        [f"{plan.plan_id}-L0p0a0", f"{plan.plan_id}-L0p0a1", f"{plan.plan_id}-L0p1a0"]


@pytest.mark.asyncio
async def test_cancel_during_countdown_never_moves(fast, scene):
    node = FakeNode(scene, confirm_s=5.0)
    plan = await node.lab.inspect()
    assert await _until(lambda: plan.state == "confirming")
    assert plan.lab["hazards"][0]["status"] == "confirming"
    assert node.plans.cancel(plan.plan_id)
    await asyncio.sleep(0.1)
    assert plan.state == "cancelled" and node.tasks.submitted == []
    assert plan.lab["hazards"][0]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_report_only_and_unpickable_hazards(fast, scene):
    out = label_json([(3, "superglue", "closed"), (5, "screwdriver", ""), (7, "file", ""), (9, "spring", "")],
                     [("spill", [9]), ("cable_near_motion", [7]), ("item_near_edge", [9])], "unknown")
    scene.truth = {}
    node = FakeNode(scene, jobs=FakeJobs(override=lambda kind, n: out if kind == "label" else None))
    plan = await node.lab.inspect()
    assert await _until(lambda: plan.state == "awaiting_answer"), plan.wire()
    hz = [(h["type"], h["status"]) for h in plan.lab["hazards"]]
    assert hz == [("spill", "report_only"), ("cable_near_motion", "report_only"), ("item_near_edge", "fixed")]
    assert len(node.tasks.submitted) == 1 and node.tasks.tasks[node.tasks.submitted[0]].args["item"] == "spring"


@pytest.mark.asyncio
async def test_fetch_found_stale_and_missing(fast, scene):
    node = FakeNode(scene)
    plan = await node.lab.inspect()
    assert await _until(lambda: plan.state == "awaiting_answer")
    n_labels = sum(j.kind == "label" for j in node.jobs.created)

    f = await node.lab.fetch("Screwdriver")                       # fresh labels: no re-labelling
    assert await _until(lambda: f.finished), f.wire()
    assert f.state == "done" and sum(j.kind == "label" for j in node.jobs.created) == n_labels
    t = node.tasks.tasks[f.steps[0]["task_id"]]
    assert t.args["item"] == "screwdriver" and t.args["blob_id"] == 5 and t.args["grasp"] == "across"
    assert t.args["to"]["table_xy"] == HANDOVER_XY and t.args["purpose"] == "handover"
    assert f.lab["report"] == "Here is the screwdriver."

    g = await node.lab.fetch("ruler")                              # in the catalog, not on the bench
    assert await _until(lambda: g.finished)
    assert g.state == "failed" and g.error == "I can't see a ruler"
    with pytest.raises(LabError, match="not a usable item name"):
        await node.lab.fetch("!!!")
    with pytest.raises(LabError, match="never pick up"):
        await node.lab.fetch("power_supply")

    for p in node.lab.lab_plans():                                 # labels age past label_max_age_s
        if p.lab.get("labels_ts"):
            p.lab["labels_ts"] -= 60
    h = await node.lab.fetch("file")
    assert await _until(lambda: h.finished), h.wire()
    assert h.state == "done" and h.lab["relabelled"] is True
    assert sum(j.kind == "label" for j in node.jobs.created) == n_labels + 2      # two fresh passes
    assert plan.state == "awaiting_answer"                          # the question still waits


@pytest.mark.asyncio
async def test_fetch_with_no_labels_relabels_then_fails(fast, scene):
    scene.truth = {"5": "screwdriver"}
    node = FakeNode(scene)
    f = await node.lab.fetch("multimeter")
    assert await _until(lambda: f.finished), f.wire()
    assert f.state == "failed" and f.error == "I can't see a multimeter"
    assert sum(j.kind == "label" for j in node.jobs.created) == 2 and node.tasks.submitted == []


def _replicate(old: FakeNode, new: FakeNode) -> None:
    new.election.is_leader = False
    for type_, body in old.sent:
        (new.plans.on_new if type_ == PLAN_NEW else new.plans.on_update)(body)
    for jid, j in old.jobs.jobs.items():                          # replicated jobs (done ones)
        if j.state == "done":
            new.jobs.jobs[jid] = j
    for tid, t in old.tasks.tasks.items():
        new.tasks.tasks[tid] = Task(**t.wire())


@pytest.mark.asyncio
async def test_new_leader_waits_for_inflight_move(fast, scene):
    old = FakeNode(scene, tasks=FakeTasks(scene, auto=False), node_id="dead", term=3)
    plan = await old.lab.inspect()
    assert await _until(lambda: old.tasks.submitted)
    assert await _until(lambda: plan.state == "moving" and plan.steps[1]["task_id"])
    old.plans.stop_drivers()                                       # the leader dies mid-move
    pid = plan.plan_id

    tasks = FakeTasks(scene, auto=False)
    new = FakeNode(scene, tasks=tasks, node_id="heir", term=4)
    _replicate(old, new)
    rep = new.plans.plans[pid]
    assert rep.state == "moving" and rep.lab["hazards"][0]["status"] == "moving"
    new.election.is_leader = True
    new.plans.resume_all()
    await asyncio.sleep(0.3)
    assert tasks.submitted == [] and rep.state == "moving"         # waits, never resubmits
    tasks.complete(f"{pid}-s1-a0")                                 # the arm host finishes the move
    assert await _until(lambda: rep.state == "awaiting_answer"), rep.wire()
    assert tasks.submitted == [] and rep.leader == "heir" and rep.lab["hazards"][0]["status"] == "fixed"
    assert any("resuming" in e for e in new.events)


@pytest.mark.asyncio
async def test_confirming_restarts_countdown_and_awaiting_survives(fast, scene):
    old = FakeNode(scene, confirm_s=30.0, node_id="dead", term=3)
    plan = await old.lab.inspect()
    assert await _until(lambda: plan.state == "confirming")
    old.plans.stop_drivers()
    pid = plan.plan_id
    new = FakeNode(scene, node_id="heir", term=4, confirm_s=0.3)
    _replicate(old, new)
    rep = new.plans.plans[pid]
    assert rep.state == "confirming"
    new.election.is_leader = True
    t0 = time.monotonic()
    new.plans.resume_all()
    assert await _until(lambda: rep.state == "moving")
    assert time.monotonic() - t0 >= 0.28                           # a full countdown on the new leader
    assert await _until(lambda: rep.state == "awaiting_answer")
    assert new.tasks.submitted == [f"{pid}-s1-a0"]

    third = FakeNode(scene, node_id="third", term=5)               # awaiting_answer replicates + answers
    _replicate(new, third)
    assert third.plans.plans[pid].state == "awaiting_answer"
    third.election.is_leader = True
    third.plans.resume_all()                                       # idle: nothing to drive
    assert not third.plans.drivers
    await third.lab.answer("no, this is for final data")
    assert await _until(lambda: third.plans.plans[pid].finished)
    done = third.plans.plans[pid]
    assert done.state == "done" and done.lab["answer"]["phase"] == "final_data"
    assert done.lab["answer"]["confirmed"] is False


@pytest.mark.asyncio
async def test_failed_move_is_retried_then_reported(fast, scene):
    tasks = FakeTasks(scene, auto=False)
    node = FakeNode(scene, tasks=tasks)
    plan = await node.lab.inspect()
    for n in (1, 2):
        assert await _until(lambda: len(tasks.submitted) == n)
        tasks.complete(tasks.submitted[-1], ok=False)
    assert await _until(lambda: plan.state == "awaiting_answer"), plan.wire()
    pid = plan.plan_id
    assert tasks.submitted == [f"{pid}-s1-a0", f"{pid}-s1-a1"]
    h = plan.lab["hazards"][0]
    assert h["status"] == "move_failed" and "not where expected" in h["error"]
    assert plan.steps[2]["state"] == "skipped"


@pytest.mark.asyncio
async def test_camera_gone_after_move_is_unverified(fast, scene):
    tasks = FakeTasks(scene, auto=False)
    node = FakeNode(scene, tasks=tasks)
    plan = await node.lab.inspect()
    assert await _until(lambda: tasks.submitted)
    scene.frozen = True                                            # the camera dies mid-move
    tasks.complete(tasks.submitted[0])
    assert await _until(lambda: plan.state == "awaiting_answer"), plan.wire()
    h = plan.lab["hazards"][0]
    assert h["status"] == "unverified" and "no camera frame since the move" in h["verify"]
    assert sum(j.kind == "label" for j in node.jobs.created) == 2  # no verify job on a stale frame


# ============================================================================ end-to-end on the simulator
def _e2e_missing() -> list[str]:
    miss = [m for m in ("cv2", "swarmmind.cell.blobs", "swarmmind.capabilities.cell_vision")
            if importlib.util.find_spec(m) is None]
    caps = ROOT / "swarmmind" / "capabilities"
    if not miss:
        if "cell.blobs" not in (caps / "cell_vision.py").read_text(encoding="utf-8"):
            miss.append("vision.cell publishing cell.blobs")
        if not any('"arm.cell"' in f.read_text(encoding="utf-8") and "move_object" in f.read_text(encoding="utf-8")
                   for f in caps.glob("*.py")):
            miss.append("an arm.cell capability with move_object")
    return miss


_MISSING = _e2e_missing()
e2e = pytest.mark.skipif(bool(_MISSING), reason=f"LabMate e2e needs: {', '.join(_MISSING)} "
                                                "(the blobs and arm.cell builders' pieces; the integrator runs it)")


def _get(url, path):
    try:
        r = httpx.get(url + path, timeout=1.5)
        return r.json() if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


def _wait(cond, timeout, step=0.1):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = cond()
        if v:
            return v
        time.sleep(step)
    return None


def _plan(url, pid):
    return next((p for p in (_get(url, "/cell/plans") or {}).get("plans", []) if p["plan_id"] == pid), None)


def _tasks(url, prefix):
    return [t for t in (_get(url, "/tasks") or {}).get("tasks", []) if t["task_id"].startswith(prefix)]


def _blobs_ready(url):
    b = ((_get(url, "/topics") or {}).get("topics") or {}).get("cell.blobs")
    d = (b or {}).get("data") or {}
    return d.get("calibrated") and len(d.get("blobs") or []) >= 3 and d.get("sim_truth")


def _leader_everywhere(cluster, leader, keys=("a", "b", "c")):
    return all((_get(cluster.url(k), "/peers") or {}).get("leader_id") == leader for k in keys)


def _settled(url):
    return not any(p["state"] not in ("done", "failed", "cancelled", "awaiting_answer")
                   for p in (_get(url, "/cell/plans") or {}).get("plans", []))


@pytest.fixture(scope="module")
def cluster():
    if _MISSING:
        pytest.skip(f"LabMate e2e needs: {', '.join(_MISSING)}")
    from scripts.simulate import SimCluster, ports_in_use
    busy = ports_in_use()
    if busy:
        pytest.fail(f"sim ports in use ({', '.join(busy)}): stop the other simulator first")
    c = SimCluster(log_dir=ROOT / "data" / "sim" / "logs_test_labmate", tps_scale=3.0)
    assert c.start(wait=True, timeout=40), "sim nodes did not come up"
    yield c
    c.stop()


@e2e
def test_leader_dies_during_move_e2e(cluster):
    """Runs first: the synthetic scene lives in sim_a (camera + leader), so killing it also
    stops the camera; the revived sim_a starts a fresh scene for the next test."""
    b, cc = cluster.url("b"), cluster.url("c")
    assert _wait(lambda: _leader_everywhere(cluster, "sim_a"), 20), "sim_a never led"
    assert _wait(lambda: _blobs_ready(cc), 30), "cell.blobs with sim_truth never reached sim_c"
    r = httpx.post(b + "/lab/inspect", timeout=10)
    assert r.status_code == 200, r.text
    pid = r.json()["plan_id"]
    at_kill = _wait(lambda: (q := _plan(cc, pid)) and q["state"] == "moving"
                    and any(s.get("task_id") for s in q["steps"]) and q, 120, step=0.05)
    assert at_kill, f"plan never reached the move: {_plan(cc, pid)}"
    cluster.kill("a")
    assert _wait(lambda: (_get(b, "/peers") or {}).get("leader_id") == "sim_b", 5), "sim_b did not take over"
    p = _wait(lambda: (q := _plan(cc, pid)) and q["state"] in ("awaiting_answer", "done", "failed", "cancelled")
              and q, 120)
    assert p and p["state"] == "awaiting_answer", p
    assert p["leader"] == "sim_b"
    ts = _tasks(b, pid)
    assert len(ts) == 1, ts                                        # no duplicate move
    assert ts[0]["attempt"] == 1 and ts[0]["state"] == "done"
    assert p["lab"]["hazards"][0]["status"] in ("fixed", "unverified"), (p["lab"]["hazards"], p["steps"])


@e2e
def test_inspect_answer_fetch_e2e(cluster):
    a, b, cc = cluster.url("a"), cluster.url("b"), cluster.url("c")
    if not cluster.alive("a"):
        cluster.revive("a", wait=True)
    assert _wait(lambda: _leader_everywhere(cluster, "sim_a"), 40), "sim_a never (re)took leadership"
    assert _wait(lambda: _settled(cc), 60)
    assert _wait(lambda: _blobs_ready(cc), 30), "cell.blobs with sim_truth never reached sim_c"
    assert _wait(lambda: "arm.cell" in next((p["capabilities"] for p in (_get(a, "/peers") or {}).get("peers", [])
                                             if p["node_id"] == "sim_b"), []), 20), "sim_b never offered arm.cell"
    m = httpx.get(cc + "/cell/marked.jpg", timeout=5)               # sim_c has no camera: from sim_a
    assert m.status_code == 200 and m.content[:2] == b"\xff\xd8"

    r = httpx.post(cc + "/lab/inspect", timeout=10)                # forwarded sim_c -> sim_a
    assert r.status_code == 200, r.text
    pid = r.json()["plan_id"]
    assert r.json()["leader_id"] == "sim_a"
    p = _wait(lambda: (q := _plan(cc, pid)) and q["state"] in ("awaiting_answer", "failed", "cancelled") and q, 120)
    assert p and p["state"] == "awaiting_answer", p
    lab = p["lab"]
    glue = [h for h in lab["hazards"] if h["type"] == "open_container"]
    assert glue and glue[0]["status"] == "fixed", lab["hazards"]
    ts = _tasks(b, pid)
    assert len(ts) == 1 and ts[0]["capability"] == "arm.cell" and ts[0]["state"] == "done"
    assert ts[0]["args"]["item"] == "superglue" and ts[0]["assignee"] == "sim_b"
    assert lab["use_case"]["id"] == "deburring" and "Filing / deburring" in lab["question"]
    st = _get(b, "/lab/state")
    assert st["awaiting_answer"] == pid and st["question"] == lab["question"]

    r = httpx.post(b + "/lab/answer", json={"text": "yes, first test run, noise is fine"}, timeout=10)
    assert r.status_code == 200, r.text
    p = _wait(lambda: (q := _plan(cc, pid)) and q["state"] in ("done", "failed", "cancelled") and q, 90)
    assert p and p["state"] == "done", p
    assert p["lab"]["answer"]["phase"] == "first_test" and p["lab"]["report"]

    r = httpx.post(a + "/lab/fetch", json={"item": "screwdriver"}, timeout=10)
    assert r.status_code == 200, r.text
    fid = r.json()["plan_id"]
    f = _wait(lambda: (q := _plan(cc, fid)) and q["state"] in ("done", "failed", "cancelled") and q, 90)
    assert f and f["state"] == "done", f
    (t,) = _tasks(b, fid)
    assert t["args"]["item"] == "screwdriver" and t["args"]["purpose"] == "handover" and t["state"] == "done"
