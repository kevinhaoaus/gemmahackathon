"""SO-101 capability with the mock driver: no cluster, no hardware.

Covers clamping + per-tick rate limiting, the home task (+ idempotency), sequences, cancellation
as e-stop, refusal while teleop is active, the teleop deadman timeout and the single-source latch.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from swarmmind.capabilities import so101  # noqa: E402
from swarmmind.tasks import REGISTRY  # noqa: E402

CAL = {
    "joints": {
        "shoulder_pan": {"min": 1000, "max": 3000, "home": 2048},
        "shoulder_lift": {"min": 1200, "max": 2900, "home": 2000},
        "elbow_flex": {"min": 1100, "max": 3000, "home": 2100},
        "wrist_flex": {"min": 1000, "max": 3000, "home": 2048},
        "wrist_roll": {"min": 500, "max": 3500, "home": 2048},
        "gripper": {"min": 1800, "max": 2600, "home": 2000},
    },
    "poses": {"ready": {"shoulder_lift": 1700, "elbow_flex": 2300}, "open": {"gripper": 1.0}},
}


def conf(tmp_path: Path, **over) -> dict:
    p = tmp_path / "so101_cal.json"
    p.write_text(json.dumps(CAL), encoding="utf-8")
    c = {"mock": True, "calibration": str(p), "max_step_ticks": 40, "rate_hz": 100,
         "telemetry_hz": 20, "mock_speed_ticks_s": 20000, "tolerance_ticks": 5, "settle_s": 0.3,
         "move_timeout_s": 10, "teleop": {"deadman_ms": 300, "release_s": 0.6}}
    c.update(over)
    return c


class FakeNode:
    def __init__(self, c: dict):
        self.cfg = SimpleNamespace(node_id="armhost", capabilities=["arm.so101"],
                                   capability_config={"arm.so101": c})
        self.election = SimpleNamespace(is_leader=True, term=1)
        self.published: list[tuple[str, dict]] = []
        self.events: list[tuple[str, str]] = []

    async def publish(self, topic, data):
        self.published.append((topic, data))
        return {"topic": topic, "data": data}

    def event(self, msg, level="info", **_):
        self.events.append((level, msg))

    def capabilities(self):
        return list(self.cfg.capabilities)


@pytest_asyncio.fixture
async def node(tmp_path):
    n = FakeNode(conf(tmp_path))
    await so101.setup(n)
    yield n
    await so101.shutdown()


class Progress:
    def __init__(self):
        self.items: list[dict] = []

    async def __call__(self, p):
        self.items.append(p)


def tele(src="phoneA", origin="armhost", **d):
    data = {"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "grip": None, "enabled": True,
            "deadman": True, "source": src}
    data.update(d)
    return {"topic": "teleop.so101", "origin": origin, "data": data, "seq": time.time_ns(), "ts": time.time()}


async def run(args):
    return await REGISTRY["arm.so101"](args, Progress())


# ------------------------------------------------------------------ tests
@pytest.mark.asyncio
async def test_setup_and_telemetry(node):
    ctl = so101.get_controller()
    assert ctl is not None and ctl.drv.name == "mock"
    await asyncio.sleep(0.2)
    states = [d for t, d in node.published if t == "arm.so101.state"]
    assert states, "telemetry not published"
    s = states[-1]
    assert set(so101.JOINTS) <= set(s["joints"])
    assert s["mode"] == "idle" and s["estop"] is False and 0 <= s["gripper"] <= 1


@pytest.mark.asyncio
async def test_setup_skipped_without_capability(tmp_path):
    n = FakeNode(conf(tmp_path))
    n.cfg.capabilities = []
    await so101.setup(n)
    assert so101.get_controller() is None
    with pytest.raises(RuntimeError, match="not set up"):
        await run({"action": "home", "_task_id": "x"})


@pytest.mark.asyncio
async def test_clamping_and_rate_limit(tmp_path):
    ctl = await so101.build_controller(conf(tmp_path), loops=False)
    assert ctl.to_ticks("shoulder_pan", 4000) == 3000
    assert ctl.to_ticks("shoulder_pan", -50) == 1000
    assert ctl.to_ticks("shoulder_pan", -1.0) == 1000
    assert ctl.to_ticks("shoulder_pan", 1.0) == 3000
    assert ctl.to_ticks("shoulder_pan", 0.0) == 2000
    assert ctl.to_ticks("wrist_roll", 3000, units="ticks") == 3000
    with pytest.raises(ValueError):
        ctl.to_ticks("tail", 1)
    # rate limit: one control tick moves the goal by at most max_step_ticks
    before = dict(ctl.drv.goal)
    ctl.target["shoulder_pan"] = 3000.0
    await ctl.tick()
    assert abs(ctl.drv.goal["shoulder_pan"] - before["shoulder_pan"]) <= ctl.max_step
    assert ctl.drv.goal["shoulder_pan"] > before["shoulder_pan"]


@pytest.mark.asyncio
async def test_move_clamped_to_bounds(node):
    ctl = so101.get_controller()
    r = await run({"action": "move_joints", "joints": {"shoulder_pan": 4095, "wrist_flex": 10}, "_task_id": "m1"})
    assert r["reached"] == {"shoulder_pan": 3000, "wrist_flex": 1000}
    assert abs(ctl.present["shoulder_pan"] - 3000) <= 5
    assert ctl.drv.goal["shoulder_pan"] <= 3000
    r = await run({"action": "gripper", "value": 7, "_task_id": "g1"})
    assert r["reached"] == {"gripper": 2600}


@pytest.mark.asyncio
async def test_home_task_and_idempotency(node):
    ctl = so101.get_controller()
    await run({"action": "move_joints", "joints": {"shoulder_pan": 1200, "elbow_flex": 2800}, "_task_id": "pre"})
    p = Progress()
    r = await REGISTRY["arm.so101"]({"action": "home", "_task_id": "h1"}, p)
    assert r["settled"] is True
    for j, b in CAL["joints"].items():
        assert abs(ctl.present[j] - b["home"]) <= 5, j
    assert any(x.get("step") == "home" for x in p.items)
    # re-delivery of the same task id must not move the arm again
    await run({"action": "move_joints", "joints": {"shoulder_pan": 2900}, "_task_id": "away"})
    writes = ctl.drv.writes
    r2 = await run({"action": "home", "_task_id": "h1"})
    assert r2 == r
    assert ctl.drv.writes == writes
    assert abs(ctl.present["shoulder_pan"] - 2900) <= 5


@pytest.mark.asyncio
async def test_sequence(node):
    ctl = so101.get_controller()
    p = Progress()
    r = await REGISTRY["arm.so101"]({"action": "sequence", "_task_id": "s1", "steps": [
        {"action": "pose", "name": "ready"},
        {"action": "pose", "name": "open"},
        {"action": "move_joints", "joints": {"wrist_roll": 0.5}},
        {"action": "gripper", "value": 0.0},
    ]}, p)
    assert len(r["steps"]) == 4
    assert abs(ctl.present["shoulder_lift"] - 1700) <= 5
    assert abs(ctl.present["elbow_flex"] - 2300) <= 5
    assert abs(ctl.present["wrist_roll"] - 2750) <= 5        # 500 + 0.75 * 3000
    assert abs(ctl.present["gripper"] - 1800) <= 5
    assert {x.get("seq_index") for x in p.items} >= {0, 1, 2, 3}
    # a bad step is rejected before anything moves
    writes = ctl.drv.writes
    with pytest.raises(ValueError):
        await run({"action": "sequence", "_task_id": "s2",
                   "steps": [{"action": "home"}, {"action": "fly"}]})
    with pytest.raises(ValueError, match="unknown pose"):
        await run({"action": "pose", "name": "nope", "_task_id": "s3"})
    assert ctl.drv.writes == writes


@pytest.mark.asyncio
async def test_cancel_is_estop(node):
    ctl = so101.get_controller()
    t = asyncio.create_task(run({"action": "move_joints", "joints": {"shoulder_pan": 3000},
                                 "speed": 0.05, "_task_id": "slow"}))   # 2 ticks/step -> ~5 s
    await asyncio.sleep(0.25)
    assert ctl.mode == "task"
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert ctl.estop is True and ctl.mode == "idle"
    await asyncio.sleep(0.1)
    frozen = ctl.present["shoulder_pan"]
    assert frozen < 2900
    await asyncio.sleep(0.3)
    assert abs(ctl.present["shoulder_pan"] - frozen) <= 1, "arm kept moving after cancel"
    assert ctl.target["shoulder_pan"] == ctl.cmd["shoulder_pan"]
    # not latched by default: the next command clears the e-stop
    await run({"action": "home", "_task_id": "after"})
    assert ctl.estop is False


@pytest.mark.asyncio
async def test_stop_action_aborts_motion(node):
    ctl = so101.get_controller()
    t = asyncio.create_task(run({"action": "move_joints", "joints": {"shoulder_pan": 1000},
                                 "speed": 0.05, "_task_id": "slow2"}))
    await asyncio.sleep(0.2)
    assert (await run({"action": "stop", "_task_id": "stop1"})) == {"stopped": True}
    with pytest.raises(RuntimeError, match="aborted"):
        await t
    assert ctl.mode == "idle"


@pytest.mark.asyncio
async def test_estop_latch(tmp_path):
    n = FakeNode(conf(tmp_path, estop_latch=True))
    await so101.setup(n)
    try:
        ctl = so101.get_controller()
        ctl.hold("test", estop=True)
        with pytest.raises(RuntimeError, match="latched"):
            await run({"action": "home", "_task_id": "l1"})
        await run({"action": "reset", "_task_id": "l2"})
        await run({"action": "home", "_task_id": "l3"})
        assert ctl.estop is False
    finally:
        await so101.shutdown()


@pytest.mark.asyncio
async def test_teleop_mapping_and_deadman_timeout(node):
    ctl = so101.get_controller()
    base = dict(ctl.cmd)
    await ctl.on_teleop(tele(alpha=350.0, beta=10.0, gamma=0.0))    # zero captured here
    assert ctl.mode == "teleop" and ctl.tele_active
    assert ctl.target["shoulder_pan"] == pytest.approx(base["shoulder_pan"], abs=1)   # no jump
    for _ in range(5):
        await asyncio.sleep(0.03)
        await ctl.on_teleop(tele(alpha=0.0, beta=20.0, gamma=-10.0, grip=1.0))     # alpha wraps 350->0
    k = so101.TICKS_PER_DEG
    assert ctl.target["shoulder_pan"] == pytest.approx(base["shoulder_pan"] + 10 * k, abs=1)
    assert ctl.target["shoulder_lift"] == pytest.approx(base["shoulder_lift"] + 5 * k, abs=1)
    assert ctl.target["elbow_flex"] == pytest.approx(base["elbow_flex"] - 5 * k, abs=1)
    assert ctl.target["wrist_roll"] == pytest.approx(base["wrist_roll"] - 10 * k, abs=1)
    assert ctl.target["gripper"] == 2600
    # tasks are refused while teleop holds the arm
    with pytest.raises(RuntimeError, match="teleop active"):
        await run({"action": "home", "_task_id": "t1"})
    # silence > 300 ms -> deadman: stop and hold
    await asyncio.sleep(0.4)
    assert ctl.tele_active is False
    held = dict(ctl.present)
    assert ctl.target == ctl.cmd
    await asyncio.sleep(0.15)
    assert all(abs(ctl.present[j] - held[j]) <= 1 for j in so101.JOINTS)
    # deadman released explicitly also holds
    await ctl.on_teleop(tele())
    assert ctl.tele_active
    await ctl.on_teleop(tele(alpha=30.0))
    await ctl.on_teleop(tele(alpha=30.0, deadman=False))
    assert not ctl.tele_active and ctl.target == ctl.cmd and ctl.mode == "teleop"
    # after release_s of silence the latch is dropped and tasks work again
    await asyncio.sleep(0.8)
    assert ctl.mode == "idle" and ctl.tele_src is None
    await run({"action": "home", "_task_id": "t2"})


@pytest.mark.asyncio
async def test_single_source_latch(node):
    ctl = so101.get_controller()
    await ctl.on_teleop(tele(src="A", alpha=0.0))
    assert ctl.tele_src == "armhost/A"
    tgt = dict(ctl.target)
    await ctl.on_teleop(tele(src="B", alpha=40.0))                 # other phone on the same node
    await ctl.on_teleop(tele(src="A", origin="laptop", alpha=40.0))  # same id, another node
    assert ctl.tele_src == "armhost/A" and ctl.target == tgt and ctl.tele_rejected == 2
    assert any("ignored" in m for _, m in node.events)
    # A keeps sending (enabled, deadman released) -> latch is kept past release_s
    for _ in range(8):
        await asyncio.sleep(0.1)
        await ctl.on_teleop(tele(src="A", deadman=False))
        await ctl.on_teleop(tele(src="B", alpha=40.0))
    assert ctl.tele_src == "armhost/A"
    # A goes silent for > release_s -> B may take over
    await asyncio.sleep(0.8)
    await ctl.on_teleop(tele(src="B", alpha=40.0))
    assert ctl.tele_src == "armhost/B" and ctl.tele_active


@pytest.mark.asyncio
async def test_teleop_preempts_running_task(node):
    ctl = so101.get_controller()
    t = asyncio.create_task(run({"action": "move_joints", "joints": {"shoulder_pan": 3000},
                                 "speed": 0.05, "_task_id": "slow3"}))
    await asyncio.sleep(0.2)
    await ctl.on_teleop(tele(src="P"))
    with pytest.raises(RuntimeError, match="preempted by teleop"):
        await t
    assert ctl.mode == "teleop" and ctl.tele_active
