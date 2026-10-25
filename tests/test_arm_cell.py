"""LabMate shared arm capability "arm.cell" (capabilities/arm_cell.py) on the mock driver with a fake
node that serves cell.blobs, plus the reBot B601 driver's kinematics and its motorbridge plumbing
against an in-process fake SDK. No hardware and no serial port is ever opened.
"""
from __future__ import annotations

import asyncio
import math
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import pytest_asyncio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from swarmmind import tasks  # noqa: E402
from swarmmind.capabilities import arm_cell, cell_arm  # noqa: E402
from swarmmind.cell import perception as P  # noqa: E402
from swarmmind.cell.drivers import rebot  # noqa: E402
from swarmmind.tasks import REGISTRY  # noqa: E402

WS = {"x": [0.15, 0.70], "y": [-0.45, 0.45], "z": [0.02, 0.60]}
BASE = [0.30, -0.25, 90.0, 0.0]


def conf(**over) -> dict:
    c = {"driver": "mock", "arm_id": "testarm", "base": BASE, "workspace": WS, "max_speed": 2.0,
         "speed": 1.0, "slow_speed": 0.5, "retreat_speed": 0.5, "carry_speed": 0.5, "insert_dwell_s": 0.0,
         "approach_h": 0.08, "telemetry_hz": 20, "health_interval_s": 3600, "mock_max_speed": 4.0,
         "mock_accel": 40.0, "mock_latency_s": 0.0, "mock_gripper_s": 0.02, "mock_tick_s": 0.005,
         "mock_yaw_speed": 3600.0, "home": {"x": 0.40, "y": 0.0, "z": 0.30, "yaw": 0.0}}
    c.update(over)
    return c


class FakeNode:
    def __init__(self, caps, cc):
        self.cfg = SimpleNamespace(node_id="legion", capabilities=list(caps), capability_config=cc, rank=100)
        self.topics = SimpleNamespace(latest={})
        self.published: list[tuple[str, dict]] = []
        self.events: list[tuple[str, str]] = []

    def set_blobs(self, blobs, age: float = 0.0, calibrated: bool = True):
        ts = time.time() - age
        self.topics.latest["cell.blobs"] = {"topic": "cell.blobs", "origin": "legion", "seq": 1, "ts": ts,
                                            "data": {"frame_ts": ts, "calibrated": calibrated,
                                                     "background": True, "blobs": blobs}}

    async def publish(self, topic, data):
        self.published.append((topic, data))

    def event(self, msg, level="info", **_):
        self.events.append((level, msg))

    def capabilities(self):
        return [c for c in self.cfg.capabilities if tasks.is_available(c)]


class Progress:
    def __init__(self):
        self.items: list[dict] = []

    async def __call__(self, p):
        self.items.append(p)


def blob(id_, x, y, yaw=0.0, length=0.18, width=0.02):
    return {"id": id_, "table_xy": [x, y], "yaw_deg": yaw, "length_m": length, "width_m": width}


def task(**over) -> dict:
    a = {"action": "move_object", "item": "screwdriver", "blob_id": 3, "table_xy": [0.30, 0.15],
         "yaw_deg": 0.0, "height": 0.03, "grasp": "across",
         "to": {"table_xy": [0.10, 0.20], "yaw_deg": None, "z": 0.04}}
    a.update(over)
    return a


async def start(**over) -> FakeNode:
    n = FakeNode(["arm.cell"], {"arm.cell": conf(**over)})
    await arm_cell.setup(n)
    return n


def svc() -> arm_cell.CellArmService:
    return arm_cell.get_service()


def moves():
    return [c for c in svc().safety.drv.log if c["cmd"] == "move_to"]


@pytest_asyncio.fixture(autouse=True)
async def _cleanup():
    yield
    await arm_cell.shutdown()
    await cell_arm.shutdown()


# ---------------------------------------------------------------- pure maths
def test_across_grasp_yaw_and_fold():
    assert arm_cell.grasp_yaw_table(0.0, "across") == 90.0
    assert arm_cell.grasp_yaw_table(30.0, "across") == 120.0
    assert arm_cell.grasp_yaw_table(30.0, "center") == 30.0
    with pytest.raises(ValueError):
        arm_cell.grasp_yaw_table(0.0, "none")
    # fold to the 180-degree equivalent within +-90 of the current tool yaw
    assert arm_cell.fold_yaw(120.0, 0.0) == pytest.approx(-60.0)
    assert arm_cell.fold_yaw(120.0, 100.0) == pytest.approx(120.0)
    assert arm_cell.fold_yaw(-170.0, 10.0) == pytest.approx(10.0)
    assert arm_cell.fold_yaw(80.0, None) == pytest.approx(80.0)
    for y in range(-180, 180, 7):
        for ref in (-150.0, -30.0, 0.0, 45.0, 170.0):
            f = arm_cell.fold_yaw(float(y), ref)
            assert abs((f - ref + 180) % 360 - 180) <= 90.0 + 1e-9
            assert abs(((f - y) % 180 + 90) % 180 - 90) < 1e-9        # same axis


@pytest.mark.asyncio
async def test_plan_table_to_arm_with_rotated_base():
    await start()
    s = svc()
    # base at (0.30, -0.25) table, rotated +90 deg, 1 cm below the table surface
    s.base = [0.30, -0.25, 90.0, -0.01]
    b = blob(3, 0.30, 0.15, yaw=30.0)
    p = s.plan(b, None, 0.03, "across", (0.10, 0.20), None, 0.04, tool_yaw=0.0)
    # table (0.30, 0.15) is 0.40 along the base's +x after the 90 deg rotation
    assert p["pick"]["x"] == pytest.approx(0.40)
    assert p["pick"]["y"] == pytest.approx(0.0, abs=1e-12)
    assert p["pick"]["z"] == pytest.approx(0.04)
    # across: blob 30 + 90 = 120 table -> 30 arm (base yaw 90); already within +-90 of tool yaw 0
    assert p["pick"]["yaw"] == pytest.approx(30.0)
    assert (p["place"]["x"], p["place"]["y"], p["place"]["z"]) == pytest.approx((0.45, 0.20, 0.05))
    assert p["place"]["yaw"] == pytest.approx(30.0)                # no to.yaw: keep the part's yaw
    # consistent with perception's own conversion
    assert P.table_to_arm((0.10, 0.20, 0.04), s.base) == pytest.approx((0.45, 0.20, 0.05))
    # tool currently at 170: pick yaw folds to 30 + 180 = -150 (within 90 of 170)
    p2 = s.plan(b, None, 0.03, "across", (0.10, 0.20), 60.0, 0.04, tool_yaw=170.0)
    assert p2["pick"]["yaw"] == pytest.approx(-150.0)
    # turn the part from 30 to 60 table: +30 at the wrist
    assert arm_cell.fold_yaw(p2["place"]["yaw"] - p2["pick"]["yaw"] - 30.0, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_relocate_rules():
    blobs = [blob(3, 0.32, 0.15), blob(7, 0.305, 0.15)]
    b, d = arm_cell.relocate(blobs, (0.30, 0.15), 3)
    assert b["id"] == 3 and d == pytest.approx(0.02)            # same id within 4 cm wins over nearer
    b, d = arm_cell.relocate(blobs, (0.30, 0.15), 99)
    assert b["id"] == 7                                          # unknown id: nearest within 4 cm
    with pytest.raises(RuntimeError, match="object not where expected"):
        arm_cell.relocate([blob(3, 0.345, 0.15)], (0.30, 0.15), 3)
    with pytest.raises(RuntimeError, match="object not where expected"):
        arm_cell.relocate([], (0.30, 0.15), 3)


# ---------------------------------------------------------------- move_object on the mock arm
@pytest.mark.asyncio
async def test_move_object_relocates_and_moves():
    n = await start()
    n.set_blobs([blob(3, 0.32, 0.16, yaw=0.0), blob(5, 0.5, 0.3)])   # nudged ~2.2 cm since labelling
    pr = Progress()
    r = await REGISTRY["arm.cell"](task(_task_id="t1"), pr)
    assert r["moved"] == "screwdriver" and r["found"]["blob_id"] == 3
    assert r["found"]["moved_m"] == pytest.approx(math.hypot(0.02, 0.01), abs=1e-4)
    exp_pick = P.table_to_arm((0.32, 0.16, 0.03), BASE)
    assert (r["pick"]["x"], r["pick"]["y"], r["pick"]["z"]) == pytest.approx(exp_pick, abs=1e-4)
    # across a blob at table yaw 0 -> finger axis at table 90 -> arm 0 (base yaw 90)
    assert r["pick"]["yaw"] == pytest.approx(0.0, abs=1e-6)
    exp_place = P.table_to_arm((0.10, 0.20, 0.04), BASE)
    assert (r["place"]["x"], r["place"]["y"], r["place"]["z"]) == pytest.approx(exp_place, abs=1e-4)
    grips = [c["value"] for c in svc().safety.drv.log if c["cmd"] == "gripper"]
    # open, grip, release; the grip closes to the blob's 2 cm width minus the 9 mm squeeze = 1.1 cm gap of 10 cm
    assert grips == pytest.approx([1.0, 0.11, 1.0], abs=1e-6)
    # the carry runs level at the carry height, at carry_speed
    carry_z = max(exp_pick[2], exp_place[2]) + 0.08
    level = [m for m in moves() if abs(m["z"] - carry_z) < 1e-6]
    assert any(abs(m["x"] - exp_place[0]) < 1e-6 and m["speed"] == pytest.approx(0.5) for m in level)
    steps = [p.get("step") for p in pr.items]
    assert steps[0] == "relocate" and "relocated" in steps and "carry" in steps and steps[-1] == "done"
    pcts = [p["pct"] for p in pr.items if "pct" in p]
    assert pcts == sorted(pcts) and pcts[-1] == 100


@pytest.mark.asyncio
async def test_move_object_refuses_when_object_moved():
    n = await start()
    n.set_blobs([blob(3, 0.30 + 0.045, 0.15)])                   # 4.5 cm away
    with pytest.raises(RuntimeError, match="object not where expected"):
        await REGISTRY["arm.cell"](task(_task_id="t2"), Progress())
    assert moves() == []                                          # never grabbed blindly


@pytest.mark.asyncio
async def test_move_object_refuses_stale_or_missing_blobs():
    n = await start()
    with pytest.raises(RuntimeError, match="no cell.blobs"):
        await REGISTRY["arm.cell"](task(), Progress())
    n.set_blobs([blob(3, 0.30, 0.15)], age=10.0)
    with pytest.raises(RuntimeError, match="old"):
        await REGISTRY["arm.cell"](task(), Progress())
    n.set_blobs([blob(3, 0.30, 0.15)], calibrated=False)
    with pytest.raises(RuntimeError, match="not calibrated"):
        await REGISTRY["arm.cell"](task(), Progress())
    n.set_blobs([blob(3, 0.30, 0.15)])
    with pytest.raises(ValueError, match="grasp"):
        await REGISTRY["arm.cell"](task(grasp="none"), Progress())
    assert moves() == []


@pytest.mark.asyncio
async def test_move_object_rejects_out_of_workspace_before_moving():
    n = await start()
    n.set_blobs([blob(3, 0.30, 0.15)])
    with pytest.raises(ValueError, match="outside workspace"):
        await REGISTRY["arm.cell"](task(to={"table_xy": [2.0, 2.0], "z": 0.04}), Progress())
    assert moves() == []


@pytest.mark.asyncio
async def test_idempotent_by_task_id():
    n = await start()
    n.set_blobs([blob(3, 0.30, 0.15)])
    r1 = await REGISTRY["arm.cell"](task(_task_id="same"), Progress())
    k = len(svc().safety.drv.log)
    n.set_blobs([])                                               # would fail if it ran again
    r2 = await REGISTRY["arm.cell"](task(_task_id="same"), Progress())
    assert r2 is r1 and len(svc().safety.drv.log) == k


@pytest.mark.asyncio
async def test_cancel_is_estop():
    n = await start(slow_speed=0.02, mock_max_speed=0.05)
    n.set_blobs([blob(3, 0.30, 0.15)])
    t = asyncio.create_task(REGISTRY["arm.cell"](task(_task_id="c1"), Progress()))
    await asyncio.sleep(0.3)
    assert svc().busy == "move_object"
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    s = svc().safety
    assert s.estop and s.estop_reason == "task cancelled"
    assert s.drv.log[-1]["cmd"] == "stop"
    assert svc().busy is None
    # stop action also works and the next command clears the (non-latched) e-stop
    assert (await REGISTRY["arm.cell"]({"action": "stop"}, Progress())) == {"stopped": True}
    r = await REGISTRY["arm.cell"]({"action": "home", "_task_id": "h1"}, Progress())
    assert r["homed"] and not s.estop


# ---------------------------------------------------------------- availability
@pytest.mark.asyncio
async def test_availability_follows_connection():
    n = await start()
    s = svc()
    assert tasks.is_available("arm.cell") and "arm.cell" in n.capabilities()
    drv = s.safety.drv
    real_pose, real_connect = drv.pose, drv.connect

    async def unplugged(*_a, **_k):
        raise ConnectionError("COM5 disappeared")
    drv.pose, drv.connect = unplugged, unplugged                  # USB cable pulled
    await s.check_health()
    assert not tasks.is_available("arm.cell") and "arm.cell" not in n.capabilities()
    await s.check_health()                                        # reconnect attempt fails: still gone
    assert not tasks.is_available("arm.cell")
    with pytest.raises(RuntimeError, match="not connected"):
        await REGISTRY["arm.cell"]({"action": "home"}, Progress())
    drv.pose, drv.connect = real_pose, real_connect               # plugged back in
    await s.check_health()
    assert tasks.is_available("arm.cell") and "arm.cell" in n.capabilities()
    st = await s.state()
    assert st["available"] and st["connected"] and st["arm_id"] == "testarm"


@pytest.mark.asyncio
async def test_health_loop_withdraws_within_seconds():
    n = await start(health_interval_s=0.1)
    drv = svc().safety.drv

    async def unplugged(*_a, **_k):
        raise ConnectionError("gone")
    drv.pose, drv.connect = unplugged, unplugged
    await asyncio.sleep(0.35)
    assert "arm.cell" not in n.capabilities()


@pytest.mark.asyncio
async def test_state_topic_and_coexistence_with_cell_arm():
    n = FakeNode(["arm.cell", "arm.big"], {"arm.cell": conf(), "arm.big": {"driver": "mock", "workspace": WS}})
    await arm_cell.setup(n)
    await cell_arm.setup(n)                                       # runs after arm_cell (alphabetical)
    assert getattr(REGISTRY["arm.cell"], "_arm_cell", False)
    assert cell_arm.get_arm("arm.cell") is None and cell_arm.get_arm("arm.big") is not None
    await asyncio.sleep(0.15)
    topics = {t for t, _ in n.published}
    assert "arm.cell.testarm.state" in topics and "arm.big.state" in topics


@pytest.mark.asyncio
async def test_takes_over_if_cell_arm_registered_first():
    n = FakeNode(["arm.cell"], {"arm.cell": conf()})
    await cell_arm.setup(n)
    assert getattr(REGISTRY["arm.cell"], "_cell_arm", False)
    await arm_cell.setup(n)
    assert getattr(REGISTRY["arm.cell"], "_arm_cell", False) and cell_arm.get_arm("arm.cell") is None


@pytest.mark.asyncio
async def test_inactive_without_capability():
    n = FakeNode(["arm.big"], {"arm.cell": conf()})
    await arm_cell.setup(n)
    assert arm_cell.get_service() is None and "arm.cell" not in REGISTRY


# ---------------------------------------------------------------- reBot kinematics
def rebot_cfg(**over) -> dict:
    c = {"driver": "rebot", "workspace": {"x": [0.10, 0.50], "y": [-0.40, 0.40], "z": [-0.10, 0.17]},
         "gripper_open_rad": -1.2, "port": "COM_TEST"}
    c.update(over)
    return c


def test_rebot_urdf_chain_matches_published_zero_pose():
    a = rebot.RebotArm(rebot_cfg())
    assert a.chain.names == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    assert a.chain.lower.tolist() == [-2.8, -3.14, -3.14, -1.87, -1.57, -3.14]
    T = a.chain.fk([0.0] * 6)
    # zero pose of ReBot_Arm_DM.urdf: tool 0.260 m forward, 0.192 m up, approach axis horizontal
    assert T[:3, 3] == pytest.approx([0.2603, 0.0, 0.1917], abs=2e-4)
    assert T[:3, 0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-3)


@pytest.mark.parametrize("target", [(0.25, 0.0, 0.05, 0.0), (0.30, 0.10, 0.02, 45.0), (0.20, -0.15, 0.10, -90.0),
                                    (0.40, 0.05, -0.05, 120.0), (0.15, 0.0, 0.12, 0.0)])
def test_rebot_ik_reaches_targets_tool_down(target):
    a = rebot.RebotArm(rebot_cfg())
    x, y, z, yaw = target
    q = a.ik(x, y, z, yaw)
    assert np.all(q >= a.chain.lower - 1e-9) and np.all(q <= a.chain.upper + 1e-9)
    T = a.chain.fk(q)
    assert np.linalg.norm(T[:3, 3] - [x, y, z]) < 1e-3                       # < 1 mm
    assert T[:3, 0] == pytest.approx([0, 0, -1], abs=5e-3)                    # tool straight down
    p = a.tool_pose(q)
    assert abs((p["yaw"] - yaw + 180) % 360 - 180) < 0.5                     # finger axis at the asked yaw


@pytest.mark.parametrize("target", [(0.90, 0.0, 0.05), (0.30, 0.0, 0.35), (0.0, 0.0, -0.30), (2.0, 2.0, 0.0)])
def test_rebot_ik_rejects_unreachable(target):
    a = rebot.RebotArm(rebot_cfg())
    with pytest.raises(rebot.IKError):
        a.ik(*target, 0.0)


def test_rebot_ik_round_trip_from_fk():
    a = rebot.RebotArm(rebot_cfg())
    rng = np.random.default_rng(1)
    ok = 0
    # re-solve from a perturbed seed near a known solution (the path planner's warm start)
    for x, y, z in [(0.22, 0.05, 0.03), (0.33, -0.08, 0.08), (0.28, 0.12, -0.02)]:
        q = a.ik(x, y, z, 10.0)
        q2 = a.ik(x, y, z, 10.0, seed=q + rng.normal(0, 0.05, 6))
        assert np.linalg.norm(a.chain.fk(q2)[:3, 3] - [x, y, z]) < 1e-3
        ok += 1
    assert ok == 3


def test_rebot_line_plan_stays_tool_down_and_continuous():
    a = rebot.RebotArm(rebot_cfg())
    q0 = a.ik(0.25, 0.0, 0.10, 0.0)
    path = a.plan_line(q0, 0.35, 0.10, 0.02, 30.0)
    assert len(path) > 10
    for qa, qb in zip(path, path[1:]):
        assert float(np.max(np.abs(qb - qa))) <= a.max_joint_step
    for q in path:
        assert a.tool_pose(q)["tilt_deg"] < 0.5


def test_rebot_config_errors():
    with pytest.raises(Exception, match="gripper_open_rad"):
        rebot.RebotArm({"workspace": WS})
    with pytest.raises(Exception, match="urdf_path"):
        rebot.RebotArm(rebot_cfg(variant="rs"))
    a = rebot.RebotArm({"workspace": WS, "gripper_open_rad": -1.0})
    assert a.port == "COM5" and a.transport == "dm_serial" and a.baud == 921600
    assert rebot.RebotArm(rebot_cfg(port="/dev/ttyACM0")).transport == "dm_serial"
    assert rebot.RebotArm(rebot_cfg(port="can0")).transport == "socketcan"


@pytest.mark.asyncio
async def test_rebot_missing_sdk_names_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "motorbridge", None)        # import motorbridge -> ImportError
    monkeypatch.setattr(rebot, "port_present", lambda p: True)
    a = rebot.RebotArm(rebot_cfg())
    with pytest.raises(RuntimeError, match="pip install motorbridge"):
        await a.connect()
    await a.close()


# ---------------------------------------------------------------- reBot against a fake motorbridge
class FakeMotor:
    def __init__(self, bus, mid, fid, model):
        self.bus, self.mid, self.fid, self.model = bus, mid, fid, model
        self.pos, self.enabled, self.mode, self.regs, self.fed = 0.0, False, None, {}, False
        self.cmds: list = []

    def clear_error(self):
        self.cmds.append("clear_error")

    def write_register_f32(self, reg, v):
        self.regs[reg] = v

    def ensure_mode(self, mode, timeout_ms):
        self.mode = mode

    def enable(self):
        self.enabled = True

    def request_feedback(self):
        if not self.bus.alive:
            raise self.bus.CallError("serial write failed")
        self.fed = True

    def get_state(self):
        if not self.fed:
            return None
        return SimpleNamespace(pos=self.pos, vel=0.0, torq=0.0, t_mos=30.0, status_code=0)

    def send_pos_vel(self, pos, vlim):
        self.cmds.append(("pos_vel", pos, vlim))
        if self.enabled:
            self.pos = pos                       # ideal servo: arrives at once

    def send_force_pos(self, pos, vlim, ratio):
        self.send_pos_vel(pos, vlim)


def fake_motorbridge(start_q):
    mb = types.ModuleType("motorbridge")
    mb.Mode = SimpleNamespace(MIT=1, POS_VEL=2, VEL=3, FORCE_POS=4)

    class CallError(Exception):
        pass

    class Controller:
        instances: list = []

        def __init__(self, channel="can0"):
            self.channel, self.alive, self.motors, self.log = channel, True, [], []
            self.CallError = CallError
            Controller.instances.append(self)

        @classmethod
        def from_dm_serial(cls, serial_port="/dev/ttyACM0", baud=921600):
            c = cls(serial_port)
            c.baud = baud
            return c

        def add_damiao_motor(self, mid, fid, model):
            m = FakeMotor(self, mid, fid, model)
            m.pos = float(start_q[len(self.motors)])
            self.motors.append(m)
            return m

        def poll_feedback_once(self):
            pass

        def disable_all(self):
            self.log.append("disable_all")
            for m in self.motors:
                m.enabled = False

        def shutdown(self):
            self.log.append("shutdown")

        def close(self):
            self.log.append("close")

    mb.Controller, mb.CallError = Controller, CallError
    return mb


@pytest.mark.asyncio
async def test_rebot_driver_with_fake_sdk(monkeypatch):
    probe = rebot.RebotArm(rebot_cfg())
    q_start = list(probe.ik(0.25, 0.0, 0.10, 0.0)) + [0.0]       # tool-down start, gripper closed
    mb = fake_motorbridge(q_start)
    monkeypatch.setitem(sys.modules, "motorbridge", mb)
    monkeypatch.setattr(rebot, "port_present", lambda p: True)
    a = rebot.RebotArm(rebot_cfg(joint_speed_max=20.0, control_hz=200, gripper_timeout_s=0.5))
    await a.connect()
    ctrl = mb.Controller.instances[-1]
    assert ctrl.channel == "COM_TEST" and ctrl.baud == 921600
    assert [m.mid for m in ctrl.motors] == [1, 2, 3, 4, 5, 6, 7]
    assert [m.fid for m in ctrl.motors] == [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
    assert all(m.enabled and m.mode == 2 for m in ctrl.motors)
    assert ctrl.motors[0].regs == {25: 0.0125, 26: 0.004, 27: 150.0, 28: 0.5}
    # the first command after enable holds the measured position (no jump)
    first = [c for c in ctrl.motors[1].cmds if isinstance(c, tuple)][0]
    assert first[1] == pytest.approx(q_start[1])
    p = await a.pose()
    assert (p["x"], p["y"], p["z"]) == pytest.approx((0.25, 0.0, 0.10), abs=1e-3)
    await a.move_to(0.32, 0.08, 0.03, 20.0, 2.0)
    p = await a.pose()
    assert (p["x"], p["y"], p["z"]) == pytest.approx((0.32, 0.08, 0.03), abs=1e-3)
    assert ((p["yaw"] - 20.0 + 180) % 360 - 180) == pytest.approx(0.0, abs=0.5)
    vl = [c[2] for c in ctrl.motors[0].cmds if isinstance(c, tuple)]
    assert max(vl) <= 20.0
    await a.gripper(1.0)
    assert ctrl.motors[6].pos == pytest.approx(-1.2) and a.grip == 1.0
    await a.ping()
    ctrl.alive = False                                           # USB pulled: the bus write fails
    with pytest.raises(Exception, match="serial write failed"):
        await a.ping()
    await a.close()
    assert ctrl.log[:1] == ["disable_all"] and "close" in ctrl.log


@pytest.mark.asyncio
async def test_arm_cell_on_rebot_fake_sdk_checks_reach(monkeypatch):
    probe = rebot.RebotArm(rebot_cfg())
    mb = fake_motorbridge(list(probe.ik(0.25, 0.0, 0.10, 0.0)) + [0.0])
    monkeypatch.setitem(sys.modules, "motorbridge", mb)
    monkeypatch.setattr(rebot, "port_present", lambda p: True)
    # base: arm origin at table (0.0, 0.0), rotated 0, on the table surface
    c = rebot_cfg(base=[0.0, 0.0, 0.0, 0.0], arm_id="rebot", joint_speed_max=20.0, control_hz=200,
                  max_speed=2.0, speed=1.0, slow_speed=1.0, retreat_speed=1.0, carry_speed=1.0,
                  approach_h=0.06, insert_dwell_s=0.0, gripper_timeout_s=0.2, health_interval_s=3600,
                  home={"x": 0.25, "y": 0.0, "z": 0.10, "yaw": 0.0})
    n = FakeNode(["arm.cell"], {"arm.cell": c})
    await arm_cell.setup(n)
    assert tasks.is_available("arm.cell")
    n.set_blobs([blob(4, 0.30, 0.05, yaw=10.0)])
    r = await REGISTRY["arm.cell"](task(blob_id=4, table_xy=[0.30, 0.05], height=0.02,
                                        to={"table_xy": [0.22, -0.10], "z": 0.02}, _task_id="rb1"), Progress())
    assert r["found"]["blob_id"] == 4
    p = await svc().safety.drv.pose()
    assert (p["x"], p["y"]) == pytest.approx((0.22, -0.10), abs=1e-3)
    assert p["z"] == pytest.approx(0.08, abs=1e-3)                # retreated approach_h above the place
    # destination outside the tool-down reach: refused before anything moves
    drv = svc().safety.drv
    before = list(mb.Controller.instances[-1].motors[0].cmds)
    n.set_blobs([blob(4, 0.22, -0.10, yaw=10.0)])
    with pytest.raises(rebot.IKError, match="not reachable|beyond"):
        await REGISTRY["arm.cell"](task(blob_id=4, table_xy=[0.22, -0.10], height=0.02,
                                        to={"table_xy": [0.48, 0.38], "z": 0.02}, _task_id="rb2"), Progress())
    assert mb.Controller.instances[-1].motors[0].cmds == before
    assert drv is svc().safety.drv
