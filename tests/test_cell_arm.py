"""Cell arm capability (capabilities/cell_arm.py + cell/arm.py) on the mock driver: no cluster,
no hardware. Vendor SDKs are not installed; the xArm/UR unit conversions are checked against tiny
in-process fakes of the SDK modules.
"""
from __future__ import annotations

import asyncio
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from swarmmind.capabilities import builtin, cell_arm  # noqa: E402,F401
from swarmmind.cell import arm as cellarm  # noqa: E402
from swarmmind.cell.drivers import ur as ur_drv  # noqa: E402
from swarmmind.tasks import REGISTRY  # noqa: E402

WS = {"x": [0.15, 0.70], "y": [-0.45, 0.45], "z": [0.02, 0.60]}


def conf(**over) -> dict:
    c = {"driver": "mock", "workspace": WS, "max_speed": 1.0, "speed": 0.5, "slow_speed": 0.1,
         "retreat_speed": 0.2, "insert_speed": 0.05, "insert_step_m": 0.005, "insert_dwell_s": 0.0,
         "approach_h": 0.08, "telemetry_hz": 20, "mock_max_speed": 2.0, "mock_accel": 20.0,
         "mock_latency_s": 0.0, "mock_gripper_s": 0.05, "mock_tick_s": 0.01, "mock_yaw_speed": 720.0,
         "home": {"x": 0.40, "y": 0.0, "z": 0.40, "yaw": 0.0}}
    c.update(over)
    return c


class FakeNode:
    def __init__(self, caps: list[str], cc: dict):
        self.cfg = SimpleNamespace(node_id="armhost", capabilities=caps, capability_config=cc)
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


class Progress:
    def __init__(self):
        self.items: list[dict] = []

    async def __call__(self, p):
        self.items.append(p)


async def start(caps=("arm.big",), **cc) -> FakeNode:
    n = FakeNode(list(caps), cc or {"arm.big": conf()})
    await cell_arm.setup(n)
    return n


@pytest_asyncio.fixture
async def node():
    n = await start()
    yield n
    await cell_arm.shutdown()


def mock_log():
    return cell_arm.get_arm("arm.big").safety.drv.log


async def run(args, p=None, name="arm.big"):
    return await REGISTRY[name](args, p or Progress())


# ------------------------------------------------------------------ setup / telemetry
@pytest.mark.asyncio
async def test_setup_registers_only_cell_arms_and_publishes_state():
    n = await start(caps=["arm.big", "arm.so101", "arm.mock", "echo"],
                    **{"arm.big": conf(), "arm.so101": {"mock": True}})
    try:
        assert "arm.big" in REGISTRY and cell_arm.get_arm("arm.big") is not None
        assert cell_arm.get_arm("arm.so101") is None and cell_arm.get_arm("arm.mock") is None
        assert getattr(REGISTRY["arm.mock"], "_cell_arm", False) is False      # builtin untouched
        await asyncio.sleep(0.2)
        states = [d for t, d in n.published if t == "arm.big.state"]
        assert len(states) >= 2
        s = states[-1]
        assert {"pose", "gripper", "busy", "driver", "estop"} <= set(s)
        assert s["driver"] == "mock" and s["busy"] is False and s["estop"] is False
        assert s["pose"] == {"x": 0.40, "y": 0.0, "z": 0.40, "yaw": 0.0}
        assert s["gripper"] == 1.0
    finally:
        await cell_arm.shutdown()
    assert "arm.big" not in REGISTRY


@pytest.mark.asyncio
async def test_state_shows_busy_and_moving_pose(node):
    t = asyncio.create_task(run({"action": "move_to", "x": 0.60, "y": 0.30, "z": 0.30, "speed": 0.2,
                                 "_task_id": "busy1"}))
    await asyncio.sleep(0.25)
    node.published.clear()
    await asyncio.sleep(0.15)
    states = [d for tp, d in node.published if tp == "arm.big.state"]
    assert states and states[-1]["busy"] is True and states[-1]["action"] == "move_to"
    assert 0.40 < states[-1]["pose"]["x"] < 0.60
    await t
    await asyncio.sleep(0.1)
    last = [d for tp, d in node.published if tp == "arm.big.state"][-1]
    assert last["busy"] is False and last["pose"]["x"] == pytest.approx(0.60)


# ------------------------------------------------------------------ safety
@pytest.mark.asyncio
async def test_workspace_rejection_and_speed_cap(node):
    with pytest.raises(ValueError, match="outside workspace"):
        await run({"action": "move_to", "x": 0.9, "y": 0.0, "z": 0.3, "_task_id": "r1"})
    with pytest.raises(ValueError, match="missing z"):
        await run({"action": "move_to", "x": 0.3, "y": 0.0, "_task_id": "r2"})
    with pytest.raises(ValueError):
        await run({"action": "move_to", "x": float("nan"), "y": 0.0, "z": 0.3, "_task_id": "r3"})
    # pick whose approach point (z + approach_h) is above the ceiling: refused before ANY motion
    with pytest.raises(ValueError, match="outside workspace"):
        await run({"action": "pick", "x": 0.4, "y": 0.0, "z": 0.55, "approach_h": 0.1, "_task_id": "r4"})
    with pytest.raises(ValueError, match="insert_depth"):
        await run({"action": "place", "x": 0.4, "y": 0.0, "z": 0.1, "approach_h": 0.01,
                   "insert_depth": 0.02, "_task_id": "r5"})
    with pytest.raises(ValueError, match="unknown action"):
        await run({"action": "fly", "_task_id": "r6"})
    assert mock_log() == []
    r = await run({"action": "move_to", "x": 0.5, "y": 0.1, "z": 0.2, "yaw": 190, "speed": 9.0,
                   "_task_id": "ok"})
    assert r["speed"] == 1.0 and r["reached"]["yaw"] == pytest.approx(-170.0)
    assert mock_log()[-1]["speed"] == 1.0


@pytest.mark.asyncio
async def test_clamp_mode():
    await start(**{"arm.big": conf(out_of_bounds="clamp")})
    try:
        r = await run({"action": "move_to", "x": 0.9, "y": -1.0, "z": 0.3, "_task_id": "c1"})
        assert r["clamped"] is True and r["reached"]["x"] == 0.70 and r["reached"]["y"] == -0.45
        # primitives never clamp
        with pytest.raises(ValueError):
            await run({"action": "pick", "x": 0.9, "y": 0.0, "z": 0.1, "_task_id": "c2"})
    finally:
        await cell_arm.shutdown()


# ------------------------------------------------------------------ primitives
@pytest.mark.asyncio
async def test_pick_substep_order(node):
    p = Progress()
    await run({"action": "move_to", "x": 0.3, "y": -0.2, "z": 0.05, "_task_id": "low"})   # start low
    mock_log().clear()
    r = await run({"action": "pick", "x": 0.5, "y": 0.1, "z": 0.03, "yaw": 30, "approach_h": 0.08,
                   "_task_id": "p1"}, p)
    subs = [x["sub"] for x in p.items if x.get("step") == "pick"]
    assert subs == ["open", "approach", "descend", "grip", "retreat", "done"]
    assert r["steps"] == ["open", "approach", "descend", "grip", "retreat"]
    cmds = [(c["cmd"], c.get("value"), c.get("z"), c.get("speed")) for c in mock_log()]
    # open, lift vertically in place, go above, slow descent, close, retreat up
    assert cmds[0] == ("gripper", 1.0, None, None)
    assert mock_log()[1]["x"] == pytest.approx(0.3) and mock_log()[1]["z"] == pytest.approx(0.11)
    assert mock_log()[2]["x"] == pytest.approx(0.5) and mock_log()[2]["z"] == pytest.approx(0.11)
    assert cmds[3][2] == pytest.approx(0.03) and cmds[3][3] == 0.1               # slow_speed
    assert cmds[4] == ("gripper", 0.0, None, None)
    assert cmds[5][2] == pytest.approx(0.11) and cmds[5][3] == 0.2               # retreat_speed
    st = await cell_arm.get_arm("arm.big").state()
    assert st["gripper"] == 0.0 and st["pose"]["yaw"] == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_place_slow_insert(node):
    p = Progress()
    r = await run({"action": "place", "x": 0.45, "y": -0.1, "z": 0.05, "yaw": 0, "approach_h": 0.08,
                   "insert_depth": 0.02, "_task_id": "pl1"}, p)
    subs = [x["sub"] for x in p.items if x.get("step") == "place"]
    first = list(dict.fromkeys(subs))
    assert first == ["approach", "descend", "insert", "release", "retreat", "done"]
    moves = [c for c in mock_log() if c["cmd"] == "move_to"]
    grips = [c for c in mock_log() if c["cmd"] == "gripper"]
    assert moves[0]["z"] == pytest.approx(0.13)                   # above = z + approach_h
    assert moves[1]["z"] == pytest.approx(0.07) and moves[1]["speed"] == 0.1   # to z + insert_depth
    ins = moves[2:-1]
    assert len(ins) == 4 == r["insert_steps"]                     # 0.02 / 0.005
    zs = [m["z"] for m in ins]
    assert all(m["speed"] == 0.05 for m in ins)                   # insert_speed
    assert zs == sorted(zs, reverse=True) and zs[-1] == pytest.approx(0.05)
    steps = [a - b for a, b in zip([0.07] + zs[:-1], zs)]
    assert all(0 < s <= 0.005 + 1e-9 for s in steps)
    assert len(grips) == 1 and grips[0]["value"] == 1.0          # opens only after seating
    assert mock_log().index(grips[0]) > mock_log().index(ins[-1])
    assert moves[-1]["z"] == pytest.approx(0.13)                  # retreat
    assert [x["insert_i"] for x in p.items if "insert_i" in x] == [1, 2, 3, 4]


# ------------------------------------------------------------------ stop / cancel / idempotency
@pytest.mark.asyncio
async def test_cancel_is_estop_mid_move(node):
    svc = cell_arm.get_arm("arm.big")
    t = asyncio.create_task(run({"action": "move_to", "x": 0.65, "y": 0.4, "z": 0.40, "speed": 0.1,
                                 "_task_id": "slow"}))
    await asyncio.sleep(0.3)
    assert svc.busy == "move_to"
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert svc.safety.estop is True and svc.busy is None
    assert any(c["cmd"] == "stop" for c in mock_log())
    frozen = (await svc.safety.pose())["x"]
    assert 0.40 < frozen < 0.6
    await asyncio.sleep(0.3)
    assert (await svc.safety.pose())["x"] == pytest.approx(frozen), "arm kept moving after cancel"
    st = await svc.state()
    assert st["estop"] is True
    # not latched by default: the next command clears it
    await run({"action": "home", "_task_id": "after"})
    assert svc.safety.estop is False
    assert (await svc.safety.pose())["x"] == pytest.approx(0.40)


@pytest.mark.asyncio
async def test_estop_latch():
    await start(**{"arm.big": conf(estop_latch=True)})
    try:
        svc = cell_arm.get_arm("arm.big")
        await svc.safety.emergency_stop("test")
        with pytest.raises(RuntimeError, match="latched"):
            await run({"action": "home", "_task_id": "l1"})
        await run({"action": "reset", "_task_id": "l2"})
        await run({"action": "home", "_task_id": "l3"})
    finally:
        await cell_arm.shutdown()


@pytest.mark.asyncio
async def test_stop_action_aborts_running_and_queued(node):
    t1 = asyncio.create_task(run({"action": "pick", "x": 0.6, "y": 0.4, "z": 0.05, "speed": 0.1,
                                  "_task_id": "long"}))
    await asyncio.sleep(0.15)
    t2 = asyncio.create_task(run({"action": "home", "_task_id": "queued"}))
    await asyncio.sleep(0.05)
    assert await run({"action": "stop", "_task_id": "s1"}) == {"stopped": True}
    with pytest.raises(RuntimeError, match="aborted"):
        await t1
    with pytest.raises(RuntimeError, match="queued"):
        await t2
    assert cell_arm.get_arm("arm.big").safety.estop is False        # stop is not an e-stop
    n_before = len(mock_log())
    await run({"action": "gripper", "value": "close", "_task_id": "g"})
    assert len(mock_log()) == n_before + 1


@pytest.mark.asyncio
async def test_idempotent_by_task_id(node):
    r1 = await run({"action": "move_to", "x": 0.5, "y": 0.2, "z": 0.3, "_task_id": "same"})
    await run({"action": "move_to", "x": 0.3, "y": -0.2, "z": 0.2, "_task_id": "other"})
    n = len(mock_log())
    r2 = await run({"action": "move_to", "x": 0.5, "y": 0.2, "z": 0.3, "_task_id": "same"})
    assert r2 == r1 and len(mock_log()) == n
    assert (await cell_arm.get_arm("arm.big").safety.pose())["x"] == pytest.approx(0.3)


# ------------------------------------------------------------------ real drivers without SDKs
@pytest.mark.asyncio
async def test_real_driver_without_sdk_fails_clearly_or_falls_back():
    n = await start(caps=["arm.big", "arm.alt"],
                    **{"arm.big": conf(driver="kinova", ip="127.0.0.1", port=1, connect_timeout_s=3, fallback_to_mock=True),
                       "arm.alt": conf(driver="xarm", ip="10.255.255.1")})
    try:
        big = cell_arm.get_arm("arm.big")
        assert big.connected and big.safety.drv.name == "mock" and big.fallback_from == "kinova"
        assert any(lvl == "error" and "MOCK" in m for lvl, m in n.events)
        r = await run({"action": "move_to", "x": 0.5, "y": 0.0, "z": 0.3, "_task_id": "fb"})
        assert r["reached"]["x"] == 0.5
        alt = cell_arm.get_arm("arm.alt")
        assert not alt.connected and "xarm-python-sdk" in alt.connect_error
        with pytest.raises(RuntimeError, match="pip install xarm-python-sdk"):
            await run({"action": "home", "_task_id": "x1"}, name="arm.alt")
        await asyncio.sleep(0.15)
        st = [d for t, d in n.published if t == "arm.alt.state"][-1]
        assert st["connected"] is False and st["driver"] == "xarm" and "xarm-python-sdk" in st["fault"]
    finally:
        await cell_arm.shutdown()


@pytest.mark.asyncio
async def test_real_driver_requires_workspace():
    c = conf(driver="ur", ip="1.2.3.4", fallback_to_mock=True)
    c.pop("workspace")
    await start(**{"arm.big": c})
    try:
        svc = cell_arm.get_arm("arm.big")
        assert not svc.connected and "workspace" in svc.connect_error     # no silent mock fallback
        with pytest.raises(RuntimeError, match="workspace"):
            await run({"action": "home", "_task_id": "w"})
    finally:
        await cell_arm.shutdown()


@pytest.mark.asyncio
async def test_so101_driver_refuses_cartesian(tmp_path):
    await start(**{"arm.big": {"driver": "so101", "mock": True, "rate_hz": 100, "max_step_ticks": 100,
                               "mock_speed_ticks_s": 40000, "tolerance_ticks": 5, "settle_s": 0.2}})
    try:
        svc = cell_arm.get_arm("arm.big")
        assert svc.connected and svc.safety.drv.supports_cartesian is False
        with pytest.raises(RuntimeError, match="inverse kinematics"):
            await run({"action": "move_to", "x": 0.3, "y": 0.0, "z": 0.2, "_task_id": "c"})
        with pytest.raises(RuntimeError, match="inverse kinematics"):
            await run({"action": "pick", "x": 0.3, "y": 0.0, "z": 0.1, "_task_id": "p"})
        r = await run({"action": "home", "_task_id": "h"})
        assert r["homed"] is True
        await run({"action": "gripper", "value": 0.0, "_task_id": "g"})
        st = await svc.state()
        assert st["driver"] == "so101" and st["pose"]["x"] is None and st["gripper"] == 0.0
        with pytest.raises(ValueError, match="unknown pose"):
            await run({"action": "named_pose", "name": "nope", "_task_id": "n"})
    finally:
        await cell_arm.shutdown()


# ------------------------------------------------------------------ unit conversions (fake SDKs)
def test_ur_rotvec_math():
    for yaw in (-170.0, -90.0, 0.0, 30.0, 90.0, 179.0):
        R = ur_drv.rpy_to_matrix(180.0, 0.0, yaw)
        rv = ur_drv.matrix_to_rotvec(R)
        assert math.sqrt(sum(c * c for c in rv)) == pytest.approx(math.pi, abs=1e-9)   # tool down = 180 deg
        R2 = ur_drv.rotvec_to_matrix(rv)
        assert all(R[i][j] == pytest.approx(R2[i][j], abs=1e-9) for i in range(3) for j in range(3))
        assert ur_drv.yaw_of(R2) == pytest.approx(yaw, abs=1e-6)
        assert R2[2][2] == pytest.approx(-1.0)                   # tool z points down
    assert ur_drv.matrix_to_rotvec(ur_drv.rpy_to_matrix(180, 0, 0)) == pytest.approx([math.pi, 0, 0])
    small = ur_drv.matrix_to_rotvec(ur_drv.rpy_to_matrix(0, 0, 10))
    assert small == pytest.approx([0, 0, math.radians(10)])


@pytest.mark.asyncio
async def test_ur_driver_units_with_fake_rtde(monkeypatch):
    calls = []

    class RC:
        def __init__(self, ip):
            self.pose = None

        def moveL(self, pose, speed, accel, asynchronous):
            calls.append(("moveL", list(pose), speed, accel, asynchronous))
            FAKE["tcp"] = list(pose)
            return True

        def getAsyncOperationProgress(self):
            return -1

        def stopL(self, a):
            calls.append(("stopL", a))

    class RR:
        def __init__(self, ip):
            pass

        def getActualTCPPose(self):
            return list(FAKE["tcp"])

        def isEmergencyStopped(self):
            return False

        def isProtectiveStopped(self):
            return False

    FAKE = {"tcp": [0.4, 0.0, 0.4, math.pi, 0.0, 0.0]}
    monkeypatch.setitem(sys.modules, "rtde_control", types.SimpleNamespace(RTDEControlInterface=RC))
    monkeypatch.setitem(sys.modules, "rtde_receive", types.SimpleNamespace(RTDEReceiveInterface=RR))
    s = cellarm.build_arm(conf(driver="ur", ip="1.2.3.4", max_speed=0.3))
    await s.drv.connect()
    await s.move_to(0.5, -0.1, 0.2, 90.0, 2.0)
    name, pose, speed, accel, asyn = calls[-1]
    assert name == "moveL" and asyn is True and speed == 0.3           # m/s, capped
    assert pose[:3] == [0.5, -0.1, 0.2]                                 # metres, unscaled
    assert math.sqrt(sum(c * c for c in pose[3:])) == pytest.approx(math.pi)   # radians
    p = await s.pose()
    assert p["yaw"] == pytest.approx(90.0) and p["x"] == 0.5


@pytest.mark.asyncio
async def test_xarm_driver_units_with_fake_sdk(monkeypatch):
    calls = []

    class API:
        def __init__(self, ip, is_radian=False):
            self.connected, self.error_code, self.warn_code, self.state = True, 0, 0, 0
            self.pos = [400.0, 0.0, 400.0, 180.0, 0.0, 0.0]

        def clean_warn(self): return 0
        def clean_error(self): return 0
        def motion_enable(self, enable): return 0
        def set_mode(self, m): return 0
        def set_gripper_mode(self, m): return 0
        def set_gripper_enable(self, e): return 0
        def set_gripper_speed(self, s): return 0

        def set_state(self, state):
            self.state = state
            return 0

        def get_position(self, is_radian=False):
            assert is_radian is False
            return 0, list(self.pos)

        def set_position(self, **kw):
            calls.append(kw)
            self.pos = [kw["x"], kw["y"], kw["z"], kw["roll"], kw["pitch"], kw["yaw"]]
            return 0

        def get_is_moving(self):
            return False

        def set_gripper_position(self, pos, wait, speed, auto_enable):
            calls.append({"gripper": pos})
            return 0

        def disconnect(self):
            pass

    mod = types.ModuleType("xarm.wrapper")
    mod.XArmAPI = API
    monkeypatch.setitem(sys.modules, "xarm", types.ModuleType("xarm"))
    monkeypatch.setitem(sys.modules, "xarm.wrapper", mod)
    s = cellarm.build_arm(conf(driver="xarm", ip="1.2.3.4", max_speed=0.2, yaw_offset_deg=10))
    await s.drv.connect()
    await s.move_to(0.5, -0.1, 0.25, 30.0, 0.1)
    kw = calls[-1]
    assert (kw["x"], kw["y"], kw["z"]) == pytest.approx((500.0, -100.0, 250.0))   # m -> mm
    assert kw["speed"] == pytest.approx(100.0) and kw["is_radian"] is False         # mm/s, degrees
    assert kw["yaw"] == pytest.approx(40.0) and kw["roll"] == 180.0 and kw["wait"] is False
    p = await s.pose()
    assert (p["x"], p["y"], p["z"], p["yaw"]) == pytest.approx((0.5, -0.1, 0.25, 30.0))
    await s.gripper(0.5)
    assert calls[-1] == {"gripper": 425.0}                                          # 0..850 mm units
    await s.drv.close()
