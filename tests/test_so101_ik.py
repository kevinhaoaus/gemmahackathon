"""SO-101 kinematics (swarmmind/cell/kinematics_so101.py) and the Cartesian so101 cell driver."""
from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmmind.capabilities import so101 as so101_cap  # noqa: E402
from swarmmind.cell import arm as cellarm  # noqa: E402
from swarmmind.cell import kinematics_so101 as K  # noqa: E402
from swarmmind.cell.drivers import so101 as so101_drv  # noqa: E402

KIN = K.SO101Kinematics()
LIM = KIN.limits


def pose_err(kin, q, x, y, z, yaw):
    p = kin.pose(q)
    d = math.dist((p["x"], p["y"], p["z"]), (x, y, z))
    return d, abs(cellarm.wrap_deg(p["yaw"] - yaw)), p["tilt"]


def vertical_q(rng):
    """Random joint vector with the tool pointing straight down, within limits, above the table."""
    dl, de, dw = KIN._d
    while True:
        q = np.array([rng.uniform(*LIM[j]) for j in K.ARM_JOINTS])
        q[3] = (-math.pi / 2 - KIN._beta0 - dl * q[1] - de * q[2]) / dw
        if KIN._within(q) and KIN.check_clear(q) is None and KIN.fk(q)[2, 3] < 0.06:
            return q


# ------------------------------------------------------------------ model / FK
def test_embedded_model_matches_urdf_numbers():
    names = [c.name for c in K.URDF_CHAIN]
    assert names == list(K.ARM_JOINTS) + ["gripper_frame_joint"]
    assert "385e8d7c" in K.URDF_SOURCE
    assert LIM["shoulder_pan"] == (-1.91986, 1.91986) and LIM["wrist_roll"] == (-2.74385, 2.84121)
    # zero pose: upper arm up, forearm and tool straight forward, TCP on the pan plane
    p = KIN.pose(np.zeros(5))
    assert p["x"] == pytest.approx(0.3525, abs=1e-3) and p["z"] == pytest.approx(0.2265, abs=1e-3)
    assert abs(p["y"]) < 1e-4 and p["tilt"] == pytest.approx(90.0, abs=0.1)


def test_urdf_override_from_file(tmp_path):
    def j(c, parent, child):
        kind = "fixed" if c.fixed else "revolute"
        lim = "" if c.fixed else f'<axis xyz="0 0 1"/><limit lower="{c.lower}" upper="{c.upper}" effort="1" velocity="1"/>'
        return (f'<joint name="{c.name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/>'
                f'<origin xyz="{" ".join(map(str, c.xyz))}" rpy="{" ".join(map(str, c.rpy))}"/>{lim}</joint>')
    links = ["base_link", "shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link",
             "gripper_frame_link"]
    body = "".join(f'<link name="{n}"/>' for n in links)
    body += "".join(j(c, links[i], links[i + 1]) for i, c in enumerate(K.URDF_CHAIN))
    body += ('<link name="jaw"/><joint name="gripper" type="revolute"><parent link="gripper_link"/>'
             '<child link="jaw"/><origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>'
             '<limit lower="-0.17" upper="1.74" effort="1" velocity="1"/></joint>')
    f = tmp_path / "so101.urdf"
    f.write_text(f'<robot name="so101">{body}</robot>', encoding="utf-8")
    kin2 = K.SO101Kinematics.from_config({"urdf": str(f)})
    q = np.array([0.3, -0.4, 0.5, 0.6, -0.7])
    assert np.allclose(kin2.fk(q), KIN.fk(q), atol=1e-9)
    longer = f.read_text().replace('xyz="-0.11257 -0.028', 'xyz="-0.12257 -0.028')
    f.write_text(longer, encoding="utf-8")
    kin3 = K.SO101Kinematics.from_config({"urdf": str(f)})
    assert not np.allclose(kin3.fk(q)[:3, 3], KIN.fk(q)[:3, 3], atol=1e-3)
    r = kin3.ik(0.15, 0.03, 0.03, 10.0)                       # IK works on the overridden model
    assert pose_err(kin3, r.q, 0.15, 0.03, 0.03, 10.0)[0] < 1e-3


# ------------------------------------------------------------------ IK accuracy
def test_fk_ik_round_trip_many_targets():
    rng = np.random.default_rng(7)
    worst_mm = worst_deg = 0.0
    n = 0
    for _ in range(60):
        q = vertical_q(rng)
        p = KIN.pose(q)
        r = KIN.ik(p["x"], p["y"], p["z"], p["yaw"])
        d, dy, tilt = pose_err(KIN, r.q, p["x"], p["y"], p["z"], p["yaw"])
        assert d < 1e-3 and dy < 1.0 and tilt < 1.0, (q, p, d, dy, tilt)
        assert KIN._within(r.q)
        worst_mm, worst_deg, n = max(worst_mm, d * 1e3), max(worst_deg, dy, tilt), n + 1
    print(f"\nFK->IK round trip over {n} random top-down poses: worst {worst_mm:.4f} mm, {worst_deg:.4f} deg")
    assert worst_mm < 0.5


def test_ik_grid_inside_workspace():
    ws = so101_drv.SO101_WORKSPACE
    worst = 0.0
    for x in np.linspace(*ws["x"], 4):
        for y in np.linspace(*ws["y"], 5):
            for z in np.linspace(*ws["z"], 3):
                for rel in (-120.0, -45.0, 0.0, 60.0, 135.0):   # yaw relative to the pan direction
                    yaw = cellarm.wrap_deg(math.degrees(math.atan2(y, x)) + rel)
                    r = KIN.ik(float(x), float(y), float(z), yaw)
                    d, dy, tilt = pose_err(KIN, r.q, x, y, z, yaw)
                    assert d < 1e-3 and dy < 1.0 and tilt < 1.0
                    worst = max(worst, d)
    assert worst < 5e-4


# ------------------------------------------------------------------ refusals / limits / yaw
def test_unreachable_reasons():
    with pytest.raises(K.Unreachable) as e:
        KIN.ik(0.40, 0.0, 0.03, 0.0)
    assert e.value.code == "out_of_reach" and "reach" in str(e.value)
    with pytest.raises(K.Unreachable) as e:
        KIN.ik(0.0, 0.0, 0.03, 0.0)
    assert e.value.code == "out_of_reach"
    with pytest.raises(K.Unreachable) as e:
        KIN.ik(0.15, 0.0, -0.01, 0.0)
    assert e.value.code == "below_table" and "table" in str(e.value)
    with pytest.raises(K.Unreachable) as e:
        KIN.ik(0.15, 0.0, 0.20, 0.0)                          # too high for a top-down grasp
    assert e.value.code in ("joint_limit", "out_of_reach")
    with pytest.raises(K.Unreachable) as e:                   # wrist_roll would need ~170 deg
        KIN.ik(0.15, 0.0, 0.03, 170.0)
    assert e.value.code == "joint_limit" and "wrist_roll" in str(e.value)
    with pytest.raises(K.Unreachable) as e:
        KIN.ik(float("nan"), 0.0, 0.03, 0.0)
    assert e.value.code == "invalid"
    raised = K.SO101Kinematics(table_z=0.02)
    with pytest.raises(K.Unreachable) as e:
        raised.ik(0.15, 0.0, 0.01, 0.0)
    assert e.value.code == "below_table"


def test_joint_limits_respected_and_override():
    tight = K.SO101Kinematics.from_config({"limits_deg": {"shoulder_pan": [-20, 20]}})
    r = tight.ik(0.15, 0.03, 0.03, 0.0)
    assert -math.radians(20) - 1e-9 <= r.q[0] <= math.radians(20) + 1e-9
    with pytest.raises(K.Unreachable) as e:
        tight.ik(0.10, 0.12, 0.03, 0.0)                       # needs ~50 deg of pan
    assert e.value.code == "joint_limit" and "shoulder_pan" in str(e.value)
    rng = np.random.default_rng(3)
    for _ in range(40):
        x, y, z, yaw = rng.uniform(0.10, 0.18), rng.uniform(-0.14, 0.14), rng.uniform(0.005, 0.06), rng.uniform(-150, 150)
        try:
            r = KIN.ik(x, y, z, yaw)
        except K.Unreachable as ex:
            assert ex.code == "joint_limit"
            continue
        for j, v in zip(K.ARM_JOINTS, r.q):
            assert LIM[j][0] - 1e-9 <= v <= LIM[j][1] + 1e-9


def test_yaw_wrap_and_roll_relation():
    a = KIN.ik(0.15, 0.05, 0.03, 100.0)
    b = KIN.ik(0.15, 0.05, 0.03, 100.0 - 360.0)
    c = KIN.ik(0.15, 0.05, 0.03, 100.0 + 720.0)
    assert np.allclose(a.q, b.q, atol=1e-6) and np.allclose(a.q, c.q, atol=1e-6)
    assert pose_err(KIN, a.q, 0.15, 0.05, 0.03, -260.0)[1] < 0.05
    # wrist_roll = yaw - pan (pan as the world angle atan2(y, x)), within the TCP-offset wobble
    pan_world = math.degrees(math.atan2(0.05, 0.15))
    assert math.degrees(a.q[4]) == pytest.approx(100.0 - pan_world, abs=4.0)
    # yaw near +-180: the roll range [-157, 163] deg covers it only on one side of the pan
    for y, yaw in ((0.05, 170.0), (0.05, -190.0), (-0.05, -170.0)):
        r = KIN.ik(0.15, y, 0.03, yaw)
        assert pose_err(KIN, r.q, 0.15, y, 0.03, yaw)[1] < 0.05
    with pytest.raises(K.Unreachable, match="wrist_roll"):
        KIN.ik(0.15, -0.05, 0.03, 175.0)


# ------------------------------------------------------------------ ticks
def test_tick_mapping_round_trip():
    cal = {"shoulder_pan": K.JointCal(2048, 1), "shoulder_lift": K.JointCal(1900, -1),
           "elbow_flex": K.JointCal(2200.5, 1), "wrist_flex": K.JointCal(2048, -1), "wrist_roll": K.JointCal(1000, 1)}
    rng = np.random.default_rng(1)
    for _ in range(200):
        q = rng.uniform(-1.6, 1.6, 5)
        t = K.joints_to_ticks(cal, q)
        back = K.ticks_to_joints(cal, t)
        assert np.max(np.abs(back - q)) <= 0.5 * 2 * math.pi / 4096 + 1e-12
        assert K.joints_to_ticks(cal, back) == t
    assert cal["shoulder_pan"].to_ticks(math.radians(90)) == 2048 + 1024
    assert cal["shoulder_lift"].to_ticks(math.radians(90)) == 1900 - 1024
    assert K.DEG_PER_TICK == pytest.approx(0.088, abs=1e-3)
    assert K.TICKS_PER_RAD * math.pi / 180 == pytest.approx(so101_cap.TICKS_PER_DEG)


def test_calibration_formats():
    ours = {"joints": {j: {"home": 2000 + i, "min": 500, "max": 3500, "sign": -1 if i % 2 else 1}
                       for i, j in enumerate(K.ARM_JOINTS)}}
    cal = K.parse_calibration(ours, require_home=True)
    assert cal["shoulder_lift"].sign == -1 and cal["elbow_flex"].home == 2002 and cal["wrist_roll"].max == 3500
    lerobot = {j: {"id": i + 1, "drive_mode": 1 if j == "elbow_flex" else 0, "homing_offset": 0,
                   "range_min": 1000, "range_max": 3000} for i, j in enumerate(K.ARM_JOINTS)}
    cal = K.parse_calibration(lerobot, require_home=True)
    assert cal["shoulder_pan"].home == 2000 and cal["elbow_flex"].sign == -1 and cal["wrist_flex"].sign == 1
    assert cal["shoulder_pan"].rad_limits() == pytest.approx((-1000 / K.TICKS_PER_RAD, 1000 / K.TICKS_PER_RAD))
    with pytest.raises(ValueError, match="home"):
        K.parse_calibration({"joints": {"shoulder_pan": {"min": 1, "max": 2}}}, require_home=True)
    bounds = {j: {"min": 1024, "max": 3072, "home": 2048} for j in K.ARM_JOINTS}
    assert K.parse_calibration({}, bounds=bounds)["wrist_flex"].home == 2048
    kin = K.SO101Kinematics()
    kin.clip_limits(K.parse_calibration({}, bounds=bounds))   # +-90 deg of servo travel
    assert kin.limits["shoulder_pan"] == pytest.approx((-math.pi / 2, math.pi / 2))
    assert kin.limits["elbow_flex"] == pytest.approx((-math.pi / 2, math.pi / 2))


# ------------------------------------------------------------------ workspace report
def test_reachable_top_down_workspace_report():
    """Prints the sampled top-down box (z 0-0.06 m, relative yaw -90/0/+90) and checks the
    driver's default workspace lies inside it. LabMate's table layout must fit in this box."""
    box = K.top_down_workspace(KIN, [0.0, 0.06], step=0.02, yaws=(-90.0, 0.0, 90.0))
    print(f"\nSO-101 top-down reachable box (arm frame): {box}")
    ws = so101_drv.SO101_WORKSPACE
    assert box["x"][0] <= ws["x"][0] and ws["x"][1] <= box["x"][1]
    assert box["y"][0] <= ws["y"][0] and ws["y"][1] <= box["y"][1]
    for x in ws["x"]:                                         # corners at the top and bottom
        for y in ws["y"]:
            for z in ws["z"]:
                for rel in (-90.0, 0.0, 90.0):
                    KIN.ik(x, y, z, math.degrees(math.atan2(y, x)) + rel)
    with pytest.raises(K.Unreachable):                        # nothing top-down above ~9 cm
        KIN.ik(0.15, 0.0, 0.10, 0.0)


# ------------------------------------------------------------------ driver (mock end to end)
MOCK = {"driver": "so101", "mock": True, "rate_hz": 200, "max_step_ticks": 200, "mock_speed_ticks_s": 60000,
        "tolerance_ticks": 4, "settle_s": 0.3, "final_tolerance_ticks": 0, "final_settle_s": 1.0,
        "max_speed": 1.0, "speed": 1.0, "slow_speed": 0.5, "retreat_speed": 0.5, "approach_h": 0.03,
        "max_joint_speed_dps": 3000}


def write_cal(tmp_path) -> str:
    """Calibration whose tick bounds are exactly the URDF limits (home 2048, sign +1)."""
    joints = {}
    for j in K.ARM_JOINTS:
        lo, hi = LIM[j]
        joints[j] = {"home": 2048, "sign": 1, "min": math.ceil(2048 + lo * K.TICKS_PER_RAD),
                     "max": math.floor(2048 + hi * K.TICKS_PER_RAD)}
    joints["gripper"] = {"min": 1500, "max": 2800, "home": 2048}
    f = tmp_path / "so101_cal.json"
    f.write_text(json.dumps({"joints": joints, "poses": {"ready": {"elbow_flex": 2300}}}), encoding="utf-8")
    return str(f)


async def make_arm(**cfg):
    drv = so101_drv.SO101Arm({**MOCK, **cfg})
    await drv.connect()
    return drv, cellarm.Safety(drv, drv.cfg)


@pytest.mark.asyncio
async def test_mock_driver_without_calibration_or_kinematics_stays_named_pose_only():
    drv, safe = await make_arm()
    try:
        assert drv.supports_cartesian is False and "calibration" in drv.info()["cartesian_off"] \
            or "kinematics" in drv.info()["cartesian_off"]
        with pytest.raises(RuntimeError, match="inverse kinematics"):
            await safe.move_to(0.15, 0.0, 0.03)
        assert (await drv.pose())["x"] is None
    finally:
        await drv.close()
    drv, safe = await make_arm(kinematics=True)
    try:
        assert drv.supports_cartesian is True                  # mock + kinematics: default calibration
        assert drv.kin.limits["shoulder_pan"] == pytest.approx((-math.pi / 2, math.pi / 2))
        p = await drv.pose()                                   # mock boots at 2048 = URDF zero
        assert p["x"] == pytest.approx(0.3525, abs=1e-3) and p["z"] == pytest.approx(0.2265, abs=1e-3)
    finally:
        await drv.close()


@pytest.mark.asyncio
async def test_mock_driver_pick_reaches_target(tmp_path):
    drv, safe = await make_arm(calibration=write_cal(tmp_path))
    try:
        assert drv.supports_cartesian and drv.info()["cartesian"] is True
        assert safe.workspace.x == (0.10, 0.18)                # measured SO-101 default
        target = {"x": 0.15, "y": 0.04, "z": 0.012, "yaw": 25.0}
        seen = {}

        async def progress(p):
            if p.get("sub") in ("grip", "retreat"):
                seen[p["sub"]] = await drv.pose()

        r = await cellarm.pick(safe, target["x"], target["y"], target["z"], target["yaw"], progress=progress)
        assert r["steps"] == ["open", "approach", "descend", "grip", "retreat"]
        at = seen["grip"]                                      # measured (FK of the mock's ticks) at grasp
        assert math.dist((at["x"], at["y"], at["z"]), (target["x"], target["y"], target["z"])) < 1.5e-3
        assert abs(cellarm.wrap_deg(at["yaw"] - target["yaw"])) < 1.0 and at["tilt_deg"] < 1.0
        end = await drv.pose()                                 # retreated straight up by approach_h
        assert math.dist((end["x"], end["y"], end["z"]), (target["x"], target["y"], target["z"] + 0.03)) < 1.5e-3
        assert end["gripper"] == pytest.approx(0.0, abs=0.01)
        # an unreachable target inside the workspace box is refused with the IK reason, nothing moves
        before = dict(drv.ctl.present)
        with pytest.raises(K.Unreachable, match="wrist_roll"):
            await safe.move_to(0.15, 0.0, 0.03, 170.0)
        assert drv.ctl.present == before
        # named poses still work
        await drv.named_pose("ready")
        assert abs(drv.ctl.present["elbow_flex"] - 2300) <= 4
    finally:
        await drv.close()


@pytest.mark.asyncio
async def test_driver_plan_speed_cap_and_table_check(tmp_path):
    drv, _ = await make_arm(calibration=write_cal(tmp_path), max_joint_speed_dps=60)
    try:
        q0 = drv.kin.ik(0.12, -0.10, 0.05, 0.0).q
        path, dt = drv.plan(0.17, 0.10, 0.01, 40.0, speed=0.05, q0=q0)
        dist = float(np.linalg.norm(drv.kin.fk(path[-1])[:3, 3] - drv.kin.fk(q0)[:3, 3]))
        total = dt * len(path)
        assert total >= dist / 0.05 - 1e-9                     # Cartesian speed cap
        assert total >= float(np.max(np.abs(path[-1] - q0))) / math.radians(60) - 1e-9   # joint cap
        prev = q0
        for q in path:
            assert float(np.max(np.abs(q - prev))) <= math.radians(3.0) + 1e-9
            assert drv.kin.check_clear(q) is None
            prev = q
        # the table check flags a path through a low posture
        low = drv.kin.ik(0.15, 0.0, 0.01, 0.0).q
        drv.kin.table_z = 0.03
        assert "below the table" in drv._path_problem(q0, [low])
        with pytest.raises(K.Unreachable, match="below the table"):
            drv.plan(0.15, 0.0, 0.01, 0.0, speed=0.05, q0=q0)
        drv.kin.table_z = 0.0
        # if the joint-space path is flagged, the driver falls back to a straight Cartesian path
        calls = []
        real = drv._path_problem

        def flag_first(a, b):
            calls.append(1)
            return "wrist_flex below the table (test)" if len(calls) == 1 else real(a, b)
        drv._path_problem = flag_first
        path, _ = drv.plan(0.17, 0.10, 0.01, 40.0, speed=0.05, q0=q0)
        p0, p1 = drv.kin.fk(q0)[:3, 3], np.array([0.17, 0.10, 0.01])
        for q in path:                                         # every waypoint on the line p0 -> p1
            p = drv.kin.fk(q)[:3, 3]
            off = (p - p0) - np.dot(p - p0, p1 - p0) / np.dot(p1 - p0, p1 - p0) * (p1 - p0)
            assert np.linalg.norm(off) < 1e-3 and drv.kin.pose(q)["tilt"] < 1.0
    finally:
        await drv.close()


@pytest.mark.asyncio
async def test_driver_stop_aborts_move(tmp_path):
    drv, safe = await make_arm(calibration=write_cal(tmp_path), max_joint_speed_dps=30)
    try:
        task = asyncio.create_task(safe.move_to(0.15, 0.0, 0.03, 0.0, speed=0.01))
        await asyncio.sleep(0.3)
        await safe.stop("test stop")
        with pytest.raises(cellarm.MotionAborted):
            await task
        p = await drv.pose()
        assert p["z"] > 0.04                                  # stopped well before the target
    finally:
        await drv.close()
