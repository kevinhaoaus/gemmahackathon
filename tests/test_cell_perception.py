"""Cell perception on synthetic AprilTag frames (swarmmind/cell/mock_scene.py): no camera, no cluster.

Covers object position / yaw accuracy over many poses and camera warps, table <-> arm round trips,
missing and occluded tags, the < 4 table tags rule, the calibration hold + staleness, annotate,
and the "vision.cell" capability loop + task actions on a fake node.
"""
from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from swarmmind.capabilities import cell_vision  # noqa: E402
from swarmmind.cell import mock_scene  # noqa: E402
from swarmmind.cell import perception as P  # noqa: E402
from swarmmind.tasks import REGISTRY  # noqa: E402

POS_TOL_M = 0.005
YAW_TOL_DEG = 3.0


def yaw_err(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def poses_for(k: int) -> dict:
    """Deterministic, non-overlapping object poses spread over the table."""
    rng = np.random.default_rng(100 + k)
    return {"servo_1": [0.10 + rng.uniform(0, 0.10), 0.09 + rng.uniform(0, 0.08), rng.uniform(-180, 180)],
            "so101_slot_1": [0.38 + rng.uniform(0, 0.12), 0.10 + rng.uniform(0, 0.14), rng.uniform(-180, 180)],
            "tray_a": [0.13 + rng.uniform(0, 0.12), 0.28 + rng.uniform(0, 0.03), rng.uniform(-180, 180)]}


def run(jpeg: bytes, calib: P.Calibration | None = None, ts: float = 1000.0) -> dict:
    return P.process(jpeg, calib or P.Calibration(), "test", frame_ts=ts)


@pytest.mark.parametrize("k", range(12))
def test_object_accuracy(k):
    jpeg, gt = mock_scene.render(None, poses_for(k), seed=k, warp=0.02 + 0.03 * (k % 4))
    st = run(jpeg)
    assert st["calibrated"] and st["reproj_err_px"] < 1.5
    for o in st["objects"]:
        g = gt["objects"][o["name"]]
        assert o["visible"], o["name"]
        assert math.dist(o["table"][:2], g["table"][:2]) < POS_TOL_M, (o, g)
        assert o["table"][2] == pytest.approx(g["table"][2])
        assert yaw_err(o["yaw_deg"], g["yaw_deg"]) < YAW_TOL_DEG, (o, g)
    for t in st["tags"]:   # every tag, table tags included, lands where it was drawn
        g = gt["tags"][t["id"]]
        assert math.dist(t["table_xy"], g["table_xy"]) < POS_TOL_M
        assert yaw_err(t["yaw_deg"], g["yaw_deg"]) < YAW_TOL_DEG


def test_state_matches_contract():
    jpeg, _ = mock_scene.render()
    st = run(jpeg)
    assert set(st) == {"frame_ts", "camera", "calibrated", "tags", "objects", "reproj_err_px"}
    assert {"id", "px", "corners", "table_xy", "yaw_deg"} == set(st["tags"][0])
    assert {"name", "kind", "tag", "visible", "table", "yaw_deg"} == set(st["objects"][0])
    assert len(st["tags"][0]["corners"]) == 4
    assert [o["name"] for o in st["objects"]] == list(P.DEFAULT_CELL["objects"])


def test_string_keys_from_json_config():
    cell = {**P.DEFAULT_CELL, "table_tags": {str(k): v for k, v in P.DEFAULT_CELL["table_tags"].items()}}
    jpeg, gt = mock_scene.render(cell)
    st = run(jpeg, P.Calibration(cell))
    assert st["calibrated"]
    servo = next(o for o in st["objects"] if o["name"] == "servo_1")
    assert math.dist(servo["table"][:2], gt["objects"]["servo_1"]["table"][:2]) < POS_TOL_M


@pytest.mark.parametrize("arm", [{"base": [0.30, -0.25, 90.0, 0.0]}, {"base": [-0.1, 0.7, -135.0, 0.05]},
                                 [0.2, 0.1, 30.0]])
def test_table_arm_round_trip(arm):
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = rng.uniform(-1, 1, 3)
        q = P.table_to_arm(p, arm)
        assert np.allclose(P.arm_to_table(q, arm), p, atol=1e-12)
        assert np.allclose(P.table_to_arm(P.arm_to_table(p, arm), arm), p, atol=1e-12)
        assert math.dist(p[:2], P._base(arm)[:2]) == pytest.approx(math.hypot(q[0], q[1]))
    yaw = 37.0
    assert P.yaw_arm_to_table(P.yaw_table_to_arm(yaw, arm), arm) == pytest.approx(yaw)


def test_table_to_arm_known_values():
    arm = P.DEFAULT_CELL["arms"]["arm.big"]          # base at (0.30, -0.25), arm +x = table +y
    assert np.allclose(P.table_to_arm([0.30, -0.25, 0.0], arm), [0, 0, 0], atol=1e-12)
    assert np.allclose(P.table_to_arm([0.30, 0.15, 0.02], arm), [0.40, 0.0, 0.02], atol=1e-12)
    assert np.allclose(P.table_to_arm([0.40, -0.25, 0.0], arm), [0.0, -0.10, 0.0], atol=1e-12)
    assert np.allclose(P.table_to_arm([0, 0, 0.1], {"base": [0, 0, 0, 0.05]}), [0, 0, 0.05])
    assert P.yaw_table_to_arm(0.0, arm) == pytest.approx(-90.0)


def test_hidden_and_occluded_objects_not_visible():
    jpeg, gt = mock_scene.render(hide=[10], occlude=[11])
    st = run(jpeg)
    assert st["calibrated"]
    by = {o["name"]: o for o in st["objects"]}
    assert not by["servo_1"]["visible"] and by["servo_1"]["table"] is None
    assert not by["tray_a"]["visible"] and by["tray_a"]["table"] is None
    assert by["so101_slot_1"]["visible"]
    assert {t["id"] for t in st["tags"]} == set(gt["visible_tags"])


@pytest.mark.parametrize("gone", [[0], [2], [1, 3], [0, 1, 2, 3]])
def test_fewer_than_four_table_tags_not_calibrated(gone):
    jpeg, _ = mock_scene.render(occlude=gone[:1], hide=gone[1:])
    st = run(jpeg)
    assert not st["calibrated"]
    assert st["reproj_err_px"] is None
    for o in st["objects"]:
        assert o["table"] is None and o["yaw_deg"] is None
    for t in st["tags"]:
        assert t["table_xy"] is None
    seen = {t["id"] for t in st["tags"]}
    assert not seen & set(gone)
    assert {o["name"] for o in st["objects"] if o["visible"]} == {"servo_1", "so101_slot_1", "tray_a"}


def test_calibration_held_then_stale():
    calib = P.Calibration(stale_s=2.0)
    full, _ = mock_scene.render(seed=1)
    covered, gt = mock_scene.render(seed=2, occlude=[3])
    assert run(full, calib, ts=100.0)["calibrated"]
    held = run(covered, calib, ts=101.5)       # one table tag under a hand: last fit still good
    assert held["calibrated"] and calib.fit_ts == 100.0
    servo = next(o for o in held["objects"] if o["name"] == "servo_1")
    assert math.dist(servo["table"][:2], gt["objects"]["servo_1"]["table"][:2]) < POS_TOL_M
    stale = run(covered, calib, ts=102.5)      # still covered, past stale_s: not calibrated
    assert not stale["calibrated"] and all(o["table"] is None for o in stale["objects"])
    assert run(full, calib, ts=103.0)["calibrated"] and calib.fit_ts == 103.0


def test_rotated_table_tag_falls_back_to_centres():
    # the config says yaw 0 but the scene has tag 2 mounted at 90 deg: corner fit disagrees,
    # the centre-only fit still calibrates and objects stay accurate
    scene = {**P.DEFAULT_CELL, "table_tags": {**P.DEFAULT_CELL["table_tags"], 2: [0.60, 0.40, 90.0]}}
    jpeg, gt = mock_scene.render(scene)
    calib = P.Calibration(P.DEFAULT_CELL)
    st = run(jpeg, calib)
    assert st["calibrated"] and not calib.corners_used
    servo = next(o for o in st["objects"] if o["name"] == "servo_1")
    assert math.dist(servo["table"][:2], gt["objects"]["servo_1"]["table"][:2]) < POS_TOL_M


def test_detect_tags_on_gray_and_garbage():
    jpeg, gt = mock_scene.render()
    gray = P.to_bgr(jpeg)[..., 1].copy()
    assert {t["id"] for t in P.detect_tags(gray)} == set(gt["visible_tags"])
    assert P.detect_tags(np.full((480, 640, 3), 128, np.uint8)) == []
    with pytest.raises(ValueError):
        P.detect_tags(b"not a jpeg")


def test_annotate_returns_jpeg():
    jpeg, _ = mock_scene.render(hide=[11])
    for st in (run(jpeg), run(mock_scene.render(hide=[0, 1])[0])):
        out = P.annotate(jpeg, st)
        assert out[:2] == b"\xff\xd8" and out[-2:] == b"\xff\xd9"
        img = P.to_bgr(out)
        assert img.shape == P.to_bgr(jpeg).shape
    raw = P.to_bgr(jpeg)
    assert P.annotate(raw, {"calibrated": False, "tags": [], "objects": []})[:2] == b"\xff\xd8"


# ---------------------------------------------------------------- capability on a fake node

class FakeNode:
    def __init__(self, conf: dict, cell: dict | None = None, frames=None):
        cc = {"vision.cell": conf}
        if cell is not None:
            cc["cell"] = cell
        self.cfg = SimpleNamespace(node_id="cam", capabilities=["vision.cell"], capability_config=cc,
                                   camera_url="usb:0")
        self.published: list[tuple[str, dict]] = []
        self.events: list[tuple[str, str]] = []
        self._frames = frames

        async def capture_frame():
            return self._frames() if self._frames else None
        self.vision = SimpleNamespace(capture_frame=capture_frame)

    async def publish(self, topic, data):
        self.published.append((topic, data))
        return {"topic": topic, "data": data}

    def event(self, msg, level="info", **_):
        self.events.append((level, msg))


async def wait_for(cond, timeout=5.0):
    t = asyncio.get_running_loop().time()
    while not cond():
        assert asyncio.get_running_loop().time() - t < timeout, "timed out"
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_capability_mock_loop_and_actions():
    node = FakeNode({"source": "mock", "rate_hz": 20}, P.DEFAULT_CELL)
    await cell_vision.setup(node)
    try:
        await wait_for(lambda: len(node.published) >= 2)
        topic, state = node.published[-1]
        assert topic == "cell.state" and state["calibrated"] and state["camera"] == "cam"
        assert node.cell_state is not None and node.cell_annotated_jpeg[:2] == b"\xff\xd8"
        handler = REGISTRY["vision.cell"]

        async def progress(_):
            pass
        snap = await handler({"action": "snapshot"}, progress)
        assert {o["name"] for o in snap["objects"] if o["visible"]} == set(P.DEFAULT_CELL["objects"])
        cal = await handler({"action": "calibrate"}, progress)
        assert cal["calibrated"] and cal["fresh_fit"] and cal["corners_used"]
        assert cal["table_tags_seen"] == [0, 1, 2, 3] and cal["table_tags_missing"] == []
        assert 10 in cal["tags_seen"] and cal["reproj_err_px"] < 1.5
        # moving a part in the synthetic scene shows up in cell.state
        await cell_vision._on_mock_poses({"topic": "cell.mock.poses", "data": {"servo_1": [0.45, 0.30, 45]}})
        await wait_for(lambda: any(o["name"] == "servo_1" and o["table"] and
                                   math.dist(o["table"][:2], [0.45, 0.30]) < POS_TOL_M
                                   for o in node.cell_state["objects"]))
        with pytest.raises(ValueError):
            await handler({"action": "explode"}, progress)
    finally:
        await cell_vision.shutdown()


@pytest.mark.asyncio
async def test_capability_survives_no_camera_and_logs_once():
    node = FakeNode({"source": "node", "rate_hz": 50})           # no cell config: example cell
    await cell_vision.setup(node)
    try:
        await asyncio.sleep(0.4)
        assert node.published == []
        warnings = [m for lvl, m in node.events if lvl == "warning" and "no camera frame" in m]
        assert len(warnings) == 1
        assert any("example cell" in m for _, m in node.events)
        assert cell_vision._CV.task is not None and not cell_vision._CV.task.done()
        async def progress(_):
            pass
        with pytest.raises(RuntimeError):
            await REGISTRY["vision.cell"]({"action": "calibrate"}, progress)
    finally:
        await cell_vision.shutdown()


@pytest.mark.asyncio
async def test_capability_node_source_with_bad_frames():
    good, _ = mock_scene.render()
    frames = iter([b"\xff\xd8garbage", good] * 1000)
    node = FakeNode({"source": "node", "rate_hz": 50}, P.DEFAULT_CELL, frames=lambda: next(frames))
    await cell_vision.setup(node)
    try:
        await wait_for(lambda: len(node.published) >= 3)
        assert all(s["calibrated"] for _, s in node.published)
        errors = [m for lvl, m in node.events if lvl == "warning" and "failed" in m]
        assert 1 <= len(errors) <= len(node.published) + 1
    finally:
        await cell_vision.shutdown()


@pytest.mark.asyncio
async def test_capability_inactive_without_capability():
    node = FakeNode({"source": "mock"})
    node.cfg.capabilities = []
    await cell_vision.setup(node)
    assert cell_vision._CV is None and not hasattr(node, "cell_state")
