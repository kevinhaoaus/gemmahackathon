"""Untagged-item blobs (swarmmind/cell/blobs.py) on synthetic scenes (swarmmind/cell/mock_scene.py).

Covers: every item found as exactly one blob (position, long-axis yaw, size), auto-exposure,
tags and tagged fixtures never becoming blobs, stable ids (jitter, motion < 3 cm/frame, dropout),
the no-background fallback, rectify geometry, annotate_marks, per-frame cost, and the
"vision.cell" capability's cell.blobs / sim_truth / background / blobs actions on a fake node.
"""
from __future__ import annotations

import asyncio
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from swarmmind.capabilities import cell_vision  # noqa: E402
from swarmmind.cell import blobs as B  # noqa: E402
from swarmmind.cell import mock_scene  # noqa: E402
from swarmmind.cell import perception as P  # noqa: E402
from swarmmind.tasks import REGISTRY  # noqa: E402

POS_TOL_M = 0.01
YAW_TOL_DEG = 5.0
TAGS_ONLY = {**P.DEFAULT_CELL, "objects": {}}          # table tags only: items can turn freely


def axis_err(a: float, b: float) -> float:
    """Angle between two undirected axes, degrees."""
    return abs((a - b + 90.0) % 180.0 - 90.0)


def random_items(k: int, kinds=("file", "screwdriver", "spring", "superglue", "stepper_motor")) -> list[dict]:
    """The items at random yaws, jittered around spots far enough apart for any yaw."""
    rng = np.random.default_rng(500 + k)
    spots = {"file": (0.20, 0.20), "screwdriver": (0.43, 0.17), "spring": (0.31, 0.055),
             "superglue": (0.31, 0.35), "stepper_motor": (0.50, 0.32)}
    out = []
    for kind in kinds:
        x, y = spots[kind]
        out.append({"item": kind, "table_xy": [x + rng.uniform(-0.008, 0.008), y + rng.uniform(-0.008, 0.008)],
                    "yaw_deg": float(rng.uniform(-180, 180)),
                    "state": ("open" if rng.random() < 0.5 else None) if kind == "superglue" else None})
    return out


def frame(cell, items, seed, **kw):
    jpeg, gt = mock_scene.render(cell, items=items, seed=seed, **kw)
    img = P.to_bgr(jpeg)
    tags = P.detect_tags(img)
    calib = P.Calibration(cell)
    assert calib.update(tags, 0.0), calib.last_error
    return img, calib.H, tags, gt


def pipeline(cell=TAGS_ONLY, background=True, poses=None, **kw) -> B.BlobPipeline:
    pipe = B.BlobPipeline(cell, **kw)
    if background:
        shots = [frame(cell, [], 900 + i, **({"object_poses": poses} if poses else {})) for i in range(5)]
        pipe.capture_background([s[0] for s in shots], [s[1] for s in shots], [s[2] for s in shots])
    return pipe


def check_items(blobs: list[dict], items: list[dict], pos_tol=POS_TOL_M, size_tol=0.007, area_hi=1.4) -> None:
    assert len(blobs) == len(items), (blobs, items)
    used = set()
    for it in items:
        d, j = min((math.dist(b["table_xy"], it["table_xy"]), j) for j, b in enumerate(blobs))
        assert d < pos_tol, (it, blobs[j])
        assert j not in used, "two items on one blob"
        used.add(j)
        b = blobs[j]
        assert abs(b["length_m"] - it["length_m"]) < size_tol, (it, b)
        assert abs(b["width_m"] - it["width_m"]) < size_tol, (it, b)
        assert 0.4 * it["length_m"] * it["width_m"] < b["area_m2"] < area_hi * it["length_m"] * it["width_m"]
        if it["length_m"] / it["width_m"] > 2.0:              # elongated: the long axis counts
            assert axis_err(b["yaw_deg"], it["yaw_deg"]) < YAW_TOL_DEG, (it, b)
        assert -90.0 <= b["yaw_deg"] < 90.0


@pytest.fixture(scope="module")
def bg_pipe_factory():
    shots = [frame(TAGS_ONLY, [], 900 + i) for i in range(5)]

    def make() -> B.BlobPipeline:
        pipe = B.BlobPipeline(TAGS_ONLY)
        pipe.capture_background([s[0] for s in shots], [s[1] for s in shots], [s[2] for s in shots])
        return pipe
    return make


@pytest.mark.parametrize("k", range(10))
def test_every_item_found_once(k, bg_pipe_factory):
    pipe = bg_pipe_factory()
    img, H, tags, gt = frame(TAGS_ONLY, random_items(k), seed=k, warp=0.02 + 0.03 * (k % 3))
    pay = pipe.process(img, H, tags, 1.0)
    assert pay["calibrated"] and pay["background"]
    check_items(pay["_blobs"], gt["items"])
    assert {b["id"] for b in pay["blobs"]} == {b["id"] for b in pay["_blobs"]}


def test_payload_matches_contract(bg_pipe_factory):
    pipe = bg_pipe_factory()
    img, H, tags, _ = frame(TAGS_ONLY, random_items(0), seed=3)
    pub = B.publishable(pipe.process(img, H, tags, 12.5))
    assert set(pub) == {"frame_ts", "calibrated", "background", "blobs"}
    assert pub["frame_ts"] == 12.5
    for b in pub["blobs"]:
        assert set(b) == {"id", "table_xy", "yaw_deg", "length_m", "width_m", "area_m2", "stable_frames"}
        assert isinstance(b["id"], int) and len(b["table_xy"]) == 2 and b["stable_frames"] == 1
    import json
    json.dumps(pub)                                   # topic-ready
    assert B.publishable(pipe.process(img, None, tags, 13.0)) == {
        "frame_ts": 13.0, "calibrated": False, "background": True, "blobs": []}


@pytest.mark.parametrize("gain", [0.75, 1.25])
def test_auto_exposure(gain, bg_pipe_factory):
    pipe = bg_pipe_factory()
    img, H, tags, gt = frame(TAGS_ONLY, random_items(4), seed=11, gain=gain)
    check_items(pipe.process(img, H, tags, 1.0)["_blobs"], gt["items"])


def test_default_simulator_scene():
    """The default cell (tagged servo, slot, tray) with the default items: exactly the 4 items,
    and the tagged fixtures and every tag stay out."""
    pipe = pipeline(P.DEFAULT_CELL)
    img, H, tags, gt = frame(P.DEFAULT_CELL, None, seed=21)
    pay = pipe.process(img, H, tags, 1.0)
    check_items(pay["_blobs"], gt["items"])
    assert sorted(B.match_truth(pay["blobs"], gt["items"]).values()) == [
        "file", "screwdriver", "spring", "superglue:open"]
    assert sorted(it["item"] for it in mock_scene.DEFAULT_ITEMS) == ["file", "screwdriver", "spring", "superglue"]
    assert set(gt["visible_tags"]) == {0, 1, 2, 3, 10, 11, 20}


def test_labmate_fixtures_default_poses():
    cell = {**P.DEFAULT_CELL, "objects": {
        "safe_tray": {"tag": 30, "kind": "fixture", "offset": [0.07, 0.0, 0.0]},
        "handover": {"tag": 31, "kind": "fixture", "offset": [0.07, 0.0, 0.0]}}}
    pipe = pipeline(cell)
    img, H, tags, gt = frame(cell, None, seed=5)
    assert {t["id"] for t in tags} == {0, 1, 2, 3, 30, 31}
    check_items(pipe.process(img, H, tags, 1.0)["_blobs"], gt["items"])


def test_tags_never_become_blobs():
    """A tagged part moved after the background (its old and new spots), all tags, items=[]."""
    pipe = pipeline(P.DEFAULT_CELL)
    moved = {"servo_1": [0.30, 0.10, 40.0]}
    for seed in range(3):
        img, H, tags, gt = frame(P.DEFAULT_CELL, [], seed, object_poses=moved)
        pay = pipe.process(img, H, tags, float(seed))
        assert pay["blobs"] == [], pay["blobs"]
    # with items present, no blob sits on a tag
    img, H, tags, gt = frame(P.DEFAULT_CELL, None, 7, object_poses=moved)
    blobs = pipe.process(img, H, tags, 9.0)["blobs"]
    assert len(blobs) == 4
    for t in gt["tags"].values():
        assert all(math.dist(b["table_xy"], t["table_xy"]) > 0.04 for b in blobs)


def test_empty_table_no_blobs_despite_camera_jitter(bg_pipe_factory):
    pipe = bg_pipe_factory()
    for seed in (1, 2, 3):          # each seed moves the mock camera a few pixels
        img, H, tags, _ = frame(TAGS_ONLY, [], seed, warp=0.08)
        assert pipe.process(img, H, tags, float(seed))["blobs"] == []


def test_ids_stable_under_jitter_and_motion(bg_pipe_factory):
    pipe = bg_pipe_factory()
    items = random_items(2)
    ids = None
    rng = np.random.default_rng(3)
    for f in range(4):              # barely moving: same ids, stable_frames counts up
        its = [{**it, "table_xy": [it["table_xy"][0] + rng.uniform(-0.002, 0.002),
                                   it["table_xy"][1] + rng.uniform(-0.002, 0.002)],
                "yaw_deg": it["yaw_deg"] + rng.uniform(-2, 2)} for it in items]
        img, H, tags, gt = frame(TAGS_ONLY, its, seed=30 + f)
        pay = pipe.process(img, H, tags, 10.0 + 0.5 * f)
        by_item = {it["item"]: min(pay["blobs"], key=lambda b: math.dist(b["table_xy"], it["table_xy"]))["id"]
                   for it in gt["items"]}
        assert ids is None or by_item == ids
        ids = by_item
        assert all(b["stable_frames"] == f + 1 for b in pay["blobs"])
    # the stepper motor slides 2.5 cm per frame along -x: keeps its id
    motor = next(it for it in items if it["item"] == "stepper_motor")
    for f in range(3):
        motor["table_xy"] = [motor["table_xy"][0] - 0.025, motor["table_xy"][1]]
        img, H, tags, gt = frame(TAGS_ONLY, items, seed=40 + f)
        pay = pipe.process(img, H, tags, 12.0 + 0.5 * f)
        b = min(pay["blobs"], key=lambda b: math.dist(b["table_xy"], motor["table_xy"]))
        assert math.dist(b["table_xy"], motor["table_xy"]) < POS_TOL_M
        assert b["id"] == ids["stepper_motor"]
    assert len(pay["blobs"]) == len(items)


def test_tracker_dropout_and_gate():
    tr = B.Tracker()
    a = {"table_xy": [0.1, 0.1]}
    assert tr.update([a], 0.0)[0]["id"] == 1
    assert tr.update([], 0.5) == []                       # hidden by a hand for 0.5 s
    back = tr.update([{"table_xy": [0.11, 0.1]}], 0.9)[0]
    assert back["id"] == 1 and back["stable_frames"] == 1
    assert tr.update([{"table_xy": [0.12, 0.1]}], 1.0)[0]["stable_frames"] == 2
    assert tr.update([], 1.5) == []
    assert tr.update([{"table_xy": [0.12, 0.1]}], 2.6)[0]["id"] == 2    # gone > 1 s: new id
    assert tr.update([{"table_xy": [0.16, 0.1]}], 2.7)[0]["id"] == 3    # jumped 4 cm: new id
    two = tr.update([{"table_xy": [0.16, 0.1]}, {"table_xy": [0.18, 0.1]}], 2.8)
    assert [b["id"] for b in two] == [3, 4]


def test_no_background_fallback():
    for cell in (TAGS_ONLY, P.DEFAULT_CELL):
        pipe = pipeline(cell, background=False)
        items = random_items(6) if cell is TAGS_ONLY else None
        img, H, tags, gt = frame(cell, items, seed=8)
        pay = pipe.process(img, H, tags, 1.0)
        assert pay["calibrated"] and pay["background"] is False
        check_items(pay["_blobs"], gt["items"], size_tol=0.01, area_hi=1.6)
        img, H, tags, _ = frame(cell, [], seed=9)
        assert pipe.process(img, H, tags, 2.0)["blobs"] == []      # tags and fixtures are not blobs


def test_rectify_geometry():
    # pixel (u, v) <-> table (u / 1000, v / 1000): a bright dot at pixel (100, 50) is table
    # (0.10, 0.05) and lands on rectified pixel (99.5, 149.5) of the extent (0, 0, 0.3, 0.2)
    img = np.zeros((300, 400, 3), np.uint8)
    cv2.circle(img, (100, 50), 3, (255, 255, 255), -1)
    H = np.diag([0.001, 0.001, 1.0])
    r = B.rectify(img, H, (0.0, 0.0, 0.3, 0.2))
    assert r.shape == (200, 300, 3)
    m = cv2.moments(r[..., 0], binaryImage=False)
    assert m["m10"] / m["m00"] == pytest.approx(99.5, abs=0.3)
    assert m["m01"] / m["m00"] == pytest.approx(149.5, abs=0.3)
    assert B.rectify(img, H, (0.3, 0.2)).shape == r.shape
    assert B.table_extent(P.DEFAULT_CELL) == pytest.approx((-0.03, -0.03, 0.63, 0.43))


def test_annotate_marks(bg_pipe_factory):
    pipe = bg_pipe_factory()
    img, H, tags, _ = frame(TAGS_ONLY, random_items(1), seed=2)
    pay = pipe.process(img, H, tags, 1.0)
    calib = SimpleNamespace(H=H)
    for src in (cv2.imencode(".jpg", img)[1].tobytes(), img):
        out = B.annotate_marks(src, pay["_blobs"], calib, background=True)
        assert out[:2] == b"\xff\xd8" and out[-2:] == b"\xff\xd9"
        dec = P.to_bgr(out)
        assert dec.shape == img.shape
        assert float(np.abs(dec.astype(int) - img.astype(int)).mean()) > 1.0     # marks drawn
    # published blobs (no table box) draw too; uncalibrated draws only the status line
    assert B.annotate_marks(img, pay["blobs"], H)[:2] == b"\xff\xd8"
    assert B.annotate_marks(img, [], None)[:2] == b"\xff\xd8"


def test_match_truth():
    blobs = [{"id": 3, "table_xy": [0.50, 0.20]}, {"id": 7, "table_xy": [0.20, 0.20]},
             {"id": 9, "table_xy": [0.40, 0.40]}]
    items = [{"item": "superglue", "state": "open", "table_xy": [0.505, 0.20]},
             {"item": "file", "state": None, "table_xy": [0.21, 0.20]},
             {"item": "spring", "state": None, "table_xy": [0.10, 0.10]}]
    assert B.match_truth(blobs, items) == {"3": "superglue:open", "7": "file"}


def test_per_frame_cost_at_720p(bg_pipe_factory):
    pipe = bg_pipe_factory()
    img, H, tags, _ = frame(TAGS_ONLY, random_items(0), seed=4)
    assert img.shape[:2] == (720, 1280)
    pipe.process(img, H, tags, 0.0)                         # warm up
    times = []
    for i in range(7):
        t0 = time.perf_counter()
        pay = pipe.process(img, H, tags, 1.0 + i)
        B.annotate_marks(img, pay["_blobs"], H, True)
        times.append(time.perf_counter() - t0)
    assert float(np.median(times)) < 0.05, times


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

    def latest(self, topic):
        return next((d for t, d in reversed(self.published) if t == topic), None)


async def wait_for(cond, timeout=8.0):
    t = asyncio.get_running_loop().time()
    while not cond():
        assert asyncio.get_running_loop().time() - t < timeout, "timed out"
        await asyncio.sleep(0.02)


async def progress(_):
    pass


@pytest.mark.asyncio
async def test_capability_mock_blobs_and_sim_truth():
    node = FakeNode({"source": "mock", "rate_hz": 20}, P.DEFAULT_CELL)
    await cell_vision.setup(node)
    try:
        await wait_for(lambda: node.latest("cell.blobs") is not None)
        pay = node.latest("cell.blobs")
        assert pay["calibrated"] and pay["background"] is True        # auto-captured in mock
        assert sorted(pay["sim_truth"].values()) == ["file", "screwdriver", "spring", "superglue:open"]
        assert set(pay["sim_truth"]) == {str(b["id"]) for b in pay["blobs"]}
        assert node.cell_blobs is not None and node.cell_marked_jpeg[:2] == b"\xff\xd8"
        assert node.published[-1][0] == "cell.state"                   # blobs first, then state
        handler = REGISTRY["vision.cell"]
        got = await handler({"action": "blobs"}, progress)
        assert got["blobs"] and "sim_truth" in got
        glue_id = next(int(k) for k, v in pay["sim_truth"].items() if v == "superglue:open")
        # the arm carries the glue 2 cm per frame: same id; capping it changes the truth label
        for step in range(1, 4):
            await cell_vision._on_mock_poses({"data": {"superglue": [0.53 - 0.02 * step, 0.20, 0]}})
            n = len(node.published)
            await wait_for(lambda: len(node.published) >= n + 4)
        await cell_vision._on_mock_poses({"data": {"superglue": {"state": None}}})
        n = len(node.published)
        await wait_for(lambda: len(node.published) >= n + 4)
        pay = node.latest("cell.blobs")
        glue = next(b for b in pay["blobs"] if math.dist(b["table_xy"], [0.47, 0.20]) < POS_TOL_M)
        assert glue["id"] == glue_id and pay["sim_truth"][str(glue_id)] == "superglue"
        # moving a tagged part re-renders the mock background: still exactly the 4 items
        await cell_vision._on_mock_poses({"data": {"servo_1": [0.30, 0.12, 30]}})
        n = len(node.published)
        await wait_for(lambda: len(node.published) >= n + 4)
        assert len(node.latest("cell.blobs")["blobs"]) == 4
        bg = await handler({"action": "background"}, progress)
        assert bg["background"] and bg["frames"] == cell_vision.MOCK_BG_FRAMES
        await cell_vision._on_mock_poses({"data": {"spring": None}})      # removed from the scene
        n = len(node.published)
        await wait_for(lambda: len(node.published) >= n + 4)
        assert sorted(node.latest("cell.blobs")["sim_truth"].values()) == ["file", "screwdriver", "superglue"]
    finally:
        await cell_vision.shutdown()


@pytest.mark.asyncio
async def test_capability_real_camera_background_and_no_sim_truth():
    empty, _ = mock_scene.render(items=[])
    full, gt = mock_scene.render(seed=3)
    scene = {"jpeg": empty}
    node = FakeNode({"source": "node", "rate_hz": 20, "background_s": 0.3}, P.DEFAULT_CELL,
                    frames=lambda: scene["jpeg"])
    await cell_vision.setup(node)
    try:
        await wait_for(lambda: node.latest("cell.blobs") is not None)
        assert node.latest("cell.blobs")["background"] is False           # nothing captured yet
        bg = await REGISTRY["vision.cell"]({"action": "background"}, progress)
        assert bg["background"] and bg["frames"] >= 2
        scene["jpeg"] = full
        await wait_for(lambda: (node.latest("cell.blobs") or {}).get("background") and
                       len(node.latest("cell.blobs")["blobs"]) == 4)
        pay = node.latest("cell.blobs")
        assert "sim_truth" not in pay
        check_items(pay["blobs"], gt["items"])
    finally:
        await cell_vision.shutdown()


@pytest.mark.asyncio
async def test_capability_background_needs_calibration():
    blank = cv2.imencode(".jpg", np.full((720, 1280, 3), 128, np.uint8))[1].tobytes()
    node = FakeNode({"source": "node", "rate_hz": 20, "background_s": 0.1}, P.DEFAULT_CELL,
                    frames=lambda: blank)
    await cell_vision.setup(node)
    try:
        with pytest.raises(RuntimeError, match="calibrated"):
            await REGISTRY["vision.cell"]({"action": "background"}, progress)
        await wait_for(lambda: node.latest("cell.blobs") is not None)
        assert node.latest("cell.blobs") == {"frame_ts": node.latest("cell.blobs")["frame_ts"],
                                             "calibrated": False, "background": False, "blobs": []}
    finally:
        await cell_vision.shutdown()
