"""Cell camera: capability "vision.cell" (AprilTag perception -> topic "cell.state").

Only active on a node whose config lists "vision.cell" in `capabilities`. The cell layout comes
from `capability_config["cell"]` (docs/ROBOT_CELL.md; without it the doc's example cell is used,
with a warning). Settings live in `capability_config["vision.cell"]`:

    source: node | mock      node: this node's camera (node.vision.capture_frame, e.g. a Brio on
                             camera_url "usb:0?w=1920&h=1080&fps=30"); mock: a synthetic tag scene
    rate_hz: 2               frames processed per second
    stale_s: 3.0             hold the last calibration this long while table tags are hidden
    camera: ""               name in cell.state (default: this node's id)
    mock_poses: {servo_1: [x, y, yaw_deg], ...}   mock scene object poses (table frame)

The loop sets `node.cell_state` (latest cell.state dict) and `node.cell_annotated_jpeg`, and
publishes "cell.state". In mock mode, publishing {name: [x, y, yaw_deg]} to topic
"cell.mock.poses" moves objects in the synthetic scene (the simulated arm can "carry" a part).

Task actions (capability "vision.cell"):
    {"action": "snapshot"}     -> the current cell.state
    {"action": "calibrate"}    -> a fresh fit on a new frame: {calibrated, reproj_err_px,
                                  table_tags_seen, table_tags_missing, tags_seen, corners_used, error}
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from ..logutil import get_logger
from ..tasks import register_capability
from ..topics import subscribe

if TYPE_CHECKING:
    from ..agent import Node

log = get_logger("cell_vision")

try:        # OpenCV is only needed on the camera node; elsewhere this module just stays inactive
    from ..cell import mock_scene
    from ..cell import perception as P
    _IMPORT_ERROR = ""
except ImportError as e:  # pragma: no cover - depends on the machine
    mock_scene = P = None  # type: ignore[assignment]
    _IMPORT_ERROR = str(e)

CAP = "vision.cell"
TOPIC = "cell.state"
MOCK_TOPIC = "cell.mock.poses"


class CellVision:
    def __init__(self, node: "Node", conf: dict, cell: dict):
        self.node = node
        self.source = str(conf.get("source", "node")).lower()
        if self.source not in ("node", "mock"):
            raise ValueError(f"{CAP}: source must be node | mock, not {self.source!r}")
        self.period = 1.0 / max(float(conf.get("rate_hz", 2.0)), 0.1)
        self.camera = str(conf.get("camera") or node.cfg.node_id)
        self.cell = P.normalize_cell(cell)
        self.calib = P.Calibration(cell, stale_s=float(conf.get("stale_s", P.STALE_S)))
        self.mock_poses = {k: list(v) for k, v in (conf.get("mock_poses") or {}).items()}
        self.frames = 0
        self.failing = ""          # current failure kind, logged once until it recovers
        self.task: asyncio.Task | None = None
        self.lock = asyncio.Lock()  # one frame through the pipeline at a time (loop vs. calibrate)

    async def grab(self) -> tuple[bytes | None, float]:
        ts = time.time()
        if self.source == "mock":
            self.frames += 1
            jpeg, _ = await asyncio.to_thread(mock_scene.render, self.cell, self.mock_poses,
                                              seed=self.frames % 1000)
            return jpeg, ts
        return await self.node.vision.capture_frame(), ts

    def _step(self, jpeg: bytes, ts: float) -> tuple[dict, list[dict], bytes]:
        tags = P.detect_tags(jpeg)
        self.calib.update(tags, ts)
        state = P.build_state(tags, self.calib, ts, self.camera)
        return state, tags, P.annotate(jpeg, state, self.cell)

    async def once(self) -> tuple[dict, list[dict]] | None:
        async with self.lock:
            jpeg, ts = await self.grab()
            if not jpeg:
                self._fail("no_frame", f"{CAP}: no camera frame (source {self.source}, "
                                       f"camera_url {self.node.cfg.camera_url!r})")
                return None
            state, tags, annotated = await asyncio.to_thread(self._step, jpeg, ts)
        self.node.cell_state, self.node.cell_annotated_jpeg = state, annotated
        await self.node.publish(TOPIC, state)
        if not state["calibrated"]:
            self._fail("uncalibrated", f"{CAP}: not calibrated ({self.calib.last_error or 'no fit yet'})")
        elif self.failing:
            self.node.event(f"{CAP}: ok again (calibrated, reprojection error {state['reproj_err_px']} px)")
            self.failing = ""
        return state, tags

    def _fail(self, kind: str, msg: str) -> None:
        if self.failing != kind:              # log each kind of trouble once, not every frame
            self.failing = kind
            self.node.event(msg, level="warning")

    async def run(self) -> None:
        while True:
            t0 = time.monotonic()
            try:
                await self.once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - perception must never take the node down
                if self.failing != "error":
                    log.exception("%s: frame processing failed", CAP)
                self._fail("error", f"{CAP}: frame processing failed: {e!r}")
            await asyncio.sleep(max(0.0, self.period - (time.monotonic() - t0)))


_CV: CellVision | None = None
_SUBSCRIBED = False


async def setup(node: "Node") -> None:
    global _CV, _SUBSCRIBED
    if CAP not in node.cfg.capabilities:
        return
    if P is None:
        raise RuntimeError(f"{CAP} needs OpenCV with cv2.aruco: {_IMPORT_ERROR}")
    conf = dict(node.cfg.capability_config.get(CAP, {}) or {})
    cell = node.cfg.capability_config.get("cell")
    if not cell:
        node.event(f"{CAP}: no capability_config.cell; using the example cell from docs/ROBOT_CELL.md",
                   level="warning")
        cell = P.DEFAULT_CELL
    node.cell_state, node.cell_annotated_jpeg = None, None
    _CV = CellVision(node, conf, cell)
    _CV.task = asyncio.create_task(_CV.run())
    if not _SUBSCRIBED:
        subscribe(MOCK_TOPIC, _on_mock_poses)
        _SUBSCRIBED = True
    node.event(f"{CAP} ready on {node.cfg.node_id}: source {_CV.source}, {1 / _CV.period:g} Hz, "
               f"{len(_CV.cell['table_tags'])} table tags, {len(_CV.cell['objects'])} objects")


async def shutdown() -> None:
    global _CV
    if _CV is not None and _CV.task is not None:
        _CV.task.cancel()
        try:
            await _CV.task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    _CV = None


async def _on_mock_poses(sample: dict) -> None:
    data = sample.get("data")
    if _CV is not None and _CV.source == "mock" and isinstance(data, dict):
        for name, pose in data.items():
            if name in _CV.cell["objects"] and isinstance(pose, (list, tuple)) and len(pose) >= 2:
                _CV.mock_poses[name] = [float(v) for v in pose[:3]] + [0.0] * (3 - len(pose[:3]))


@register_capability(CAP)
async def cell_vision(args: dict, progress) -> Any:
    if _CV is None:
        raise RuntimeError(f"{CAP} is not set up on this node (add it to capabilities)")
    action = args.get("action", "snapshot")
    if action == "snapshot":
        state = _CV.node.cell_state
        if state is None:                     # nothing processed yet: take one frame now
            res = await _CV.once()
            state = res[0] if res else None
        if state is None:
            raise RuntimeError(f"{CAP}: no frame yet")
        return state
    if action == "calibrate":
        await progress({"step": "capturing"})
        res = await _CV.once()
        if res is None:
            raise RuntimeError(f"{CAP}: no camera frame")
        state, tags = res
        seen = sorted(t["id"] for t in tags)
        table = sorted(_CV.cell["table_tags"])
        fresh = _CV.calib.fit_ts == state["frame_ts"]
        return {"calibrated": state["calibrated"], "fresh_fit": fresh,
                "reproj_err_px": state["reproj_err_px"],
                "table_tags_seen": [i for i in table if i in seen],
                "table_tags_missing": [i for i in table if i not in seen],
                "tags_seen": seen, "corners_used": _CV.calib.corners_used if fresh else None,
                "error": "" if fresh else (_CV.calib.last_error or "fit failed")}
    raise ValueError(f"unknown action {action!r} (want snapshot | calibrate)")
