"""Cell camera: capability "vision.cell" (AprilTag perception -> topic "cell.state", untagged
items -> topic "cell.blobs").

Only active on a node whose config lists "vision.cell" in `capabilities`. The cell layout comes
from `capability_config["cell"]` (docs/ROBOT_CELL.md; without it the doc's example cell is used,
with a warning). Settings live in `capability_config["vision.cell"]`:

    source: node | mock      node: this node's camera (node.vision.capture_frame, e.g. a Brio on
                             camera_url "usb:0?w=1920&h=1080&fps=30"); mock: a synthetic tag scene
    rate_hz: 2               frames processed per second
    stale_s: 3.0             hold the last calibration this long while table tags are hidden
    camera: ""               name in cell.state (default: this node's id)
    mock_poses: {servo_1: [x, y, yaw_deg], ...}   mock scene object poses (table frame)
    blobs: true              untagged-item detection (docs/LABMATE.md "Perception additions")
    background_s: 1.0        how long the "background" action averages the empty table
    blob_min_area_m2: 1.5e-4, blob_mask_margin_m: 0.01, blob_px_per_m: 1000   (swarmmind/cell/blobs.py)
    mock_items: [{item, table_xy, yaw_deg, state, name?}, ...]   mock untagged items
                             (default: mock_scene.DEFAULT_ITEMS; [] = an empty table)

The loop sets `node.cell_state` (latest cell.state dict), `node.cell_annotated_jpeg`,
`node.cell_blobs` (latest cell.blobs dict) and `node.cell_marked_jpeg` (the frame with numbered
blob marks), and publishes "cell.blobs" then "cell.state" each cycle. In mock mode, publishing
{name: [x, y, yaw_deg]} to topic "cell.mock.poses" moves tagged objects AND untagged items (by
item name) in the synthetic scene (the simulated arm can "carry" things); an item can also take
{"table_xy": [x, y], "yaw_deg": d, "state": "open" | null}, and null removes it. Mock mode
captures the background (the empty-table render) at startup and again whenever a tagged object
moves, and adds `sim_truth` {"<blob id>": "<item>[:<state>]"} to cell.blobs. Real cameras never
get sim_truth.

Task actions (capability "vision.cell"):
    {"action": "snapshot"}     -> the current cell.state
    {"action": "calibrate"}    -> a fresh fit on a new frame: {calibrated, reproj_err_px,
                                  table_tags_seen, table_tags_missing, tags_seen, corners_used, error}
    {"action": "background"}   -> capture the empty table now (median over ~background_s; mock:
                                  the empty-table render): {background, frames, ts}
    {"action": "blobs"}        -> the latest cell.blobs
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
    from ..cell import blobs as B
    from ..cell import mock_scene
    from ..cell import perception as P
    _IMPORT_ERROR = ""
except ImportError as e:  # pragma: no cover - depends on the machine
    mock_scene = P = B = None  # type: ignore[assignment]
    _IMPORT_ERROR = str(e)

CAP = "vision.cell"
TOPIC = "cell.state"
BLOBS_TOPIC = "cell.blobs"
MOCK_TOPIC = "cell.mock.poses"
MOCK_BG_FRAMES = 5


class CellVision:
    def __init__(self, node: "Node", conf: dict, cell: dict):
        self.node = node
        self.source = str(conf.get("source", "node")).lower()
        if self.source not in ("node", "mock"):
            raise ValueError(f"{CAP}: source must be node | mock, not {self.source!r}")
        self.period = 1.0 / max(float(conf.get("rate_hz", 2.0)), 0.1)
        self.camera = str(conf.get("camera") or node.cfg.node_id)
        self.cell = P.normalize_cell(cell)
        self._cell_raw, self._conf = cell, conf
        self.calib = P.Calibration(cell, stale_s=float(conf.get("stale_s", P.STALE_S)))
        self.mock_poses = {k: list(v) for k, v in (conf.get("mock_poses") or {}).items()}
        self.mock_items = mock_scene.normalize_items(conf.get("mock_items"))
        self.mock_bg_dirty = self.source == "mock"     # mock: capture the background first
        self.background_s = max(float(conf.get("background_s", 1.0)), 0.0)
        self.blobs: B.BlobPipeline | None = None
        if conf.get("blobs", True):
            self.blobs = B.BlobPipeline(
                cell, px_per_m=float(conf.get("blob_px_per_m", B.PX_PER_M)),
                min_area_m2=float(conf.get("blob_min_area_m2", B.MIN_AREA_M2)),
                mask_margin_m=float(conf.get("blob_mask_margin_m", B.MASK_MARGIN_M)))
        self.frames = 0
        self.failing = ""          # current failure kind, logged once until it recovers
        self.blob_failing = False
        self.task: asyncio.Task | None = None
        self.lock = asyncio.Lock()  # one frame through the pipeline at a time (loop vs. actions)

    def reset_camera(self) -> None:
        """A different camera: its table fit and empty-table background no longer apply."""
        conf = self._conf
        self.calib = P.Calibration(self._cell_raw, stale_s=float(conf.get("stale_s", P.STALE_S)))
        if conf.get("blobs", True):
            self.blobs = B.BlobPipeline(
                self._cell_raw, px_per_m=float(conf.get("blob_px_per_m", B.PX_PER_M)),
                min_area_m2=float(conf.get("blob_min_area_m2", B.MIN_AREA_M2)),
                mask_margin_m=float(conf.get("blob_mask_margin_m", B.MASK_MARGIN_M)))

    async def grab(self) -> tuple[bytes | None, float, dict | None]:
        ts = time.time()
        if self.source == "mock":
            self.frames += 1
            jpeg, truth = await asyncio.to_thread(mock_scene.render, self.cell, self.mock_poses,
                                                  seed=self.frames % 1000, items=self.mock_items)
            return jpeg, ts, truth
        return await self.node.vision.capture_frame(), ts, None

    def _step(self, jpeg: bytes, ts: float, truth: dict | None
              ) -> tuple[dict, list[dict], bytes, dict | None, bytes | None]:
        img = P.to_bgr(jpeg)                       # decode once for tags, blobs and both overlays
        tags = P.detect_tags(img)
        self.calib.update(tags, ts)
        state = P.build_state(tags, self.calib, ts, self.camera)
        need = sorted(int(k) for k in self.cell["table_tags"])
        seen = {int(t["id"]) for t in tags}
        state["table_tags_need"], state["table_tags_seen"] = need, [i for i in need if i in seen]
        annotated = P.annotate(img, state, self.cell)
        payload = marked = None
        if self.blobs is not None:
            try:
                H = self.calib.H if state["calibrated"] else None
                full = self.blobs.process(img, H, tags, ts)
                payload = B.publishable(full)
                if self.source == "mock" and truth is not None:
                    payload["sim_truth"] = B.match_truth(full["_blobs"], truth.get("items") or [])
                marked = B.annotate_marks(img, full["_blobs"], H, full["background"])
                if self.blob_failing:
                    self.blob_failing = False
                    self.node.event(f"{CAP}: blob detection ok again")
            except Exception as e:  # noqa: BLE001 - tags (cell.state) must keep flowing
                if not self.blob_failing:
                    log.exception("%s: blob detection failed", CAP)
                    self.node.event(f"{CAP}: blob detection failed: {e!r}", level="warning")
                self.blob_failing = True
                payload = marked = None
        return state, tags, annotated, payload, marked

    async def once(self) -> tuple[dict, list[dict]] | None:
        async with self.lock:
            if self.mock_bg_dirty and self.blobs is not None:
                await self._capture_background_locked()
            jpeg, ts, truth = await self.grab()
            if not jpeg:
                self._fail("no_frame", f"{CAP}: no camera frame (source {self.source}, "
                                       f"camera_url {self.node.cfg.camera_url!r})")
                return None
            state, tags, annotated, payload, marked = await asyncio.to_thread(self._step, jpeg, ts, truth)
        self.node.cell_state, self.node.cell_annotated_jpeg = state, annotated
        if payload is not None:
            self.node.cell_blobs, self.node.cell_marked_jpeg = payload, marked
            await self.node.publish(BLOBS_TOPIC, payload)
        await self.node.publish(TOPIC, state)
        if not state["calibrated"]:
            self._fail("uncalibrated", f"{CAP}: not calibrated ({self.calib.last_error or 'no fit yet'})")
        elif self.failing:
            self.node.event(f"{CAP}: ok again (calibrated, reprojection error {state['reproj_err_px']} px)")
            self.failing = ""
        return state, tags

    async def capture_background(self) -> dict:
        if self.blobs is None:
            raise RuntimeError(f"{CAP}: blobs are disabled (capability_config.{CAP}.blobs: false)")
        async with self.lock:
            return await self._capture_background_locked()

    async def _capture_background_locked(self) -> dict:
        """Median of the empty table. Mock: renders without items (tagged objects where they are
        now); real: every frame over background_s that gives a table fit."""
        self.mock_bg_dirty = False
        t0 = time.time()
        imgs: list[Any] = []
        if self.source == "mock":
            for i in range(MOCK_BG_FRAMES):
                jpeg, _ = await asyncio.to_thread(mock_scene.render, self.cell, self.mock_poses,
                                                  seed=(self.frames + 500 + i) % 1000, items=[])
                imgs.append(jpeg)
        else:
            gap = self.background_s / B.BG_FRAMES
            deadline = time.monotonic() + self.background_s
            give_up = deadline + 2.0                         # a camera that gives nothing
            while True:
                jpeg = await self.node.vision.capture_frame()
                if jpeg:
                    imgs.append(jpeg)
                now = time.monotonic()
                if (now >= deadline and imgs) or now >= give_up or len(imgs) >= 2 * B.BG_FRAMES:
                    break
                await asyncio.sleep(gap)
        if not imgs:
            raise RuntimeError(f"{CAP}: no camera frame for the background")

        def fit_all() -> tuple[int, str]:
            frames, Hs, tags_list = [], [], []
            for jpeg in imgs:
                try:
                    img = P.to_bgr(jpeg)
                except ValueError:
                    continue
                tags = P.detect_tags(img)
                fit = self.calib.fit(tags)
                if fit is None:
                    continue
                frames.append(img)
                Hs.append(fit[0])
                tags_list.append(tags)
            if not frames:
                return 0, self.calib.last_error or "no table fit"
            self.blobs.capture_background(frames, Hs, tags_list, t0)
            return len(frames), ""

        n_ok, err = await asyncio.to_thread(fit_all)
        if not n_ok:
            raise RuntimeError(f"{CAP}: background needs a calibrated view of the table ({err})")
        self.node.event(f"{CAP}: background captured ({n_ok} frames, source {self.source})")
        return {"background": True, "frames": n_ok, "ts": t0}

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

    def move_mock(self, data: dict) -> None:
        """cell.mock.poses: {name: [x, y, yaw]} for tagged objects and untagged items."""
        for name, pose in data.items():
            if name in self.cell["objects"]:
                if isinstance(pose, (list, tuple)) and len(pose) >= 2:
                    self.mock_poses[name] = [float(v) for v in pose[:3]] + [0.0] * (3 - len(pose[:3]))
                    self.mock_bg_dirty = True     # the empty table changed: re-render it
                continue
            idx = next((i for i, it in enumerate(self.mock_items) if it["name"] == name), None)
            if idx is None:
                continue
            if pose is None:
                del self.mock_items[idx]
                continue
            it = dict(self.mock_items[idx])
            if isinstance(pose, (list, tuple)) and len(pose) >= 2:
                it["table_xy"] = [float(pose[0]), float(pose[1])]
                if len(pose) >= 3:
                    it["yaw_deg"] = float(pose[2])
            elif isinstance(pose, dict):
                if pose.get("table_xy") is not None:
                    it["table_xy"] = [float(v) for v in pose["table_xy"][:2]]
                if pose.get("yaw_deg") is not None:
                    it["yaw_deg"] = float(pose["yaw_deg"])
                if "state" in pose:
                    it["state"] = pose["state"] or None
            else:
                continue
            self.mock_items[idx] = it


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
    node.cell_blobs, node.cell_marked_jpeg = None, None
    _CV = CellVision(node, conf, cell)
    _CV.task = asyncio.create_task(_CV.run())
    if not _SUBSCRIBED:
        subscribe(MOCK_TOPIC, _on_mock_poses)
        _SUBSCRIBED = True
    node.event(f"{CAP} ready on {node.cfg.node_id}: source {_CV.source}, {1 / _CV.period:g} Hz, "
               f"{len(_CV.cell['table_tags'])} table tags, {len(_CV.cell['objects'])} objects, "
               f"blobs {'on' if _CV.blobs is not None else 'off'}")


async def shutdown() -> None:
    global _CV
    if _CV is not None and _CV.task is not None:
        _CV.task.cancel()
        try:
            await _CV.task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    _CV = None


def reset_camera() -> None:
    if _CV is not None:
        _CV.reset_camera()


async def _on_mock_poses(sample: dict) -> None:
    data = sample.get("data")
    if _CV is not None and _CV.source == "mock" and isinstance(data, dict):
        _CV.move_mock(data)


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
    if action == "background":
        await progress({"step": "capturing background"})
        return await _CV.capture_background()
    if action == "blobs":
        if _CV.blobs is None:
            raise RuntimeError(f"{CAP}: blobs are disabled (capability_config.{CAP}.blobs: false)")
        if getattr(_CV.node, "cell_blobs", None) is None:
            await _CV.once()
        if _CV.node.cell_blobs is None:
            raise RuntimeError(f"{CAP}: no frame yet")
        return _CV.node.cell_blobs
    raise ValueError(f"unknown action {action!r} (want snapshot | calibrate | background | blobs)")
