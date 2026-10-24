"""AprilTag perception for the robot cell: pixels -> table plane (metres) -> arm base frame.

Pure functions plus one small stateful `Calibration` (the homography cache). No node, no asyncio,
so everything here is unit-testable on a synthetic frame (see swarmmind/cell/mock_scene.py).

Frames (docs/ROBOT_CELL.md):
  * pixel:  OpenCV image coordinates, u right, v down, pixel centres on integers.
  * table:  metres on the table surface, defined by `table_tags` {id: [x, y]} (optionally
            [x, y, yaw_deg]; default yaw 0 = the tag's top edge runs along table +x).
            z is up, out of the table.
  * arm:    `arms.<id>.base` = [x, y, yaw_deg, z_table] is the arm base POSE in the table frame
            (z_table = height of the arm's base origin above the table surface). Arm coordinates
            are that pose's inverse applied to a table point.

Tag yaw = direction of the tag's top edge (corner 0 -> corner 1) in the table frame, so a tag
printed upright and laid square to the table reads 0 deg; counter-clockwise (seen from above) is +.

Limits: the homography is for the table plane. A tag on top of a part at height h is seen slightly
displaced (parallax, about h * r / camera_height for a point r from the camera's foot); with the
camera ~1 m above the table and a 3 cm tall servo that's a few mm at the edge of the view.
"""
from __future__ import annotations

import math
import time
from typing import Any, Iterable

import cv2
import numpy as np

# The example cell from docs/ROBOT_CELL.md; used by the mock scene and as the simulator default.
DEFAULT_CELL: dict = {
    "tag_size_m": 0.04,
    "table_tags": {0: [0.0, 0.0], 1: [0.60, 0.0], 2: [0.60, 0.40], 3: [0.0, 0.40]},
    "objects": {
        "servo_1": {"tag": 10, "kind": "part", "offset": [0.0, 0.0, 0.015], "grasp_yaw": 0},
        "so101_slot_1": {"tag": 20, "kind": "slot", "offset": [0.07, 0.0, 0.03], "insert_depth": 0.02},
        "tray_a": {"tag": 11, "kind": "fixture", "offset": [0.0, 0.0, 0.0]},
    },
    "arms": {"arm.big": {"base": [0.30, -0.25, 90.0, 0.0], "approach_h": 0.08, "speed": 0.05}},
}

MIN_TABLE_TAGS = 4
STALE_S = 3.0               # a cached homography older than this no longer counts as calibrated
MAX_REPROJ_PX = 6.0         # a fresh fit worse than this is rejected (a tag was misdetected)
RANSAC_PX = 4.0           # inlier threshold for the table-tag fit

_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


# ---------------------------------------------------------------- config

def normalize_cell(cell: dict | None) -> dict:
    """Cell config with int tag ids and float lists (YAML gives int keys, JSON gives strings)."""
    cell = dict(cell or {})
    tt = {}
    for k, v in (cell.get("table_tags") or {}).items():
        v = [float(a) for a in v]
        tt[int(k)] = v + [0.0] * (3 - len(v)) if len(v) < 3 else v[:3]
    objs = {}
    for name, o in (cell.get("objects") or {}).items():
        o = dict(o or {})
        off = [float(a) for a in (o.get("offset") or [0.0, 0.0, 0.0])]
        o["offset"] = (off + [0.0, 0.0, 0.0])[:3]
        o["tag"] = int(o["tag"])
        o.setdefault("kind", "part")
        objs[str(name)] = o
    size = cell.get("tag_size_m")
    return {**cell, "table_tags": tt, "objects": objs,
            "tag_size_m": float(size) if size else None, "arms": dict(cell.get("arms") or {})}


# ---------------------------------------------------------------- detection

def make_detector() -> cv2.aruco.ArucoDetector:
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX   # APRILTAG refinement is slower
    p.cornerRefinementWinSize = 4                               # and no better on sharp prints
    p.cornerRefinementMaxIterations = 50
    p.cornerRefinementMinAccuracy = 0.01
    p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 5, 35, 6
    p.minMarkerPerimeterRate = 0.02                             # small tags on a 1080p frame
    return cv2.aruco.ArucoDetector(_DICT, p)


_DETECTOR: cv2.aruco.ArucoDetector | None = None


def to_bgr(image: Any) -> np.ndarray:
    """JPEG/PNG bytes or an ndarray (gray or BGR) -> BGR ndarray."""
    if isinstance(image, (bytes, bytearray, memoryview)):
        img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode image")
        return img
    img = np.asarray(image)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def _center(c: np.ndarray) -> np.ndarray:
    """Projective centre of a quad: intersection of its diagonals (exact under perspective)."""
    p0, p1, p2, p3 = c
    d1, d2 = p2 - p0, p3 - p1
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return c.mean(axis=0)
    t = ((p1[0] - p0[0]) * d2[1] - (p1[1] - p0[1]) * d2[0]) / den
    return p0 + t * d1


def detect_tags(image: Any, detector: cv2.aruco.ArucoDetector | None = None) -> list[dict]:
    """tag36h11 detections: [{id, px: [u, v], corners: [[u, v] x4]}], corners in AprilTag order
    (top-left, top-right, bottom-right, bottom-left of the printed tag). Duplicate ids keep the
    largest instance."""
    global _DETECTOR
    if detector is None:
        _DETECTOR = _DETECTOR or make_detector()
        detector = _DETECTOR
    img = image if isinstance(image, np.ndarray) and image.ndim == 2 else to_bgr(image)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    best: dict[int, tuple[float, dict]] = {}
    if ids is None:
        return []
    for c, i in zip(corners, ids.reshape(-1)):
        c = c.reshape(4, 2).astype(np.float64)
        area = abs(cv2.contourArea(c.astype(np.float32)))
        tag = {"id": int(i), "px": [round(float(v), 2) for v in _center(c)],
               "corners": [[round(float(u), 2), round(float(v), 2)] for u, v in c]}
        if int(i) not in best or area > best[int(i)][0]:
            best[int(i)] = (area, tag)
    return [best[k][1] for k in sorted(best)]


# ---------------------------------------------------------------- geometry

def apply_h(H: np.ndarray, pts: Iterable) -> np.ndarray:
    """Apply a 3x3 homography to Nx2 points."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = np.hstack([p, np.ones((len(p), 1))]) @ np.asarray(H, dtype=np.float64).T
    return q[:, :2] / q[:, 2:3]


def tag_corners_table(x: float, y: float, yaw_deg: float, size: float) -> np.ndarray:
    """Table-frame corners of a tag (AprilTag order) centred at (x, y) with the given yaw."""
    h = size / 2.0
    local = np.array([[-h, h], [h, h], [h, -h], [-h, -h]])
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return local @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def tag_pose_table(H: np.ndarray, corners_px: Iterable) -> tuple[float, float, float]:
    """(x, y, yaw_deg) of a tag in the table frame from its pixel corners."""
    t = apply_h(H, corners_px)
    x, y = t.mean(axis=0)                  # on the plane the centre is the mean of the corners
    top, bottom = t[1] - t[0], t[2] - t[3]
    left, right = t[0] - t[3], t[1] - t[2]
    # average the two "x" edges and the two "y" edges (y edge rotated -90 deg onto x)
    vx = top + bottom + np.array([left[1] + right[1], -(left[0] + right[0])])
    return float(x), float(y), _wrap(math.degrees(math.atan2(vx[1], vx[0])))


def object_pose(tag_xy_yaw: tuple[float, float, float], offset: Iterable) -> tuple[list[float], float]:
    """Object [x, y, z] in the table frame: tag (x, y) + offset rotated by the tag yaw, z = offset z."""
    x, y, yaw = tag_xy_yaw
    ox, oy, oz = (list(offset) + [0.0, 0.0, 0.0])[:3]
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return [x + c * ox - s * oy, y + s * ox + c * oy, float(oz)], yaw


def _base(arm_cfg: Any) -> tuple[float, float, float, float]:
    base = arm_cfg.get("base") if isinstance(arm_cfg, dict) else arm_cfg
    b = [float(v) for v in (base or [0.0, 0.0, 0.0, 0.0])]
    b += [0.0] * (4 - len(b))
    return b[0], b[1], b[2], b[3]


def table_to_arm(xyz_table: Iterable, arm_cfg: Any) -> tuple[float, float, float]:
    """Table-frame point -> arm base frame. arm_cfg is an `arms.<id>` entry (or its `base` list)."""
    bx, by, yaw, bz = _base(arm_cfg)
    x, y, z = (list(xyz_table) + [0.0])[:3]
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    dx, dy = x - bx, y - by
    return c * dx + s * dy, -s * dx + c * dy, z - bz


def arm_to_table(xyz_arm: Iterable, arm_cfg: Any) -> tuple[float, float, float]:
    """Inverse of table_to_arm."""
    bx, by, yaw, bz = _base(arm_cfg)
    x, y, z = (list(xyz_arm) + [0.0])[:3]
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return bx + c * x - s * y, by + s * x + c * y, z + bz


def yaw_table_to_arm(yaw_deg: float, arm_cfg: Any) -> float:
    return _wrap(float(yaw_deg) - _base(arm_cfg)[2])


def yaw_arm_to_table(yaw_deg: float, arm_cfg: Any) -> float:
    return _wrap(float(yaw_deg) + _base(arm_cfg)[2])


# ---------------------------------------------------------------- calibration

class Calibration:
    """Pixel -> table homography from the table tags, with a short-lived cache.

    update(tags) refits whenever >= 4 table tags are visible. If fewer are visible (a hand over a
    tag), the last good fit is held for `stale_s` seconds, then calibration is dropped."""

    def __init__(self, cell: dict | None = None, stale_s: float = STALE_S,
                 max_reproj_px: float = MAX_REPROJ_PX):
        self.cell = normalize_cell(cell if cell is not None else DEFAULT_CELL)
        self.stale_s, self.max_reproj_px = float(stale_s), float(max_reproj_px)
        self.H: np.ndarray | None = None          # pixel -> table
        self.reproj_err_px: float | None = None
        self.fit_ts = 0.0                         # when H was last refit
        self.tags_used: list[int] = []
        self.corners_used = False
        self.last_error = ""

    def fit(self, tags: list[dict]) -> tuple[np.ndarray, float, list[int], bool] | None:
        """One fit from these detections (no caching). None if < 4 table tags or the fit is bad."""
        tt, size = self.cell["table_tags"], self.cell["tag_size_m"]
        seen = [t for t in tags if t["id"] in tt]
        if len(seen) < MIN_TABLE_TAGS:
            self.last_error = f"{len(seen)} of {MIN_TABLE_TAGS} table tags visible"
            return None
        centers_px = np.array([t["px"] for t in seen], dtype=np.float64)
        centers_tb = np.array([tt[t["id"]][:2] for t in seen], dtype=np.float64)
        if size:   # corners give 5x the points: a real error estimate and a better fit
            px = np.vstack([centers_px] + [np.array(t["corners"], dtype=np.float64) for t in seen])
            tb = np.vstack([centers_tb] + [tag_corners_table(*tt[t["id"]], size) for t in seen])
            best = self._solve(px, tb)
            if best is not None:
                return best[0], best[1], sorted(t["id"] for t in seen), True
        # no size, or the corners disagree (e.g. a table tag mounted rotated): centres only
        best = self._solve(centers_px, centers_tb)
        return None if best is None else (best[0], best[1], sorted(t["id"] for t in seen), False)

    def _solve(self, px: np.ndarray, tb: np.ndarray) -> tuple[np.ndarray, float] | None:
        # fit table -> pixel so RANSAC and the least-squares refinement work in pixels, then invert
        G, _ = cv2.findHomography(tb, px, cv2.RANSAC if len(px) > 4 else 0, RANSAC_PX)
        if G is None or abs(np.linalg.det(G)) < 1e-12:
            self.last_error = "homography fit failed"
            return None
        err = float(np.mean(np.linalg.norm(apply_h(G, tb) - px, axis=1)))
        if err > self.max_reproj_px:
            self.last_error = f"reprojection error {err:.1f} px"
            return None
        return np.linalg.inv(G), err

    def update(self, tags: list[dict], now: float | None = None) -> bool:
        now = time.time() if now is None else now
        res = self.fit(tags)
        if res is not None:
            self.H, self.reproj_err_px, self.tags_used, self.corners_used = res
            self.fit_ts, self.last_error = now, ""
        return self.valid(now)

    def valid(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return self.H is not None and now - self.fit_ts <= self.stale_s

    def age_s(self, now: float | None = None) -> float | None:
        return None if self.H is None else (time.time() if now is None else now) - self.fit_ts


# ---------------------------------------------------------------- state

def build_state(tags: list[dict], calib: Calibration, frame_ts: float, camera: str,
                now: float | None = None) -> dict:
    """The `cell.state` dict from docs/ROBOT_CELL.md. Coordinates (table_xy, table, yaw_deg) are
    null when not calibrated; an object whose tag isn't seen is visible: false."""
    cell = calib.cell
    ok = calib.valid(frame_ts if now is None else now)
    poses: dict[int, tuple[float, float, float]] = {}
    out_tags = []
    for t in tags:
        entry = {"id": t["id"], "px": t["px"], "corners": t["corners"], "table_xy": None, "yaw_deg": None}
        if ok:
            poses[t["id"]] = tag_pose_table(calib.H, t["corners"])
            x, y, yaw = poses[t["id"]]
            entry["table_xy"], entry["yaw_deg"] = [round(x, 4), round(y, 4)], round(yaw, 2)
        out_tags.append(entry)
    seen = {t["id"] for t in tags}
    objects = []
    for name, o in cell["objects"].items():
        ob = {"name": name, "kind": o["kind"], "tag": o["tag"], "visible": o["tag"] in seen,
              "table": None, "yaw_deg": None}
        if ok and o["tag"] in poses:
            xyz, yaw = object_pose(poses[o["tag"]], o["offset"])
            ob["table"], ob["yaw_deg"] = [round(v, 4) for v in xyz], round(yaw, 2)
        objects.append(ob)
    return {"frame_ts": frame_ts, "camera": camera, "calibrated": ok, "tags": out_tags,
            "objects": objects,
            "reproj_err_px": round(calib.reproj_err_px, 3) if ok and calib.reproj_err_px is not None else None}


def process(image: Any, calib: Calibration, camera: str = "", frame_ts: float | None = None) -> dict:
    """detect -> calibrate -> state, in one call (what the capability loop runs per frame)."""
    ts = time.time() if frame_ts is None else frame_ts
    tags = detect_tags(image)
    calib.update(tags, ts)
    return build_state(tags, calib, ts, camera)


# ---------------------------------------------------------------- annotation

GREEN, ORANGE, GRAY, RED, BLUE, WHITE, BLACK = ((60, 200, 60), (0, 150, 255), (170, 170, 170),
                                                (40, 40, 230), (230, 120, 40), (255, 255, 255), (0, 0, 0))


def _label(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color, thick: int) -> None:
    """Text on a dark box so it stays legible on any background (and to a vision model)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), base = cv2.getTextSize(text, font, scale, thick)
    x = int(min(max(org[0], 2), img.shape[1] - w - 4))
    y = int(min(max(org[1], h + 4), img.shape[0] - base - 2))
    cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base + 2), BLACK, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def _table_to_px(state: dict) -> np.ndarray | None:
    """Recover the table -> pixel homography from the state itself (tags with px and table_xy)."""
    pairs = [(t["px"], t["table_xy"]) for t in state.get("tags", []) if t.get("table_xy")]
    if len(pairs) < 4:
        return None
    px = np.array([p for p, _ in pairs], dtype=np.float64)
    tb = np.array([q for _, q in pairs], dtype=np.float64)
    H, _ = cv2.findHomography(tb, px, 0)
    return H


def annotate(image: Any, state: dict, cell: dict | None = None, quality: int = 85) -> bytes:
    """JPEG with tag outlines, object names, the calibration status and the table axes.
    `cell` (optional) tells table tags from object tags; it defaults to DEFAULT_CELL's ids plus
    whatever the state's objects reference."""
    img = to_bgr(image).copy()
    hgt, wid = img.shape[:2]
    k = max(wid / 1280.0, 0.5)
    scale, thick = 1.0 * k, max(2, int(round(2 * k)))
    table_ids = set(normalize_cell(cell)["table_tags"]) if cell else set(DEFAULT_CELL["table_tags"])
    names = {o["tag"]: o for o in state.get("objects", [])}
    for t in state.get("tags", []):
        c = np.array(t["corners"], dtype=np.float64)
        color = GREEN if t["id"] in table_ids else ORANGE if t["id"] in names else GRAY
        cv2.polylines(img, [np.round(c).astype(np.int32)], True, color, thick, cv2.LINE_AA)
        cv2.circle(img, tuple(int(round(v)) for v in c[0]), thick + 2, RED, -1)   # corner 0 = top-left
        u, v = (int(round(a)) for a in t["px"])
        size = float(np.linalg.norm(c[0] - c[2]))
        if t["id"] in names:
            o = names[t["id"]]
            _label(img, o["name"], (u - int(size / 2), v - int(size * 0.6) - 6), scale, WHITE, thick)
        else:
            _label(img, f"tag {t['id']}", (u - int(size / 2), v + int(size * 0.6) + int(24 * k)),
                   0.6 * k, color, max(1, thick - 1))
    H = _table_to_px(state) if state.get("calibrated") else None
    if H is not None:
        o, ax, ay = apply_h(H, [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
        po = tuple(int(round(v)) for v in o)
        for end, col, name in ((ax, RED, "X"), (ay, GREEN, "Y")):
            pe = tuple(int(round(v)) for v in end)
            cv2.arrowedLine(img, po, pe, col, thick + 1, cv2.LINE_AA, tipLength=0.2)
            cv2.putText(img, name, (pe[0] + 6, pe[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, scale, col,
                        thick, cv2.LINE_AA)
        for ob in state.get("objects", []):   # object reference points (tag + offset)
            if ob.get("table"):
                p = tuple(int(round(v)) for v in apply_h(H, [ob["table"][:2]])[0])
                cv2.drawMarker(img, p, BLUE, cv2.MARKER_CROSS, int(18 * k), thick, cv2.LINE_AA)
    if state.get("calibrated"):
        err = state.get("reproj_err_px")
        status, col = "CALIBRATED" + (f"  err {err:.2f} px" if err is not None else ""), GREEN
    else:
        status, col = "NOT CALIBRATED", RED
    _label(img, status, (12, int(40 * k)), 1.1 * k, col, thick + 1)
    missing = [o["name"] for o in state.get("objects", []) if not o.get("visible")]
    if missing:
        _label(img, "not visible: " + ", ".join(missing), (12, int(86 * k)), 0.8 * k, ORANGE, thick)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG encode failed")
    return buf.tobytes()
