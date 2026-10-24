"""Synthetic top-down camera frames of the cell with real tag36h11 AprilTags (simulator + tests).

    jpeg, truth = render(cell_cfg, {"servo_1": [0.25, 0.12, 30.0]})

Each object pose is [x, y, yaw_deg] of the object's reference point in the table frame (what
perception reports as `table`); its tag is placed at that point minus the rotated offset. The
camera is a mild keystone + jitter perspective over the table, rendered 2x supersampled, then
blurred, lit unevenly, noised and JPEG-compressed, so the detector and the homography get a
realistic workout. Objects not listed in `object_poses` get the defaults below, so the example
cell in docs/ROBOT_CELL.md works out of the box.
"""
from __future__ import annotations

import functools
import math
from typing import Iterable

import cv2
import numpy as np

from .perception import DEFAULT_CELL, apply_h, normalize_cell, object_pose

SIZE = (1280, 720)
MARGIN_M = 0.07             # table surface beyond the table-tag centres
SS = 2                      # supersampling factor
CELL_PX = 12                # source pixels per tag bit when generating markers

DEFAULT_POSES = {"servo_1": [0.20, 0.15, 0.0], "so101_slot_1": [0.42, 0.24, 25.0],
                 "tray_a": [0.12, 0.27, -15.0]}
# footprint (w, h) in metres and BGR colour of the thing each tag sits on
FOOTPRINT = {"part": ((0.060, 0.052), (120, 70, 40)), "slot": ((0.095, 0.065), (95, 95, 95)),
             "fixture": ((0.120, 0.090), (175, 175, 165))}
_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
_MARKERS: dict[int, np.ndarray] = {}


def camera_homography(cell: dict, size: tuple[int, int] = SIZE, warp: float = 0.06,
                      seed: int = 0) -> np.ndarray:
    """Table (metres) -> image pixel homography: fit the table into the frame (table +y = image
    up), pinch the far edge by `warp` (camera tilted towards the operator) and jitter the corners."""
    tt = normalize_cell(cell)["table_tags"]
    xs, ys = [v[0] for v in tt.values()], [v[1] for v in tt.values()]
    x0, x1, y0, y1 = min(xs) - MARGIN_M, max(xs) + MARGIN_M, min(ys) - MARGIN_M, max(ys) + MARGIN_M
    w, h = size
    pad = 0.06 * h
    s = min((w - 2 * pad) / (x1 - x0), (h - 2 * pad) / (y1 - y0))
    cu, cv = w / 2.0, h / 2.0
    hw, hh = s * (x1 - x0) / 2.0, s * (y1 - y0) / 2.0
    rng = np.random.default_rng(seed)
    jit = rng.uniform(-1, 1, (4, 2)) * warp * 0.25 * hh
    pinch = warp * hw
    dst = np.array([[cu - hw, cv + hh], [cu + hw, cv + hh],                   # near edge (y0)
                    [cu + hw - pinch, cv - hh], [cu - hw + pinch, cv - hh]]) + jit   # far edge (y1)
    src = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(np.float64)


@functools.lru_cache(maxsize=4)
def _light(w: int, h: int) -> np.ndarray:
    """Soft off-centre lamp: brightness 0.85..1.1 across the frame (HxWx1, float32)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-(((xx - 0.35 * w) / (0.8 * w)) ** 2 + ((yy - 0.3 * h) / (0.9 * h)) ** 2))
    return (0.85 + 0.25 * g)[..., None].astype(np.float32)


def _marker_patch(tag_id: int) -> np.ndarray:
    """Marker (8x8 bits incl. the black border) with a one-bit white quiet zone: 10 bits wide."""
    if tag_id not in _MARKERS:
        m = cv2.aruco.generateImageMarker(_DICT, int(tag_id), 8 * CELL_PX, borderBits=1)
        _MARKERS[tag_id] = cv2.copyMakeBorder(m, CELL_PX, CELL_PX, CELL_PX, CELL_PX,
                                              cv2.BORDER_CONSTANT, value=255)
    return _MARKERS[tag_id]


def _pose_mat(x: float, y: float, yaw_deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1.0]])


def _blit(canvas: np.ndarray, patch: np.ndarray, M: np.ndarray) -> None:
    """Warp `patch` (gray or BGR) into `canvas` through M (patch px -> canvas px), alpha-masked,
    touching only the destination bounding box."""
    ph, pw = patch.shape[:2]
    quad = apply_h(M, [[-0.5, -0.5], [pw - 0.5, -0.5], [pw - 0.5, ph - 0.5], [-0.5, ph - 0.5]])
    x0, y0 = np.floor(quad.min(axis=0)).astype(int) - 1
    x1, y1 = np.ceil(quad.max(axis=0)).astype(int) + 1
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, canvas.shape[1]), min(y1, canvas.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    Mr = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]]) @ M
    dsize = (int(x1 - x0), int(y1 - y0))
    src = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR) if patch.ndim == 2 else patch
    warped = cv2.warpPerspective(src, Mr, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    alpha = cv2.warpPerspective(np.ones((ph, pw), np.float32), Mr, dsize, flags=cv2.INTER_LINEAR,
                                borderValue=0)[..., None]
    roi = canvas[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha) + warped * alpha).astype(np.uint8)


def _poly(canvas: np.ndarray, G: np.ndarray, pts_table: Iterable, color) -> None:
    p = apply_h(G, pts_table)
    cv2.fillPoly(canvas, [np.round(p * 16).astype(np.int32)], color, cv2.LINE_AA, shift=4)


def _rect(x: float, y: float, yaw: float, w: float, h: float) -> np.ndarray:
    local = np.array([[-w / 2, -h / 2, 1], [w / 2, -h / 2, 1], [w / 2, h / 2, 1], [-w / 2, h / 2, 1]])
    return (local @ _pose_mat(x, y, yaw).T)[:, :2]


def render(cell_cfg: dict | None = None, object_poses: dict | None = None, *,
           size: tuple[int, int] = SIZE, warp: float = 0.06, seed: int = 0, noise: float = 3.0,
           blur: float = 0.8, occlude: Iterable[int] = (), hide: Iterable[int] = (),
           quality: int = 88) -> tuple[bytes, dict]:
    """-> (jpeg, ground_truth). `hide`: tag ids not drawn at all. `occlude`: tag ids drawn but
    covered by a "hand". ground_truth = {table_to_px, tags: {id: {table_xy, yaw_deg, px}},
    objects: {name: {tag, kind, table, yaw_deg}}, visible_tags}."""
    cell = normalize_cell(cell_cfg if cell_cfg is not None else DEFAULT_CELL)
    size_m = cell["tag_size_m"] or 0.04
    G = camera_homography(cell, size, warp, seed)                        # table -> final px
    A = np.array([[SS, 0, (SS - 1) / 2.0], [0, SS, (SS - 1) / 2.0], [0, 0, 1.0]])
    Gs = A @ G                                                           # table -> supersampled px
    w, h = size
    rng = np.random.default_rng(seed + 1)

    # floor, then the table top with a soft wood-ish tone
    canvas = np.empty((h * SS, w * SS, 3), np.uint8)
    canvas[:] = (58, 62, 66)
    xs = [v[0] for v in cell["table_tags"].values()]
    ys = [v[1] for v in cell["table_tags"].values()]
    tx0, tx1, ty0, ty1 = min(xs) - MARGIN_M, max(xs) + MARGIN_M, min(ys) - MARGIN_M, max(ys) + MARGIN_M
    _poly(canvas, Gs, [[tx0, ty0], [tx1, ty0], [tx1, ty1], [tx0, ty1]], (150, 178, 196))

    # tags: table tags as placed in config, object tags from the requested object poses
    tags: dict[int, tuple[float, float, float]] = {k: (v[0], v[1], v[2]) for k, v in cell["table_tags"].items()}
    poses = {**DEFAULT_POSES, **{k: list(v) for k, v in (object_poses or {}).items()}}
    truth_objs = {}
    for i, (name, o) in enumerate(cell["objects"].items()):
        x, y, yaw = (poses.get(name) or [0.1 + 0.12 * (i % 4), 0.1 + 0.1 * (i // 4), 0.0])[:3]
        ox, oy, oz = o["offset"]
        # tag = reference point minus the offset rotated by yaw
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        tag_xy = (x - (c * ox - s * oy), y - (s * ox + c * oy))
        tags[o["tag"]] = (tag_xy[0], tag_xy[1], float(yaw))
        xyz, _ = object_pose((tag_xy[0], tag_xy[1], yaw), o["offset"])
        truth_objs[name] = {"tag": o["tag"], "kind": o["kind"], "table": xyz, "yaw_deg": float(yaw)}
        (fw, fh), col = FOOTPRINT.get(o["kind"], FOOTPRINT["part"])
        _poly(canvas, Gs, _rect(tag_xy[0], tag_xy[1], yaw, fw, fh), col)

    hide, occlude = set(hide), set(occlude)
    patch_m = size_m * 10 / 8                                          # marker + quiet zone
    truth_tags = {}
    for tid, (x, y, yaw) in tags.items():
        px = apply_h(G, [[x, y]])[0]
        truth_tags[tid] = {"table_xy": [x, y], "yaw_deg": yaw, "px": [float(px[0]), float(px[1])]}
        if tid in hide:
            continue
        patch = _marker_patch(tid)
        n = patch.shape[0]
        k = patch_m / n          # patch pixel centre p -> local metres (p + 0.5 - n/2) * k, y flipped
        P = np.array([[k, 0, (0.5 - n / 2) * k], [0, -k, -(0.5 - n / 2) * k], [0, 0, 1.0]])
        _blit(canvas, patch, Gs @ _pose_mat(x, y, yaw) @ P)
    for tid in occlude & set(tags):                                    # a hand over the tag
        x, y, _ = tags[tid]
        r = size_m * 0.75
        ang = np.linspace(0, 2 * math.pi, 40)
        _poly(canvas, Gs, np.c_[x + r * np.cos(ang), y + 0.8 * r * np.sin(ang)], (120, 150, 205))

    img = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    # uneven lighting, lens blur, sensor noise
    img = img.astype(np.float32) * _light(w, h)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if noise > 0:
        img += rng.standard_normal(img.shape, dtype=np.float32) * np.float32(noise)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG encode failed")
    visible = sorted(t for t in tags if t not in hide and t not in occlude)
    return buf.tobytes(), {"table_to_px": G.tolist(), "tags": truth_tags, "objects": truth_objs,
                           "visible_tags": visible}
