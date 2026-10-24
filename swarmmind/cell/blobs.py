"""Untagged things on the table ("blobs") for LabMate: background subtraction on the rectified
top-down table image, minAreaRect measurement in metres, stable track ids, numbered marks.

Pure library (no node, no asyncio), like perception.py. The pipeline per frame:

    rect = rectify(frame, calib.H, extent)              # 1 px = 1 mm, table +y = image up
    fg   = |LAB(rect) - background| (+ new edges)       # brightness/white balance normalised
    fg  &= table polygon, minus every AprilTag (+ quiet zone + margin)
    close -> open -> fill holes -> contours >= min_area -> minAreaRect -> metres
    Tracker: nearest neighbour (gate 3 cm), an id survives <= 1 s of dropout

Without a captured background a table-colour model from a ring of table surface around each
table tag stands in (`background: false` in the payload: lower quality, textured tables and
shadows give false blobs, blobs touching a tagged fixture are dropped).

Frames: rectified pixel (i, j) has its centre at table x = x0 + (i + 0.5) / ppm,
y = y1 - (j + 0.5) / ppm, so the image looks like the camera's top-down view, unmirrored.
Yaw is the long axis direction in the table frame, counter-clockwise from +x seen from above
(perception.py's convention), folded into [-90, 90) because an axis has no head.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import cv2
import numpy as np

from .perception import apply_h, normalize_cell, tag_corners_table, to_bgr

PX_PER_M = 1000.0
MIN_AREA_M2 = 1.5e-4          # 1.5 cm^2: smaller components are specks
MASK_MARGIN_M = 0.01          # masked around each tag beyond its quiet zone
QUIET_ZONE = 1.25             # printed quiet zone: tag (8 bits) + 1 white bit each side = 10/8
GATE_M = 0.03                 # tracker: max centre jump per frame
MAX_GAP_S = 1.0               # tracker: an unseen track keeps its id this long
BG_FRAMES = 7                 # frames in a background median

# foreground thresholds (OpenCV 8-bit LAB units: L 0..255, a/b offset by 128), raised
# automatically when the frame is noisier than this
L_THRESH, C_THRESH = 28.0, 13.0
FALLBACK_L_THRESH, FALLBACK_C_THRESH = 28.0, 15.0
CLOSE_M, OPEN_M = 0.007, 0.003
NEAR_EDGE_M = 0.025           # fallback: edges count only this close to colour foreground


# ---------------------------------------------------------------- table geometry

def table_extent(cell: dict, margin_m: float | None = None) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of the table region in metres: the table tags' bounding box grown by
    `margin_m` (default 0.75 tag widths, i.e. just past each tag's quiet zone)."""
    cell = normalize_cell(cell)
    tt = cell["table_tags"]
    if not tt:
        raise ValueError("cell has no table_tags")
    m = table_margin(cell) if margin_m is None else float(margin_m)
    xs, ys = [v[0] for v in tt.values()], [v[1] for v in tt.values()]
    return min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m


def table_margin(cell: dict) -> float:
    return 0.75 * float(normalize_cell(cell)["tag_size_m"] or 0.04)


def _extent(table_size_m: Iterable[float]) -> tuple[float, float, float, float]:
    e = [float(v) for v in table_size_m]
    if len(e) == 2:
        return 0.0, 0.0, e[0], e[1]
    if len(e) == 4:
        return e[0], e[1], e[2], e[3]
    raise ValueError("table_size_m must be (w, h) or (x0, y0, x1, y1) in metres")


def table_to_rect_matrix(extent: Iterable[float], px_per_m: float = PX_PER_M) -> np.ndarray:
    """Table metres -> rectified pixel (3x3)."""
    x0, _, _, y1 = _extent(extent)
    k = float(px_per_m)
    return np.array([[k, 0.0, -x0 * k - 0.5], [0.0, -k, y1 * k - 0.5], [0.0, 0.0, 1.0]])


def rect_size(extent: Iterable[float], px_per_m: float = PX_PER_M) -> tuple[int, int]:
    x0, y0, x1, y1 = _extent(extent)
    return int(math.ceil((x1 - x0) * px_per_m)), int(math.ceil((y1 - y0) * px_per_m))


def rectify(image: Any, homography: np.ndarray, table_size_m: Iterable[float],
            px_per_m: float = PX_PER_M) -> np.ndarray:
    """Top-down table image (BGR), 1 px = 1/px_per_m metres (1 mm by default).
    `homography` is pixel -> table (Calibration.H); `table_size_m` is the region, (w, h) from the
    origin or (x0, y0, x1, y1), normally table_extent(cell)."""
    ext = _extent(table_size_m)
    M = table_to_rect_matrix(ext, px_per_m) @ np.asarray(homography, dtype=np.float64)
    return cv2.warpPerspective(to_bgr(image), M, rect_size(ext, px_per_m), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def tag_quad_table(H: np.ndarray, corners_px: Iterable, grow_m: float, quiet: float = QUIET_ZONE) -> np.ndarray:
    """A detected tag's table-frame quad grown to its quiet zone plus `grow_m` on every side."""
    t = apply_h(H, corners_px)
    c = t.mean(axis=0)
    half = 0.25 * sum(float(np.linalg.norm(t[i] - t[(i + 1) % 4])) for i in range(4)) / 2.0
    s = (half * quiet + grow_m) / max(half, 1e-6)
    return c + (t - c) * s


# ---------------------------------------------------------------- background

class Background:
    """The empty table in rectified coordinates (median of frames), its LAB and its edges."""

    def __init__(self) -> None:
        self.bgr: np.ndarray | None = None
        self.lab: np.ndarray | None = None       # float32 HxWx3
        self.edges: np.ndarray | None = None     # uint8, dilated: "edges that were already there"
        self.tag_quads: list[np.ndarray] = []    # table-frame quads of the tags seen at capture
        self.frames = 0
        self.ts = 0.0

    @property
    def ready(self) -> bool:
        return self.lab is not None

    def capture(self, rect_frames: list[np.ndarray], tag_quads: Iterable[np.ndarray] = (),
                ts: float = 0.0) -> None:
        if not rect_frames:
            raise ValueError("no frames for the background")
        shape = rect_frames[0].shape
        if any(f.shape != shape for f in rect_frames):
            raise ValueError("background frames differ in size")
        stack = np.stack(rect_frames) if len(rect_frames) > 1 else rect_frames[0][None]
        med = np.median(stack, axis=0).astype(np.uint8) if len(rect_frames) > 1 else rect_frames[0].copy()
        self.bgr = med
        sm = cv2.GaussianBlur(med, (0, 0), 1.0)
        lab8 = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB)
        self.lab = lab8.astype(np.float32)
        e = cv2.Canny(lab8[..., 0], 40, 100)
        self.edges = cv2.dilate(e, np.ones((5, 5), np.uint8))
        self.tag_quads = [np.asarray(q, dtype=np.float64) for q in tag_quads]
        self.frames, self.ts = len(rect_frames), float(ts)

    def clear(self) -> None:
        self.__init__()


# ---------------------------------------------------------------- detection

def _odd(n: float) -> int:
    n = max(1, int(round(n)))
    return n if n % 2 else n + 1


def _robust_sigma(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return 1.4826 * float(np.median(np.abs(x - med)))


class BlobDetector:
    """Background subtraction + measurement on the rectified table image (stateless per frame
    apart from the background)."""

    def __init__(self, cell: dict, px_per_m: float = PX_PER_M, min_area_m2: float = MIN_AREA_M2,
                 mask_margin_m: float = MASK_MARGIN_M, table_margin_m: float | None = None):
        self.cell = normalize_cell(cell)
        self.ppm = float(px_per_m)
        self.min_area_m2 = float(min_area_m2)
        self.mask_margin_m = float(mask_margin_m)
        self.extent = table_extent(self.cell, table_margin_m)
        self.size = rect_size(self.extent, self.ppm)
        self.R = table_to_rect_matrix(self.extent, self.ppm)
        self.Rinv = np.linalg.inv(self.R)
        self.background = Background()
        self.tag_size = float(self.cell["tag_size_m"] or 0.04)
        self.table_mask = self._table_mask()
        k = lambda m: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(m * self.ppm),) * 2)
        self.k_close, self.k_open, self.k_near = k(CLOSE_M), k(OPEN_M), k(NEAR_EDGE_M)
        self.last_mask: np.ndarray | None = None     # final foreground of the last frame (debug)

    # -- frames
    def _table_mask(self) -> np.ndarray:
        """Convex hull of the table tags grown by the table margin (rectified pixels)."""
        w, h = self.size
        pts = np.array([v[:2] for v in self.cell["table_tags"].values()], dtype=np.float64)
        mask = np.zeros((h, w), np.uint8)
        if len(pts) >= 3:
            hull = cv2.convexHull(self.to_rect(pts).astype(np.float32)).reshape(-1, 2)
            cv2.fillPoly(mask, [np.round(hull).astype(np.int32)], 255)
            m = table_margin(self.cell) * self.ppm
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(2 * m),) * 2))
        else:
            mask[:] = 255
        return mask

    def to_rect(self, pts_table: Iterable) -> np.ndarray:
        return apply_h(self.R, pts_table)

    def to_table(self, pts_rect: Iterable) -> np.ndarray:
        return apply_h(self.Rinv, pts_rect)

    def rectify(self, image: Any, H: np.ndarray) -> np.ndarray:
        M = self.R @ np.asarray(H, dtype=np.float64)
        return cv2.warpPerspective(to_bgr(image), M, self.size, flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)

    def tag_quads(self, tags: list[dict], H: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """(table-tag quads, object-tag quads) in the table frame, grown for masking. Configured
        table tags are always included, seen or not."""
        table_ids = set(self.cell["table_tags"])
        tq, oq = [], []
        for t in tags:
            q = tag_quad_table(H, t["corners"], self.mask_margin_m)
            (tq if t["id"] in table_ids else oq).append(q)
        half = self.tag_size / 2.0
        s = (half * QUIET_ZONE + self.mask_margin_m) / half
        for x, y, yaw in self.cell["table_tags"].values():
            tq.append(tag_corners_table(x, y, yaw, self.tag_size * s))
        return tq, oq

    def capture_background(self, images: list[Any], Hs: list[np.ndarray], tags_list: list[list[dict]],
                           ts: float = 0.0) -> None:
        rects, quads = [], []
        for img, H, tags in zip(images, Hs, tags_list):
            rects.append(self.rectify(img, H))
            tq, oq = self.tag_quads(tags, H)
            quads += oq
        self.background.capture(rects, quads, ts)

    def _fill(self, mask: np.ndarray, quads: Iterable[np.ndarray], value: int) -> None:
        for q in quads:   # one at a time: overlapping polygons in one fillPoly call cancel out
            cv2.fillPoly(mask, [np.round(self.to_rect(q) * 16).astype(np.int32)], value, cv2.LINE_8, shift=4)

    # -- per frame
    def foreground(self, rect: np.ndarray, valid: np.ndarray) -> np.ndarray:
        sm = cv2.GaussianBlur(rect, (0, 0), 1.0)
        lab8 = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB)
        lab = lab8.astype(np.float32)
        sel = valid[::3, ::3] > 0
        bg = self.background
        if bg.ready:
            ref = bg.lab
            Ls, Lr = lab[::3, ::3, 0][sel], ref[::3, ::3, 0][sel]
            gain = float(np.median(Lr / np.maximum(Ls, 1.0))) if Ls.size else 1.0
            gain = min(max(gain, 0.4), 2.5)            # auto-exposure swings, not a black frame
            L = lab[..., 0] * gain
            da = lab[..., 1] - ref[..., 1]
            db = lab[..., 2] - ref[..., 2]
            if Ls.size:
                da -= float(np.median(da[::3, ::3][sel]))   # white-balance drift
                db -= float(np.median(db[::3, ::3][sel]))
            dL = np.abs(L - ref[..., 0])
            dC = np.sqrt(da * da + db * db)
            tl = max(L_THRESH, 5.0 * _robust_sigma(dL[::3, ::3][sel]))
            tc = max(C_THRESH, 5.0 * _robust_sigma(dC[::3, ::3][sel]))
            fg = ((dL > tl) | (dC > tc)).astype(np.uint8) * 255
            Ln = np.clip(L, 0, 255).astype(np.uint8)
            edges = cv2.Canny(Ln, 40, 100)
            edges[bg.edges > 0] = 0
            fg |= edges
            fg &= valid
            joined = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.k_close) & valid
            return cv2.morphologyEx(joined, cv2.MORPH_OPEN, self.k_open) & valid, joined
        # no background: measure on colour alone. Edges close to colour foreground (luma and
        # chroma: grey metal on a tinted table) only decide which pieces belong together (a
        # handle and its shaft across a low-contrast ferrule); their halo would skew the size.
        fg = self._fallback_fg(lab, valid) & valid
        near = cv2.dilate(fg, self.k_near)
        e = cv2.Canny(lab8[..., 0], 15, 40)
        for ch in (1, 2):
            e |= cv2.Canny(cv2.convertScaleAbs(lab8[..., ch], alpha=3.0, beta=-3.0 * 128 + 128), 30, 80)
        closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.k_close) & valid
        joined = cv2.morphologyEx(fg | (e & near & valid), cv2.MORPH_CLOSE, self.k_close) & valid | closed
        return cv2.morphologyEx(closed, cv2.MORPH_OPEN, self.k_open) & valid, joined

    def _fallback_fg(self, lab: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """No background: table colour as an affine function of position, fitted to the median
        colour of a ring of table surface around each table tag (outside its quiet zone)."""
        h, w = valid.shape
        half_diag = self.tag_size * QUIET_ZONE / 2.0 * math.sqrt(2.0)
        r1, r2 = (half_diag + 0.004) * self.ppm, (half_diag + 0.03) * self.ppm
        yy, xx = np.mgrid[0:h:2, 0:w:2]
        vsub, labsub = valid[::2, ::2] > 0, lab[::2, ::2]
        rows, samples, sigmas = [], [], []
        for x, y, _ in self.cell["table_tags"].values():
            cu, cv = self.to_rect([[x, y]])[0]
            d2 = (xx - cu) ** 2 + (yy - cv) ** 2
            ring = vsub & (d2 >= r1 * r1) & (d2 <= r2 * r2)
            if ring.sum() < 30:
                continue
            px = labsub[ring]
            med = np.median(px, axis=0)
            dev = np.abs(px - med).max(axis=1)
            keep = px[dev < max(20.0, 3.0 * _robust_sigma(dev) + float(np.median(dev)))]
            px = keep if len(keep) >= 15 else px
            rows.append([1.0, cu / w, cv / h])
            samples.append(np.median(px, axis=0))
            sigmas.append([_robust_sigma(px[:, i]) for i in range(3)])
        if not samples:
            ref = np.median(lab[valid > 0], axis=0) if (valid > 0).any() else np.array([128.0] * 3)
            model = np.broadcast_to(ref.astype(np.float32), lab.shape)
            sig = np.array([4.0, 2.0, 2.0])
        else:
            A, B = np.array(rows), np.array(samples)
            if len(rows) >= 3:
                coef, *_ = np.linalg.lstsq(A, B, rcond=None)      # 3x3: [1, u, v] -> L, a, b
            else:
                coef = np.vstack([B.mean(axis=0), np.zeros(3), np.zeros(3)])
            uu = (np.arange(w, dtype=np.float32) / w)[None, :, None]
            vv = (np.arange(h, dtype=np.float32) / h)[:, None, None]
            model = (coef[0][None, None, :] + uu * coef[1][None, None, :] +
                     vv * coef[2][None, None, :]).astype(np.float32)
            sig = np.median(np.array(sigmas), axis=0)
        d = lab - model
        dL = np.abs(d[..., 0])
        dC = np.sqrt(d[..., 1] ** 2 + d[..., 2] ** 2)
        tl = max(FALLBACK_L_THRESH, 6.0 * float(sig[0]))
        tc = max(FALLBACK_C_THRESH, 6.0 * float(max(sig[1], sig[2])))
        return ((dL > tl) | (dC > tc)).astype(np.uint8) * 255

    def detect(self, image: Any, H: np.ndarray, tags: list[dict]) -> list[dict]:
        """Measured blobs (no ids yet): [{table_xy, yaw_deg, length_m, width_m, area_m2,
        box: 4x2 table-frame corners}]."""
        rect = self.rectify(image, H)
        tq, oq = self.tag_quads(tags, H)
        valid = self.table_mask.copy()
        self._fill(valid, tq + oq + self.background.tag_quads, 0)
        fg, joined = self.foreground(rect, valid)
        self.last_mask = fg
        drop = None
        if not self.background.ready and oq:     # fallback: a tagged fixture's body isn't an item
            drop = np.zeros_like(valid)
            self._fill(drop, oq, 255)
            drop = cv2.dilate(drop, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_px = self.min_area_m2 * self.ppm * self.ppm
        # pieces the opening cut apart but the closing had joined (a handle and its shaft across
        # a low-contrast ferrule) are one item
        _, labels = cv2.connectedComponents(joined, connectivity=8)
        groups: dict[int, list[np.ndarray]] = {}
        for c in contours:
            u, v = c[0, 0]
            groups.setdefault(int(labels[v, u]), []).append(c)
        out = []
        for parts in groups.values():
            areas = [cv2.contourArea(c) for c in parts]
            area = float(sum(areas))
            if area < min_px:
                continue
            if drop is not None:
                hit = False
                for c in parts:
                    x, y, bw, bh = cv2.boundingRect(c)
                    roi = np.zeros((bh, bw), np.uint8)
                    cv2.drawContours(roi, [c - [x, y]], -1, 255, -1)
                    hit = hit or bool((roi & drop[y:y + bh, x:x + bw]).any())
                if hit:
                    continue
            out.append(self.measure(np.vstack(parts), area))
        out.sort(key=lambda b: (-b["area_m2"], b["table_xy"]))
        return out

    def measure(self, contour: np.ndarray, area_px: float | None = None) -> dict:
        (cu, cv), (rw, rh), ang = cv2.minAreaRect(contour)
        if rw < rh:
            rw, rh, ang = rh, rw, ang + 90.0
        a = math.radians(ang)
        yaw = math.degrees(math.atan2(-math.sin(a), math.cos(a)))   # image v is down, table y up
        yaw = (yaw + 90.0) % 180.0 - 90.0
        box = self.to_table(cv2.boxPoints(((cu, cv), (rw, rh), ang)))
        x, y = self.to_table([[cu, cv]])[0]
        area = cv2.contourArea(contour) if area_px is None else area_px
        return {"table_xy": [float(x), float(y)], "yaw_deg": float(yaw),
                "length_m": float(rw / self.ppm), "width_m": float(rh / self.ppm),
                "area_m2": float(area / self.ppm ** 2), "box": box}


# ---------------------------------------------------------------- tracking

class Tracker:
    """Stable ids by greedy nearest neighbour within `gate_m`; an id whose blob isn't seen
    survives `max_gap_s`. stable_frames = consecutive frames the track has been seen."""

    def __init__(self, gate_m: float = GATE_M, max_gap_s: float = MAX_GAP_S):
        self.gate_m, self.max_gap_s = float(gate_m), float(max_gap_s)
        self.tracks: dict[int, dict] = {}
        self.next_id = 1
        self.frame = 0

    def update(self, blobs: list[dict], ts: float) -> list[dict]:
        self.frame += 1
        self.tracks = {i: t for i, t in self.tracks.items() if ts - t["ts"] <= self.max_gap_s}
        pairs = []
        for tid, t in self.tracks.items():
            for j, b in enumerate(blobs):
                d = math.dist(t["xy"], b["table_xy"])
                if d <= self.gate_m:
                    pairs.append((d, tid, j))
        pairs.sort()
        used_t, assign = set(), {}
        for _, tid, j in pairs:
            if tid in used_t or j in assign:
                continue
            used_t.add(tid)
            assign[j] = tid
        out = []
        for j, b in enumerate(blobs):
            tid = assign.get(j)
            if tid is None:
                tid, self.next_id = self.next_id, self.next_id + 1
                self.tracks[tid] = {"xy": b["table_xy"], "ts": ts, "stable": 1, "frame": self.frame}
            else:
                t = self.tracks[tid]
                t["stable"] = t["stable"] + 1 if t["frame"] == self.frame - 1 else 1
                t.update(xy=b["table_xy"], ts=ts, frame=self.frame)
            out.append({**b, "id": tid, "stable_frames": self.tracks[tid]["stable"]})
        out.sort(key=lambda b: b["id"])
        return out

    def reset(self) -> None:
        self.__init__(self.gate_m, self.max_gap_s)


# ---------------------------------------------------------------- pipeline + payload

def public_blob(b: dict) -> dict:
    """A blob as published in cell.blobs (docs/LABMATE.md)."""
    return {"id": int(b["id"]), "table_xy": [round(b["table_xy"][0], 4), round(b["table_xy"][1], 4)],
            "yaw_deg": round(b["yaw_deg"], 1), "length_m": round(b["length_m"], 4),
            "width_m": round(b["width_m"], 4), "area_m2": round(b["area_m2"], 6),
            "stable_frames": int(b["stable_frames"])}


class BlobPipeline:
    """Detector + background + tracker -> the `cell.blobs` payload."""

    def __init__(self, cell: dict, px_per_m: float = PX_PER_M, min_area_m2: float = MIN_AREA_M2,
                 mask_margin_m: float = MASK_MARGIN_M, gate_m: float = GATE_M,
                 max_gap_s: float = MAX_GAP_S, table_margin_m: float | None = None):
        self.detector = BlobDetector(cell, px_per_m, min_area_m2, mask_margin_m, table_margin_m)
        self.tracker = Tracker(gate_m, max_gap_s)
        self.latest: dict | None = None

    @property
    def background(self) -> Background:
        return self.detector.background

    def capture_background(self, images: list[Any], Hs: list[np.ndarray], tags_list: list[list[dict]],
                           ts: float = 0.0) -> None:
        self.detector.capture_background(images, Hs, tags_list, ts)
        self.tracker.reset()               # the old foreground is meaningless now

    def process(self, image: Any, H: np.ndarray | None, tags: list[dict], ts: float) -> dict:
        """-> payload; payload["_blobs"] keeps the full blobs (with table-frame boxes) for drawing
        and is stripped by `publishable`."""
        if H is None:
            payload = {"frame_ts": ts, "calibrated": False, "background": self.background.ready,
                       "blobs": [], "_blobs": []}
        else:
            tracked = self.tracker.update(self.detector.detect(image, H, tags), ts)
            payload = {"frame_ts": ts, "calibrated": True, "background": self.background.ready,
                       "blobs": [public_blob(b) for b in tracked], "_blobs": tracked}
        self.latest = payload
        return payload


def publishable(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if not k.startswith("_")}


def match_truth(blobs: list[dict], items: list[dict], max_dist_m: float = 0.04) -> dict[str, str]:
    """Greedy nearest match of blobs to ground-truth items -> {"<blob id>": "<item>[:<state>]"}."""
    pairs = sorted((math.dist(b["table_xy"], it["table_xy"]), str(b["id"]), k)
                   for b in blobs for k, it in enumerate(items))
    out: dict[str, str] = {}
    used = set()
    for d, bid, k in pairs:
        if d > max_dist_m or bid in out or k in used:
            continue
        it = items[k]
        out[bid] = it["item"] + (f":{it['state']}" if it.get("state") else "")
        used.add(k)
    return out


# ---------------------------------------------------------------- marks

PALETTE = [(0, 220, 255), (255, 200, 0), (80, 255, 80), (255, 90, 255), (0, 150, 255),
           (255, 255, 120), (120, 120, 255), (200, 255, 0)]
WHITE, BLACK = (255, 255, 255), (0, 0, 0)


def _box_table(b: dict) -> np.ndarray:
    if b.get("box") is not None:
        return np.asarray(b["box"], dtype=np.float64)
    x, y = b["table_xy"]
    c, s = math.cos(math.radians(b["yaw_deg"])), math.sin(math.radians(b["yaw_deg"]))
    hl, hw = b["length_m"] / 2.0, b["width_m"] / 2.0
    loc = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    return loc @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def annotate_marks(image: Any, blobs: list[dict], calib: Any, background: bool | None = None,
                   quality: int = 85) -> bytes:
    """JPEG of the original camera frame with a big numbered mark ("#3") beside each blob, a
    leader line to its centre and a thin outline of its box. `calib` is a Calibration (uses
    .H) or a pixel -> table homography; None draws no marks (not calibrated)."""
    img = to_bgr(image).copy()
    hgt, wid = img.shape[:2]
    k = max(wid / 1280.0, 0.4)
    H = getattr(calib, "H", calib)
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thick = 1.45 * k, max(2, int(round(3 * k)))
    pad = int(round(7 * k))
    line = max(2, int(round(2 * k)))
    if H is not None and blobs:
        G = np.linalg.inv(np.asarray(H, dtype=np.float64))     # table -> pixel
        q = 4                                                   # occupancy grid at 1/4 scale
        occ = np.zeros((hgt // q + 1, wid // q + 1), np.uint8)
        polys = {}
        for b in blobs:
            p = apply_h(G, _box_table(b))
            polys[b["id"]] = p
            cv2.fillPoly(occ, [np.round(p / q).astype(np.int32)], 1)
        for n, b in enumerate(sorted(blobs, key=lambda b: b["id"])):
            col = PALETTE[b["id"] % len(PALETTE)]
            p = polys[b["id"]]
            cv2.polylines(img, [np.round(p).astype(np.int32)], True, col, line, cv2.LINE_AA)
            cu, cv = apply_h(G, [b["table_xy"]])[0]
            text = f"#{b['id']}"
            (tw, th), base = cv2.getTextSize(text, font, scale, thick)
            bw, bh = tw + 2 * pad, th + base + 2 * pad
            # candidate spots: either side across the long axis, then beyond the ends, diagonals
            ax = p[1] - p[0] if np.linalg.norm(p[1] - p[0]) >= np.linalg.norm(p[2] - p[1]) else p[2] - p[1]
            ax = ax / max(np.linalg.norm(ax), 1e-6)
            nrm = np.array([-ax[1], ax[0]])
            half_len = max(float(np.linalg.norm(p[1] - p[0])), float(np.linalg.norm(p[2] - p[1]))) / 2
            half_wid = min(float(np.linalg.norm(p[1] - p[0])), float(np.linalg.norm(p[2] - p[1]))) / 2
            rad = 0.5 * math.hypot(bw, bh) + 6 * k
            cands = []
            for sgn in (-1, 1):
                cands.append(np.array([cu, cv]) + nrm * sgn * (half_wid + rad))
            for sgn in (-1, 1):
                cands.append(np.array([cu, cv]) + ax * sgn * (half_len + rad))
            for sa in (-1, 1):
                for sn in (-1, 1):
                    cands.append(np.array([cu, cv]) + (ax * sa * half_len + nrm * sn * half_wid) +
                                 (ax * sa + nrm * sn) / math.sqrt(2) * rad)
            best, best_cost = None, None
            for i, c in enumerate(cands):
                x0, y0 = int(round(c[0] - bw / 2)), int(round(c[1] - bh / 2))
                off = max(0, -x0) + max(0, -y0) + max(0, x0 + bw - wid) + max(0, y0 + bh - hgt)
                xa, ya = max(x0, 0) // q, max(y0, 0) // q
                xb, yb = min(x0 + bw, wid) // q + 1, min(y0 + bh, hgt) // q + 1
                cost = float(occ[ya:yb, xa:xb].sum()) * 10 + off * 50 + i
                if best_cost is None or cost < best_cost:
                    best, best_cost = (x0, y0), cost
            x0, y0 = best
            x0 = min(max(x0, 0), wid - bw)
            y0 = min(max(y0, 0), hgt - bh)
            occ[y0 // q:(y0 + bh) // q + 1, x0 // q:(x0 + bw) // q + 1] = 1
            bc = np.array([x0 + bw / 2, y0 + bh / 2])
            # leader from the box edge towards the centre
            d = np.array([cu, cv]) - bc
            if np.linalg.norm(d) > 1:
                t = min(bw / 2 / max(abs(d[0]), 1e-6), bh / 2 / max(abs(d[1]), 1e-6), 1.0)
                start = bc + d * t
                cv2.line(img, tuple(int(round(v)) for v in start), (int(round(cu)), int(round(cv))),
                         BLACK, line + 2, cv2.LINE_AA)
                cv2.line(img, tuple(int(round(v)) for v in start), (int(round(cu)), int(round(cv))),
                         col, line, cv2.LINE_AA)
            cv2.circle(img, (int(round(cu)), int(round(cv))), line + 3, BLACK, -1, cv2.LINE_AA)
            cv2.circle(img, (int(round(cu)), int(round(cv))), line + 1, col, -1, cv2.LINE_AA)
            cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), BLACK, -1)
            cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), col, line)
            cv2.putText(img, text, (x0 + pad, y0 + pad + th), font, scale, WHITE, thick, cv2.LINE_AA)
    # status line, bottom left (perception.annotate uses the top left)
    if H is None:
        status = "MARKS: NOT CALIBRATED"
    else:
        status = f"{len(blobs)} marks" + ("" if background is None else
                                         ("  background ok" if background else "  NO BACKGROUND (approx.)"))
    s2, t2 = 0.7 * k, max(1, int(round(2 * k)))
    (sw, sh), sb = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, s2, t2)
    y = hgt - int(12 * k)
    cv2.rectangle(img, (8, y - sh - 6), (8 + sw + 10, y + sb + 2), BLACK, -1)
    cv2.putText(img, status, (13, y), cv2.FONT_HERSHEY_SIMPLEX, s2, WHITE, t2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG encode failed")
    return buf.tobytes()
