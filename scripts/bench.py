"""Standalone hardware bench: webcam -> AprilTag table frame -> blobs -> reBot arm.

No swarm, no LLM, no networking. Reuses swarmmind/cell/{perception,blobs,arm}.py.

    .venv\\Scripts\\python scripts\\bench.py cam                 # camera + tags + blobs (no arm)
    .venv\\Scripts\\python scripts\\bench.py arm-read            # READ-ONLY motor status (never enables)
    .venv\\Scripts\\python scripts\\bench.py arm-jog             # jog joints (type ENABLE first)
    .venv\\Scripts\\python scripts\\bench.py calib-base          # touch tag centres -> arm base pose
    .venv\\Scripts\\python scripts\\bench.py pick [--dry-run] [--mock]

See docs/BENCH.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swarmmind.cell import perception as P  # noqa: E402
from swarmmind.cell import blobs as B  # noqa: E402

OUT = ROOT / "data" / "bench"
CALIB_FILE = OUT / "bench_calib.json"

# The real bench (measured): tag36h11, 5 cm black squares, tags 0-3 on a 0.458 x 0.585 m rectangle.
# The tags lie rotated: their printed top edge points along table -y (yaw -90, measured live).
BENCH_CELL = {
    "tag_size_m": 0.05,
    "table_tags": {0: [0.0, 0.0, -90.0], 3: [0.458, 0.0, -90.0], 1: [0.0, 0.585, -90.0],
                   2: [0.458, 0.585, -90.0]},
    "objects": {},
}
DEFAULT_BASE = [-0.08, 0.20, 0.0, 0.0]   # rough guess until calib-base is run


# ================================================================ camera

def open_camera(index: int, width: int = 1920, height: int = 1080) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    if not ok or frame is None:
        cap.release()
        sys.exit(f"camera index {index} gives no frames. Probe with:\n  .venv\\Scripts\\python -c "
                 "\"import cv2; [print(i, cv2.VideoCapture(i, cv2.CAP_DSHOW).read()[0]) for i in range(4)]\"")
    if frame.mean() < 3:
        print(f"WARNING: camera {index} delivers black frames (integrated camera with shutter closed?)")
    print(f"camera {index}: {frame.shape[1]}x{frame.shape[0]}")
    return cap


class Vision:
    """Camera frame -> tags, calibration, blobs; draws the overlay; maps clicks to table x/y."""

    def __init__(self, cell: dict, base: list[float] | None = None):
        self.cell = P.normalize_cell(cell)
        self.calib = P.Calibration(cell)
        self.pipe = B.BlobPipeline(cell)
        self.base = base
        self.tags: list[dict] = []
        self.blobs: list[dict] = []
        self.frame: np.ndarray | None = None
        self.bg_buf: list[tuple] | None = None
        self.bg_until = 0.0
        self.marks: list[tuple[str, tuple[float, float], tuple]] = []   # (label, table xy, colour)
        self.status = ""

    @property
    def H(self):
        return self.calib.H if self.calib.valid() else None

    def process(self, frame: np.ndarray) -> None:
        now = time.time()
        self.frame = frame
        self.tags = P.detect_tags(frame)
        self.calib.update(self.tags, now)
        H = self.H
        if self.bg_buf is not None:
            if H is not None:
                self.bg_buf.append((frame.copy(), H.copy(), self.tags))
            if now >= self.bg_until:
                buf, self.bg_buf = self.bg_buf, None
                if len(buf) >= 2:
                    step = max(1, len(buf) // B.BG_FRAMES)
                    sel = buf[::step][:B.BG_FRAMES]
                    self.pipe.capture_background([b[0] for b in sel], [b[1] for b in sel],
                                                 [b[2] for b in sel], now)
                    print(f"background captured from {len(sel)} frames")
                else:
                    print("background FAILED: not calibrated during capture (all 4 table tags visible?)")
        if H is not None and self.pipe.background.ready:
            self.blobs = self.pipe.process(frame, H, self.tags, now)["_blobs"]
        else:
            self.blobs = []

    def start_background(self, seconds: float = 1.0) -> None:
        print("capturing background: keep the table EMPTY and hands out ...")
        self.bg_buf, self.bg_until = [], time.time() + seconds

    def px_to_table(self, u: float, v: float):
        H = self.H
        return None if H is None else tuple(P.apply_h(H, [[u, v]])[0])

    def table_to_px(self, xy):
        H = self.H
        if H is None:
            return None
        return tuple(int(round(a)) for a in P.apply_h(np.linalg.inv(H), [xy[:2]])[0])

    def draw(self, fps: float) -> np.ndarray:
        img = self.frame.copy()
        k = max(img.shape[1] / 1280.0, 0.5)
        th = max(2, int(round(2 * k)))
        H = self.H
        if H is not None:
            G = np.linalg.inv(H)
            x0, y0, x1, y1 = B.table_extent(self.cell, 0.0)
            for x in np.arange(x0, x1 + 1e-6, 0.05):
                a, b = P.apply_h(G, [[x, y0], [x, y1]])
                cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), (90, 90, 90), 1, cv2.LINE_AA)
            for y in np.arange(y0, y1 + 1e-6, 0.05):
                a, b = P.apply_h(G, [[x0, y], [x1, y]])
                cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), (90, 90, 90), 1, cv2.LINE_AA)
            o, ax, ay = P.apply_h(G, [[0, 0], [0.1, 0], [0, 0.1]])
            cv2.arrowedLine(img, tuple(np.int32(o)), tuple(np.int32(ax)), P.RED, th + 1, tipLength=0.2)
            cv2.arrowedLine(img, tuple(np.int32(o)), tuple(np.int32(ay)), P.GREEN, th + 1, tipLength=0.2)
            if self.base:
                p = self.table_to_px(self.base)
                cv2.drawMarker(img, p, (255, 0, 255), cv2.MARKER_TILTED_CROSS, int(30 * k), th)
                P._label(img, "arm base", (p[0] + 10, p[1] - 10), 0.6 * k, (255, 0, 255), max(1, th - 1))
        for t in self.tags:
            c = np.round(np.array(t["corners"])).astype(np.int32)
            col = P.GREEN if t["id"] in self.cell["table_tags"] else P.ORANGE
            cv2.polylines(img, [c], True, col, th, cv2.LINE_AA)
            u, v = (int(a) for a in t["px"])
            P._label(img, f"id {t['id']}", (u - 20, v - int(30 * k)), 0.8 * k, col, th)
        # blobs: number + table x/y in cm
        if H is not None:
            G = np.linalg.inv(H)
            for b in self.blobs:
                col = B.PALETTE[b["id"] % len(B.PALETTE)]
                poly = P.apply_h(G, B._box_table(b))
                cv2.polylines(img, [np.round(poly).astype(np.int32)], True, col, th, cv2.LINE_AA)
                cu, cv_ = P.apply_h(G, [b["table_xy"]])[0]
                cv2.circle(img, (int(cu), int(cv_)), th + 3, col, -1)
                x, y = b["table_xy"]
                P._label(img, f"#{b['id']}", (int(cu) + 12, int(cv_) - 12), 1.6 * k, col, th + 1)
                P._label(img, f"{x * 100:.1f},{y * 100:.1f} cm", (int(cu) + 12, int(cv_) + int(30 * k)),
                         0.7 * k, P.WHITE, max(1, th - 1))
        for label, xy, col in self.marks:
            p = self.table_to_px(xy)
            if p:
                cv2.drawMarker(img, p, col, cv2.MARKER_CROSS, int(40 * k), th + 1)
                cv2.circle(img, p, int(20 * k), col, th)
                P._label(img, label, (p[0] + 14, p[1] + int(40 * k)), 0.9 * k, col, th)
        if H is not None:
            s, col = f"CALIBRATED  err {self.calib.reproj_err_px:.2f} px", P.GREEN
        else:
            s, col = f"NOT CALIBRATED ({self.calib.last_error or 'no fit'})", P.RED
        P._label(img, s, (12, int(40 * k)), 1.0 * k, col, th)
        bg = ("bg: capturing..." if self.bg_buf is not None else
              "bg: ok" if self.pipe.background.ready else "bg: none (press b on EMPTY table)")
        P._label(img, f"{fps:4.1f} fps  tags {[t['id'] for t in self.tags]}  {bg}  blobs {len(self.blobs)}",
                 (12, int(80 * k)), 0.7 * k, P.WHITE, max(1, th - 1))
        if self.status:
            P._label(img, self.status, (12, img.shape[0] - int(20 * k)), 0.9 * k, (0, 255, 255), th)
        return img


class Window:
    """cv2 window scaled to fit the screen; clicks reported in full-frame pixels."""

    def __init__(self, name: str, display_w: int = 1280):
        self.name, self.display_w, self.scale = name, display_w, 1.0
        self.clicks: list[tuple[float, float]] = []
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(name, self._mouse)

    def _mouse(self, ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.clicks.append((x / self.scale, y / self.scale))

    def show(self, img: np.ndarray) -> int:
        self.scale = min(1.0, self.display_w / img.shape[1])
        disp = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA) \
            if self.scale < 1.0 else img
        cv2.imshow(self.name, disp)
        return cv2.waitKey(1) & 0xFF

    def pop_click(self):
        return self.clicks.pop(0) if self.clicks else None


def snapshot(vis: Vision, img: np.ndarray) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(str(OUT / f"snap_{stamp}_raw.jpg"), vis.frame)
    cv2.imwrite(str(OUT / f"snap_{stamp}_annot.jpg"), img)
    (OUT / f"snap_{stamp}.json").write_text(json.dumps({
        "tags": vis.tags, "calibrated": vis.H is not None, "reproj_err_px": vis.calib.reproj_err_px,
        "blobs": [B.public_blob(b) for b in vis.blobs]}, indent=1))
    print(f"saved {OUT / ('snap_' + stamp)}_*.jpg/.json")


class Fps:
    def __init__(self):
        self.t, self.v = time.time(), 0.0

    def tick(self) -> float:
        now = time.time()
        dt, self.t = now - self.t, now
        self.v = 0.9 * self.v + 0.1 * (1.0 / max(dt, 1e-3)) if self.v else 1.0 / max(dt, 1e-3)
        return self.v


def cmd_cam(args) -> None:
    cap = open_camera(args.camera)
    vis = Vision(BENCH_CELL, load_base(args))
    win = Window("bench cam  [b]=background  [s]=snapshot  [click]=coords  [q]=quit")
    fps = Fps()
    t_end = time.time() + args.seconds if args.seconds else None
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed")
                break
            vis.process(frame)
            n += 1
            img = vis.draw(fps.tick())
            key = win.show(img)
            c = win.pop_click()
            if c:
                xy = vis.px_to_table(*c)
                if xy is None:
                    print(f"click px ({c[0]:.0f},{c[1]:.0f}): not calibrated")
                else:
                    vis.marks = [("click", xy, (0, 255, 255))]
                    msg = f"click px ({c[0]:.0f},{c[1]:.0f}) -> table x={xy[0]*100:.1f} cm  y={xy[1]*100:.1f} cm"
                    if vis.base:
                        a = P.table_to_arm([xy[0], xy[1], 0.0], vis.base)
                        msg += f"  | arm ({a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}) m"
                    print(msg)
            if key == ord("q") or key == 27:
                break
            if key == ord("b"):
                vis.start_background()
            if key == ord("s"):
                snapshot(vis, img)
            if t_end and time.time() > t_end:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"{n} frames, last: tags {[t['id'] for t in vis.tags]}, calibrated={vis.H is not None}, "
              f"err={vis.calib.reproj_err_px}")


# ================================================================ calibration file

def load_base(args) -> list[float] | None:
    if getattr(args, "base", None):
        return [float(v) for v in args.base.split(",")]
    if CALIB_FILE.exists():
        return json.loads(CALIB_FILE.read_text())["base"]
    return None


# ================================================================ main

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cam", help="live camera, tags, grid, blobs")
    c.add_argument("--camera", type=int, default=1)
    c.add_argument("--seconds", type=float, default=0, help="auto-quit after N s (testing)")
    c.add_argument("--base", default="", help="arm base x,y,yaw_deg,z_table (default: bench_calib.json)")
    args = ap.parse_args(argv)
    {"cam": cmd_cam}[args.cmd](args)


if __name__ == "__main__":
    main()
