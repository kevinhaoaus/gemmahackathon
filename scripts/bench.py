"""Standalone hardware bench: webcam -> AprilTag table frame -> blobs -> reBot arm.

No swarm, no LLM, no networking. Reuses swarmmind/cell/{perception,blobs,arm}.py.

    .venv\\Scripts\\python scripts\\bench.py cam                 # camera + tags + blobs (no arm)
    .venv\\Scripts\\python scripts\\bench.py arm-read            # READ-ONLY motor status (never enables)
    .venv\\Scripts\\python scripts\\bench.py arm-jog             # jog joints (type ENABLE first)
    .venv\\Scripts\\python scripts\\bench.py calib-base          # touch tag centres -> arm base pose
    .venv\\Scripts\\python scripts\\bench.py pick [--dry-run]       # click object, click destination, y

See docs/BENCH.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
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

def load_calib() -> dict:
    try:
        return json.loads(CALIB_FILE.read_text())
    except (OSError, ValueError):
        return {}


def save_calib(**kv) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_calib()
    d.update(kv)
    CALIB_FILE.write_text(json.dumps(d, indent=1))
    print(f"saved {CALIB_FILE}: {kv}")


def load_base(args) -> list[float] | None:
    if getattr(args, "base", None):
        return [float(v) for v in args.base.split(",")]
    return load_calib().get("base")


# ================================================================ arm (reBot B601-DM via motorbridge)

# Damiao status codes (high nibble of the feedback's first byte), for the printout only.
DM_STATUS = {0: "disabled", 1: "enabled", 8: "OVERVOLTAGE", 9: "UNDERVOLTAGE", 10: "OVERCURRENT",
             11: "MOS OVERTEMP", 12: "ROTOR OVERTEMP", 13: "COMM LOST", 14: "OVERLOAD"}


def list_com_ports() -> list[str]:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\SERIALCOMM") as k:
            out, i = [], 0
            while True:
                try:
                    out.append(str(winreg.EnumValue(k, i)[1]))
                except OSError:
                    return sorted(out)
                i += 1
    except OSError:
        return []


def make_driver(args, base: list[float] | None = None):
    """RebotArm (not connected). Slow defaults; workspace derived from the base's z_table."""
    from swarmmind.cell.drivers.rebot import RebotArm
    cal = load_calib()
    bz = (base or [0, 0, 0, 0.0])[3]
    cfg = {
        "driver": "rebot", "variant": "dm", "port": args.port, "baud": args.baud,
        # arm base frame; the floor is 5 mm above the table surface (table z_arm = -z_table)
        "workspace": {"x": [-0.55, 0.55], "y": [-0.55, 0.55], "z": [-bz + 0.005, -bz + 0.40]},
        "joint_speed_max": math.radians(args.joint_speed),
        "max_speed": 0.05, "speed": 0.03, "slow_speed": 0.015, "retreat_speed": 0.02,
        "approach_h": getattr(args, "approach", 0.08),
        # placeholder 0.0 when unknown: grip commands are refused until it is measured
        "gripper_open_rad": cal.get("gripper_open_rad", 0.0),
        "gripper_closed_rad": cal.get("gripper_closed_rad", 0.0),
        "gripper_vlim": 1.0,
        "gripper_squeeze_rad": math.radians(getattr(args, "grip_squeeze_deg", 3.0)),
        # force_pos = Damiao force-position mode (a real force cap); posvel = position + contact back-off
        "gripper_mode": "force_pos" if getattr(args, "grip_force", 0) else "posvel",
        "gripper_force_ratio": getattr(args, "grip_force", 0) or 0.3,
    }
    return RebotArm(cfg)


def gripper_known() -> bool:
    c = load_calib()
    return c.get("gripper_open_rad") is not None and c.get("gripper_closed_rad") is not None


def open_bus(drv) -> None:
    """Open the serial bridge and register the motors. Sends NOTHING to the motors."""
    try:
        import motorbridge  # noqa: F401
    except ImportError:
        sys.exit("motorbridge (Seeed's reBot SDK) is not installed. With internet, run:\n"
                 "  .venv\\Scripts\\pip install motorbridge")
    present = P_port_present(drv.port)
    if present is False:
        sys.exit(f"{drv.port} is not present. COM ports right now: {list_com_ports() or 'none'}\n"
                 f"  -> USB cable unplugged? Device Manager > Ports: the bridge shows as 'USB Serial Device "
                 f"(COMx)' (VID 2E88 PID 4603). Pass --port COMx.")
    try:
        drv._open_sync()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"could not open {drv.port} @ {drv.baud}: {type(e).__name__}: {e}\n"
                 f"  -> port in use by another program (another bench, Seeed tool, serial monitor)? close it.\n"
                 f"  -> wrong port? COM ports: {list_com_ports()}")


def P_port_present(port):
    from swarmmind.cell.drivers.rebot import port_present
    return port_present(port)


def close_bus(drv, disable: bool) -> None:
    drv.disable_on_close = disable
    try:
        drv._close_sync()
    except Exception as e:  # noqa: BLE001
        print(f"close: {e}")


def read_states(drv, tries: int = 3) -> list:
    """[(name, state or None)] after `tries` feedback requests (status query frames only)."""
    for _ in range(tries):
        for m in drv.motors:
            m.request_feedback()
        drv.ctrl.poll_feedback_once()
        time.sleep(0.02)
    return [(spec[0], m.get_state()) for spec, m in zip(drv.motor_spec, drv.motors)]


def print_states(drv, states) -> int:
    ok = 0
    for name, st in states:
        if st is None:
            print(f"  {name:8s} NO FEEDBACK")
            continue
        ok += 1
        code = getattr(st, "status_code", None)
        print(f"  {name:8s} pos {math.degrees(st.pos):+8.2f} deg ({st.pos:+.4f} rad)  "
              f"vel {getattr(st, 'vel', float('nan')):+.3f}  torq {getattr(st, 'torq', float('nan')):+.3f}  "
              f"T_mos {getattr(st, 't_mos', float('nan')):.0f}C T_rotor {getattr(st, 't_rotor', float('nan')):.0f}C  "
              f"status {code} {DM_STATUS.get(code, '') if isinstance(code, int) else ''}")
    if ok >= 6 and all(st is not None for _, st in states[:6]):
        q = drv.motor_to_urdf(np.array([st.pos for _, st in states[:6]]))
        p = drv.tool_pose(q)
        print(f"  FK tool (arm base frame): x {p['x']:+.4f} y {p['y']:+.4f} z {p['z']:+.4f} m  "
              f"yaw {p['yaw']:+.1f}  tilt-from-down {p['tilt_deg']:.1f} deg")
    return ok


def cmd_arm_read(args) -> None:
    drv = make_driver(args)
    print(f"READ-ONLY: opening {drv.port} @ {drv.baud} (no enable, no motion, no register writes)")
    open_bus(drv)
    ok = 0
    try:
        for k in range(args.n):
            print(f"--- read {k + 1}/{args.n}")
            ok = print_states(drv, read_states(drv))
            time.sleep(0.5)
    finally:
        close_bus(drv, disable=False)     # strictly read-only: not even a disable frame
    n = len(drv.motor_spec)
    if ok == n:
        print(f"OK: all {n} motors answered. Next: arm-jog.")
    elif ok == 0:
        print("NO MOTOR ANSWERED. Check: 24 V supply on (motor LEDs lit)? e-stop released? CAN cable from the "
              "bridge to the arm seated? Bridge CAN rate 1 Mbit/s? Baud (--baud, default 921600)?")
    else:
        print(f"only {ok}/{n} answered: check the CAN daisy chain at the first silent motor, or its id.")


# ---------------------------------------------------------------- jog shell (sync, main thread)

MAX_JOINT_SPEED_DEG = 60.0     # hard ceiling for `speed` / --joint-speed, deg/s


class Jog:
    """Torque-off by default. Every motion: joint-space, <= max_step per command, speed-capped.
    Thread-safe (the GUI polls status while a motion runs): bus I/O goes through `lock`, and
    `abort` (set by stop) interrupts a motion mid-way and holds where the arm is."""

    def __init__(self, drv, args):
        self.drv, self.args = drv, args
        self.cap = math.radians(args.max_step)
        self.speed = math.radians(min(args.joint_speed, MAX_JOINT_SPEED_DEG))   # rad/s
        self.lock = threading.RLock()
        self.abort = threading.Event()
        self.enabled = False
        self.home_q = None
        self.touches: list[tuple[list[float], list[float]]] = []   # (table xy, FK xyz)

    # -- state
    def read_motor(self) -> np.ndarray:
        with self.lock:
            states = read_states(self.drv, 1)
        bad = [n for n, s in states if s is None]
        if bad:
            raise ConnectionError(f"no feedback from {bad} (24 V off?)")
        return np.array([s.pos for _, s in states])

    def read_q(self) -> np.ndarray:
        return self.drv.motor_to_urdf(self.read_motor())

    # -- torque
    def enable(self, confirm: bool = True) -> None:
        """confirm=False only for callers that already asked (the GUI's confirmation dialog)."""
        if confirm and input("Torque ON. The arm will hold its current pose. Type ENABLE to continue: ").strip() != "ENABLE":
            print("not enabled")
            return
        with self.lock:
            self._enable_locked()

    def _enable_locked(self) -> None:
        drv = self.drv
        for i, m in enumerate(drv.motors):
            try:
                m.clear_error()
            except Exception:  # noqa: BLE001
                pass
            if drv.write_gains:
                for reg, val in zip((25, 26, 27, 28), drv.gains[i]):
                    m.write_register_f32(reg, float(val))
            m.ensure_mode(drv.Mode.POS_VEL, 1000)
        pos = self.read_motor()
        self.hold(pos)                    # targets = where it is, BEFORE enabling (gripper too)
        for m in drv.motors:
            m.enable()
        self.hold(pos)
        self.enabled = True
        self.home_q = self.home_q if self.home_q is not None else drv.motor_to_urdf(pos)
        print("ENABLED. Ctrl-C or 'disable' = torque OFF (the arm goes limp: support it).")

    def hold(self, motor_pos) -> None:
        with self.lock:
            for i, m in enumerate(self.drv.motors):
                m.send_pos_vel(float(motor_pos[i]), 0.3)

    def stop(self) -> None:
        """Interrupt any motion and hold the current position (torque stays ON)."""
        self.abort.set()
        if self.enabled:
            self.hold(self.read_motor())
            print("holding")

    def disable(self) -> None:
        self.abort.set()
        try:
            with self.lock:
                self.drv.ctrl.disable_all()
        except Exception as e:  # noqa: BLE001
            print(f"disable_all failed: {e}")
        self.enabled = False
        print("torque OFF")

    # -- motion
    def move_q(self, q_target: np.ndarray) -> None:
        if not self.enabled:
            print("not enabled (type: enable)")
            return
        drv = self.drv
        q0 = self.read_q()
        d = np.asarray(q_target, float) - q0
        big = float(np.max(np.abs(d)))
        if big > self.cap + 1e-9:
            print(f"refused: joint delta {math.degrees(big):.1f} deg > cap {math.degrees(self.cap):.1f} deg")
            return
        lo, hi = drv.chain.lower, drv.chain.upper
        for j in range(6):
            t = q_target[j]
            if (t < lo[j] and t < q0[j]) or (t > hi[j] and t > q0[j]):
                print(f"refused: j{j + 1} target {math.degrees(t):.1f} deg outside URDF limit "
                      f"[{math.degrees(lo[j]):.0f}, {math.degrees(hi[j]):.0f}]")
                return
        dur = max(big / self.speed, 0.3)
        n = max(1, math.ceil(dur * 50))
        vlim = np.full(6, min(1.5 * self.speed, math.radians(1.5 * MAX_JOINT_SPEED_DEG)))
        self.abort.clear()
        t0 = time.monotonic()
        for k in range(1, n + 1):
            if self.abort.is_set():
                print("stopped mid-move")
                return
            s = k / n
            s = s * s * (3 - 2 * s)
            with self.lock:
                drv._send_sync(drv.urdf_to_motor(q0 + d * s), vlim)
            delay = t0 + k / 50 - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        t1 = time.monotonic()
        while time.monotonic() - t1 < 2.0:
            err = float(np.max(np.abs(self.read_q() - q_target)))
            if err < 0.02:
                break
            time.sleep(0.05)
        else:
            print(f"WARNING: not settled (max error {math.degrees(err):.1f} deg)")
        self.pose()

    def move_grip(self, target_rad: float) -> None:
        if not self.enabled:
            print("not enabled")
            return
        cur = float(self.read_motor()[6])
        if abs(target_rad - cur) > math.radians(self.args.max_grip_step) + 1e-9:
            print(f"refused: gripper delta {math.degrees(target_rad - cur):.1f} deg > "
                  f"{self.args.max_grip_step} deg (use g +/-deg, or measure open/closed first)")
            return
        with self.lock:
            self.drv.motors[6].send_pos_vel(float(target_rad), 1.0)
        time.sleep(1.0)
        print(f"gripper motor now {math.degrees(float(self.read_motor()[6])):+.1f} deg")

    def pose(self) -> dict:
        m = self.read_motor()
        q = self.drv.motor_to_urdf(m)
        p = self.drv.tool_pose(q)
        print(f"joints deg {np.round(np.degrees(q), 1).tolist()}  gripper {math.degrees(m[6]):+.1f} deg")
        print(f"tool (arm base): x {p['x']:+.4f}  y {p['y']:+.4f}  z {p['z']:+.4f} m  yaw {p['yaw']:+.1f}  "
              f"tilt {p['tilt_deg']:.1f} deg")
        return p

    # -- calibration
    def touch(self, parts: list[str]) -> None:
        tt = BENCH_CELL["table_tags"]
        if len(parts) == 1 and int(parts[0]) in tt:
            xy = tt[int(parts[0])][:2]
        elif len(parts) == 2:
            xy = [float(parts[0]) / 100, float(parts[1]) / 100]       # cm, e.g. a point clicked in cam
        else:
            print("touch <tag 0-3>  or  touch <x_cm> <y_cm>")
            return
        p = self.pose()
        if p["tilt_deg"] > 10:
            print(f"WARNING: tool tilted {p['tilt_deg']:.0f} deg from vertical: xy will be off")
        self.touches.append((list(xy), [p["x"], p["y"], p["z"]]))
        print(f"touch {len(self.touches)}: table {xy} <- tool {np.round([p['x'], p['y'], p['z']], 4).tolist()}")

    def solve(self) -> None:
        if len(self.touches) < 3:
            print(f"need >= 3 touches, have {len(self.touches)}")
            return
        base, res_mm, z_sd = solve_base([t[0] for t in self.touches], [t[1] for t in self.touches])
        print(f"base = [{base[0]:.4f}, {base[1]:.4f}, {base[2]:.2f}, {base[3]:.4f}]   "
              f"xy residual RMS {res_mm:.1f} mm, z spread {z_sd * 1000:.1f} mm")
        if res_mm > 10:
            print("residual > 10 mm: a touch is wrong, or joint signs/offsets don't match the URDF "
                  "(check each joint's direction with j<N> +5 and the printed pose)")
        save_calib(base=[round(v, 4) for v in base], touches=self.touches)
        print("paste into configs/laptop.yaml under arm.cell:\n"
              f"    base: [{base[0]:.4f}, {base[1]:.4f}, {base[2]:.2f}, {base[3]:.4f}]")


def solve_base(table_xy, arm_xyz) -> tuple[list[float], float, float]:
    """Least-squares 2D rigid fit table = R(yaw) @ arm_xy + (bx, by); z_table = -mean(arm z)
    (table_to_arm: z_arm = z - z_table, the surface is z = 0). -> ([bx, by, yaw_deg, z_table],
    xy RMS residual mm, z std m)."""
    T = np.asarray(table_xy, float)
    A = np.asarray(arm_xyz, float)
    a, ca, ct = A[:, :2], A[:, :2].mean(0), T.mean(0)
    a0, t0 = a - ca, T - ct
    yaw = math.atan2(float(np.sum(a0[:, 0] * t0[:, 1] - a0[:, 1] * t0[:, 0])),
                     float(np.sum(a0[:, 0] * t0[:, 0] + a0[:, 1] * t0[:, 1])))
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    b = ct - R @ ca
    res = T - (a @ R.T + b)
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))) * 1000
    return [float(b[0]), float(b[1]), math.degrees(yaw), float(-A[:, 2].mean())], rms, float(A[:, 2].std())


def _self_check_solve_base() -> None:
    """Tiny numeric check: known base -> arm points -> solve -> same base."""
    base = [-0.08, 0.20, 37.0, 0.012]
    tab = [[0, 0], [0.458, 0], [0, 0.585]]
    arm = [list(P.table_to_arm([x, y, 0.0], base)) for x, y in tab]
    got, rms, _ = solve_base(tab, arm)
    assert np.allclose(got, base, atol=1e-9) and rms < 1e-6, (got, base, rms)


JOG_HELP = """commands:
  enable | disable | stop          torque on (type ENABLE) / off (limp!) / hold here
  j<1-6> <+-deg>                   jog one joint, e.g. j2 +5   (cap --max-step deg per command)
  x|y|z <+-mm>                     tool-down Cartesian step (IK; refused if it needs > cap)
  g <+-deg>                        gripper motor step
  grip open|close                  needs set-open/set-closed first
  grip set-open | grip set-closed  store the gripper motor angle now as open / closed
  home | sethome                   go toward the pose at enable (cap per command) / store this pose
  pose | read                      FK tool pose / full motor status
  touch <tag 0-3> | touch <x_cm> <y_cm>   record the tool tip at a known table point (calib)
  solve                            fit the arm base from >= 3 touches -> data/bench/bench_calib.json
  speed <deg/s> | help | q"""


def run_shell(args, calib_mode: bool) -> None:
    drv = make_driver(args)
    print(f"opening {drv.port} (torque stays OFF until you type enable / ENABLE)")
    open_bus(drv)
    jog = Jog(drv, args)
    try:
        if print_states(drv, read_states(drv)) < 7:
            print("not every motor answers: fix that first (arm-read). Continuing read-only is possible.")
        print(JOG_HELP)
        if calib_mode:
            print("\nCALIB-BASE: touch the tool tip (the end_link point you will grip with) to >= 3 known points, "
                  "tool vertical.\n  Either jog (enable, j<N>/x/y/z) or: disable, support the arm and move it "
                  "by hand, then 'touch'.\n  Points: tag centres 0 (0,0)  3 (45.8,0)  1 (0,58.5)  2 (45.8,58.5) cm;"
                  " far ones may be out of reach: use 'touch <x_cm> <y_cm>' for any point you know.\n"
                  "  Then: solve")
        jog.enable()
        while True:
            try:
                line = input(f"[{'ON ' if jog.enabled else 'off'}]> ").strip()
            except EOFError:
                break
            if not line:
                continue
            p = line.split()
            c = p[0].lower()
            try:
                if c in ("q", "quit", "exit"):
                    break
                elif c == "help":
                    print(JOG_HELP)
                elif c == "enable":
                    jog.enable()
                elif c == "disable":
                    jog.disable()
                elif c == "stop":
                    jog.stop()
                elif len(c) == 2 and c[0] == "j" and c[1] in "123456" and len(p) == 2:
                    q = jog.read_q()
                    q[int(c[1]) - 1] += math.radians(float(p[1]))
                    jog.move_q(q)
                elif c in ("x", "y", "z") and len(p) == 2:
                    q0 = jog.read_q()
                    t = drv.tool_pose(q0)
                    t[c] += float(p[1]) / 1000
                    jog.move_q(drv.ik(t["x"], t["y"], t["z"], t["yaw"], q0))
                elif c == "g" and len(p) == 2:
                    jog.move_grip(float(jog.read_motor()[6]) + math.radians(float(p[1])))
                elif c == "grip" and len(p) == 2:
                    if p[1] in ("set-open", "set-closed"):
                        key = "gripper_open_rad" if p[1] == "set-open" else "gripper_closed_rad"
                        save_calib(**{key: round(float(jog.read_motor()[6]), 4)})
                    elif p[1] in ("open", "close"):
                        if not gripper_known():
                            print("measure first: grip set-open / grip set-closed")
                        else:
                            cal = load_calib()
                            jog.move_grip(cal["gripper_open_rad" if p[1] == "open" else "gripper_closed_rad"])
                elif c == "home":
                    if jog.home_q is None:
                        print("no home yet (enable first)")
                    else:
                        q0 = jog.read_q()
                        d = jog.home_q - q0
                        step = d * min(1.0, jog.cap / max(float(np.max(np.abs(d))), 1e-9))
                        jog.move_q(q0 + step)
                        left = float(np.max(np.abs(jog.home_q - jog.read_q())))
                        print("at home" if left < 0.03 else f"{math.degrees(left):.0f} deg to go: repeat home")
                elif c == "sethome":
                    jog.home_q = jog.read_q()
                    print("home = this pose")
                elif c == "pose":
                    jog.pose()
                elif c == "read":
                    print_states(drv, read_states(drv))
                elif c == "touch":
                    jog.touch(p[1:])
                elif c == "solve":
                    jog.solve()
                elif c == "speed" and len(p) == 2:
                    jog.speed = math.radians(min(float(p[1]), MAX_JOINT_SPEED_DEG))
                    print(f"joint speed {math.degrees(jog.speed):.0f} deg/s (max {MAX_JOINT_SPEED_DEG:.0f})")
                else:
                    print("? (help)")
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"error: {type(e).__name__}: {e}")
                if jog.enabled:
                    try:
                        jog.hold(jog.read_motor())
                    except Exception:  # noqa: BLE001
                        pass
    except KeyboardInterrupt:
        print("\nCtrl-C")
    finally:
        if jog.enabled:
            print("disabling torque (arm goes limp)")
        close_bus(drv, disable=jog.enabled)


def cmd_arm_jog(args) -> None:
    run_shell(args, calib_mode=False)


def cmd_calib_base(args) -> None:
    _self_check_solve_base()
    run_shell(args, calib_mode=True)


# ---------------------------------------------------------------- pick (camera + arm)

def fold90(a: float) -> float:
    return (a + 90.0) % 180.0 - 90.0


class ArmRunner:
    """The async arm stack (Safety + pick/place) on a background event loop."""

    def __init__(self, drv):
        import asyncio
        import threading
        from swarmmind.cell.arm import Safety
        self.asyncio = asyncio
        self.drv, self.safety = drv, Safety(drv)
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.fut = None
        self.connected = False
        self.msg = ""

    def run(self, coro, timeout=None):
        return self.asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def connect(self) -> None:
        drv = self.drv
        orig = drv._enable_sync

        def enable_sync():            # as the driver's, but also hold the gripper where it is
            pos = orig()
            drv.motors[6].send_pos_vel(float(pos[6]), 0.5)
            return pos
        drv._enable_sync = enable_sync
        self.run(drv.connect(), 30)
        self.connected = True

    def busy(self) -> bool:
        return self.fut is not None and not self.fut.done()

    def start_job(self, grasp, place_t, yaw, approach, grip_value=None) -> None:
        from swarmmind.cell.arm import pick, place

        async def job():
            gen = self.safety.begin_command()

            async def prog(p):
                self.msg = f"{p['step']}: {p['sub']}"
                print(f"  {self.msg} {p.get('target', '')}")
            await pick(self.safety, *grasp, yaw=yaw, approach_h=approach, progress=prog, gen=gen,
                       grip_value=grip_value)
            await place(self.safety, *place_t, yaw=yaw, approach_h=approach, progress=prog, gen=gen)
            self.msg = "done"
        self.fut = self.asyncio.run_coroutine_threadsafe(job(), self.loop)

    def stop(self) -> None:
        self.run(self.safety.stop("space / Ctrl-C"), 5)

    def close(self) -> None:
        try:
            self.run(self.drv.close(), 10)
        except Exception as e:  # noqa: BLE001
            print(f"close: {e}")


def cmd_pick(args) -> None:
    base = load_base(args)
    if base is None:
        if not args.dry_run:
            sys.exit("no arm base calibration (data/bench/bench_calib.json): run calib-base first, or pass "
                     "--base x,y,yaw,z. (--dry-run works with the rough default.)")
        base = DEFAULT_BASE
        print(f"WARNING: uncalibrated, using rough base {base}")
    if not args.dry_run and not gripper_known():
        sys.exit("gripper open/closed angles not measured: arm-jog -> 'grip set-open' / 'grip set-closed'")
    drv = make_driver(args, base)            # kinematics usable without connecting (dry-run IK check)
    runner = None if args.dry_run else ArmRunner(drv)
    safety = runner.safety if runner else __import__("swarmmind.cell.arm", fromlist=["Safety"]).Safety(drv)
    cap = open_camera(args.camera)
    vis = Vision(BENCH_CELL, base)
    win = Window("bench pick  click object, click destination, y=execute  SPACE=stop  b=bg  q=quit")
    fps = Fps()
    state, sel, plan = "pick", None, None
    vis.status = "click the OBJECT (or press its number)" + ("   [DRY RUN]" if args.dry_run else "")
    t_end = time.time() + args.seconds if args.seconds else None

    def plan_for(src, dst):
        sx, sy, syaw = src[:3]
        dx, dy = dst
        yaw = fold90(P.yaw_table_to_arm(syaw + 90.0, base))       # fingers close across the long axis
        g = P.table_to_arm([sx, sy, args.height], base)
        pl = P.table_to_arm([dx, dy, args.height], base)
        for name, t in (("grasp", g), ("place", pl)):
            for z in (t[2], t[2] + args.approach):
                safety.check(t[0], t[1], z, yaw, allow_clamp=False)
                if math.hypot(t[0], t[1]) < 0.12:
                    raise ValueError(f"{name} is within 12 cm of the arm base axis")
                drv.check_reachable(t[0], t[1], z, yaw)            # IK, tool down
        return g, pl, yaw

    def grip_for(blob) -> tuple[float | None, str]:
        """Linear model: gripper value from the object's width seen by the camera (short side)."""
        from swarmmind.cell.arm import width_to_grip
        if blob is None or not blob.get("width_m"):
            return None, "no object width (raw click): closes until contact"
        w = float(blob["width_m"])
        v = width_to_grip(w, args.grip_open_cm / 100, args.squeeze_mm / 1000)
        return v, (f"object width {w * 100:.1f} cm -> gripper to {v * args.grip_open_cm:.1f} cm "
                   f"(value {v:.2f}, squeeze {args.squeeze_mm:.0f} mm)")

    def select(xy, blob=None):
        nonlocal state, sel
        gv, gmsg = grip_for(blob)
        sel = ((blob["table_xy"][0], blob["table_xy"][1], blob["yaw_deg"], gv) if blob
               else (xy[0], xy[1], 0.0, gv))
        print(f"grip: {gmsg}")
        vis.marks = [("PICK", sel[:2], (0, 255, 255))]
        state = "place"
        vis.status = "click the DESTINATION   (n = cancel)"
        print(f"pick: table ({sel[0] * 100:.1f}, {sel[1] * 100:.1f}) cm, yaw {sel[2]:.0f}" +
              (f"  [blob #{blob['id']}]" if blob else ""))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed")
                break
            vis.process(frame)
            if runner and runner.busy():
                vis.status = f"MOVING: {runner.msg}   SPACE = STOP"
            elif runner and runner.fut is not None and state == "run":
                try:
                    runner.fut.result()
                    print("pick + place done")
                except Exception as e:  # noqa: BLE001
                    print(f"job ended: {type(e).__name__}: {e}")
                state, vis.marks = "pick", []
                vis.status = "click the OBJECT (or press its number)"
            key = win.show(vis.draw(fps.tick()))
            if key == ord(" ") and runner:
                runner.stop()
                print("STOP")
            if key in (ord("q"), 27):
                if runner and runner.busy():
                    runner.stop()
                break
            if key == ord("b"):
                vis.start_background()
            if key == ord("n") and state in ("place", "confirm"):
                state, vis.marks = "pick", []
                vis.status = "click the OBJECT (or press its number)"
            if state == "pick" and ord("1") <= key <= ord("9"):
                b = next((b for b in vis.blobs if b["id"] == key - ord("0")), None)
                if b:
                    select(None, b)
                else:
                    print(f"no blob #{key - ord('0')}")
            c = win.pop_click()
            if c and state in ("pick", "place"):
                xy = vis.px_to_table(*c)
                if xy is None:
                    print("not calibrated: can't map the click")
                elif state == "pick":
                    near = [b for b in vis.blobs if math.dist(b["table_xy"], xy) < 0.04]
                    select(xy, min(near, key=lambda b: math.dist(b["table_xy"], xy)) if near else None)
                else:
                    try:
                        g, pl, yaw = plan_for(sel, xy)
                    except Exception as e:  # noqa: BLE001
                        print(f"REFUSED: {e}")
                        vis.status = f"REFUSED: {str(e)[:70]}  (click another destination, n = cancel)"
                        continue
                    plan = (g, pl, yaw)
                    vis.marks = [("PICK", sel[:2], (0, 255, 255)), ("PLACE", xy, (255, 0, 255))]
                    print(f"plan (arm base frame, m): grasp ({g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f})  "
                          f"place ({pl[0]:.3f}, {pl[1]:.3f}, {pl[2]:.3f})  yaw {yaw:.0f} deg  "
                          f"approach +{args.approach:.2f} m  speeds travel {safety.speed} / descend "
                          f"{safety.slow_speed} m/s")
                    state = "confirm"
                    vis.status = ("DRY RUN: plan ok (n = new pick)" if args.dry_run
                                  else "press y to EXECUTE, n to cancel")
            if state == "confirm" and key == ord("y") and not args.dry_run:
                if not runner.connected:
                    print("enabling the arm (torque ON) ...")
                    runner.connect()
                g, pl, yaw = plan
                runner.start_job(g, pl, yaw, args.approach, grip_value=sel[3])
                state = "run"
            if t_end and time.time() > t_end:
                break
    except KeyboardInterrupt:
        print("\nCtrl-C")
        if runner and runner.connected:
            runner.stop()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if runner and runner.connected:
            print("disabling torque (arm goes limp)")
            runner.close()


# ================================================================ main

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def arm_args(p):
        p.add_argument("--port", default="COM5")
        p.add_argument("--grip-squeeze-deg", type=float, default=3.0,
                       help="on contact, hold this far past where the fingers stopped (gentle grip)")
        p.add_argument("--grip-force", type=float, default=0.0,
                       help="0 = position mode; 0.05-1 = Damiao force-position mode with this force ratio")
        p.add_argument("--baud", type=int, default=921600)
        p.add_argument("--joint-speed", type=float, default=30.0, help="deg/s joint speed cap (default 30, max 60)")

    c = sub.add_parser("cam", help="live camera, tags, grid, blobs")
    c.add_argument("--camera", type=int, default=1)
    c.add_argument("--seconds", type=float, default=0, help="auto-quit after N s (testing)")
    c.add_argument("--base", default="", help="arm base x,y,yaw_deg,z_table (default: bench_calib.json)")
    r = sub.add_parser("arm-read", help="READ-ONLY motor status")
    arm_args(r)
    r.add_argument("-n", type=int, default=3, help="number of reads")
    for name in ("arm-jog", "calib-base", "jog-gui"):
        j = sub.add_parser(name)
        arm_args(j)
        j.add_argument("--max-step", type=float, default=30.0, help="max joint delta per command, deg (default 30)")
        j.add_argument("--max-grip-step", type=float, default=90.0, help="max gripper delta per command, deg")
        j.add_argument("--seconds", type=float, default=0, help=argparse.SUPPRESS)   # GUI smoke test
    k = sub.add_parser("pick", help="camera + arm pick and place")
    arm_args(k)
    k.add_argument("--camera", type=int, default=1)
    k.add_argument("--base", default="")
    k.add_argument("--height", type=float, default=0.02, help="grip height above the table, m")
    k.add_argument("--approach", type=float, default=0.08, help="approach height above grip, m")
    k.add_argument("--dry-run", action="store_true", help="plan and show, never connect or move")
    k.add_argument("--grip-open-cm", type=float, default=10.0, help="finger gap when fully open (linear model)")
    k.add_argument("--squeeze-mm", type=float, default=5.0, help="close this much narrower than the object")
    k.add_argument("--seconds", type=float, default=0, help="auto-quit after N s (testing)")
    args = ap.parse_args(argv)
    if args.cmd == "jog-gui":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from jog_gui import run_gui
        run_gui(args)
        return
    {"cam": cmd_cam, "arm-read": cmd_arm_read, "arm-jog": cmd_arm_jog, "calib-base": cmd_calib_base,
     "pick": cmd_pick}[args.cmd](args)


if __name__ == "__main__":
    main()
