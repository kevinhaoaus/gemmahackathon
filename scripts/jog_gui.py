"""reBot jog GUI (tkinter): connect, enable, jog joints / Cartesian / gripper, calibrate, all with
buttons. Same safety logic as `bench.py arm-jog` (it drives the same Jog class).

    .venv\\Scripts\\python scripts\\bench.py jog-gui [--port COM5] [--max-step 30] [--joint-speed 30]

Keys: SPACE = STOP (hold here). Closing the window asks whether to turn torque off.
"""
from __future__ import annotations

import io
import math
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

import bench as B

POLL_S = 0.25
GRIP_OPEN_CM = 10.0      # finger gap when fully open; linear model: 0 cm = closed, 10 cm = open


class _Tee(io.TextIOBase):
    """print() from the Jog code -> the GUI log (and the terminal)."""

    def __init__(self, q: queue.Queue, orig):
        self.q, self.orig = q, orig

    def write(self, s: str) -> int:
        if s:
            self.q.put(s)
            try:
                self.orig.write(s)
            except Exception:  # noqa: BLE001
                pass
        return len(s)

    def flush(self) -> None:
        try:
            self.orig.flush()
        except Exception:  # noqa: BLE001
            pass


class JogGui:
    def __init__(self, root: tk.Tk, args):
        self.root, self.args = root, args
        self.drv = None
        self.jog: B.Jog | None = None
        self.work: queue.Queue = queue.Queue()
        self.logq: queue.Queue = queue.Queue()
        self.busy = False
        self.state: dict = {}
        sys.stdout = _Tee(self.logq, sys.__stdout__)
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._poller, daemon=True).start()
        self._build()
        root.bind("<space>", lambda e: self.stop())
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(100, self._tick)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        r = self.root
        r.title("reBot jog")
        style = ttk.Style(r)
        style.configure("Big.TButton", font=("Segoe UI", 14, "bold"), padding=8)
        style.configure("Jog.TButton", font=("Segoe UI", 12, "bold"), padding=4, width=4)
        f = ("Segoe UI", 11)

        top = ttk.Frame(r, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Port", font=f).pack(side="left")
        self.port = tk.StringVar(value=self.args.port)
        ttk.Entry(top, textvariable=self.port, width=8, font=f).pack(side="left", padx=4)
        ttk.Button(top, text="Connect (read-only)", command=self.connect).pack(side="left", padx=4)
        ttk.Button(top, text="Disconnect", command=self.disconnect).pack(side="left", padx=4)
        self.status = tk.StringVar(value="not connected")
        self.status_lbl = tk.Label(top, textvariable=self.status, font=("Segoe UI", 13, "bold"), fg="#555")
        self.status_lbl.pack(side="left", padx=16)

        safety = ttk.Frame(r, padding=(8, 0))
        safety.pack(fill="x")
        tk.Button(safety, text="ENABLE torque", font=("Segoe UI", 14, "bold"), bg="#2e7d32", fg="white",
                  command=self.enable).pack(side="left", padx=4, pady=4)
        tk.Button(safety, text="STOP (space)\nhold here", font=("Segoe UI", 16, "bold"), bg="#f9a825",
                  command=self.stop, width=14).pack(side="left", padx=4, pady=4)
        tk.Button(safety, text="DISABLE\n(arm goes limp!)", font=("Segoe UI", 12, "bold"), bg="#c62828",
                  fg="white", command=self.disable).pack(side="left", padx=4, pady=4)

        opts = ttk.Frame(r, padding=8)
        opts.pack(fill="x")
        ttk.Label(opts, text="Joint step °", font=f).pack(side="left")
        self.step = tk.DoubleVar(value=5.0)
        for v in (1, 5, 10, 30):
            if v <= self.args.max_step:
                ttk.Radiobutton(opts, text=str(v), value=float(v), variable=self.step).pack(side="left")
        ttk.Label(opts, text="   Cartesian step mm", font=f).pack(side="left")
        self.mm = tk.DoubleVar(value=10.0)
        for v in (2, 5, 10, 20, 50):
            ttk.Radiobutton(opts, text=str(v), value=float(v), variable=self.mm).pack(side="left")
        ttk.Label(opts, text="   Speed °/s", font=f).pack(side="left")
        self.speed = tk.DoubleVar(value=min(self.args.joint_speed, B.MAX_JOINT_SPEED_DEG))
        tk.Scale(opts, from_=5, to=B.MAX_JOINT_SPEED_DEG, orient="horizontal", variable=self.speed,
                 length=160, command=lambda _v: self._set_speed()).pack(side="left")

        mid = ttk.Frame(r, padding=8)
        mid.pack(fill="x")
        jf = ttk.LabelFrame(mid, text="Joints", padding=6)
        jf.pack(side="left", fill="y")
        self.jlabels = []
        for j in range(6):
            ttk.Label(jf, text=f"J{j + 1}", font=("Segoe UI", 12, "bold")).grid(row=j, column=0, padx=4)
            ttk.Button(jf, text="−", style="Jog.TButton",
                       command=lambda j=j: self.queue(self._jog_joint, j, -1)).grid(row=j, column=1)
            lbl = ttk.Label(jf, text="   —   ", font=("Consolas", 13), width=10, anchor="center")
            lbl.grid(row=j, column=2)
            self.jlabels.append(lbl)
            ttk.Button(jf, text="+", style="Jog.TButton",
                       command=lambda j=j: self.queue(self._jog_joint, j, +1)).grid(row=j, column=3)

        cf = ttk.LabelFrame(mid, text="Tool (Cartesian, tool down)", padding=6)
        cf.pack(side="left", fill="y", padx=8)
        for row, ax in enumerate("xyz"):
            ttk.Button(cf, text=f"{ax.upper()}−", style="Jog.TButton",
                       command=lambda a=ax: self.queue(self._jog_cart, a, -1)).grid(row=row, column=0, pady=2)
            ttk.Button(cf, text=f"{ax.upper()}+", style="Jog.TButton",
                       command=lambda a=ax: self.queue(self._jog_cart, a, +1)).grid(row=row, column=1, pady=2)
        self.pose = tk.StringVar(value="pose: —")
        ttk.Label(cf, textvariable=self.pose, font=("Consolas", 11), justify="left").grid(
            row=3, column=0, columnspan=2, pady=6)
        ttk.Button(cf, text="Home (pose at enable)", command=lambda: self.queue(self._home)).grid(
            row=4, column=0, columnspan=2, sticky="ew")
        ttk.Button(cf, text="Set home = here", command=lambda: self.queue(self._sethome)).grid(
            row=5, column=0, columnspan=2, sticky="ew")

        gf = ttk.LabelFrame(mid, text="Gripper", padding=6)
        gf.pack(side="left", fill="y")
        self.gstep = tk.DoubleVar(value=10.0)
        gs = ttk.Frame(gf)
        gs.grid(row=0, column=0, columnspan=2)
        ttk.Label(gs, text="step °").pack(side="left")
        for v in (2, 10, 30):
            ttk.Radiobutton(gs, text=str(v), value=float(v), variable=self.gstep).pack(side="left")
        ttk.Button(gf, text="G−", style="Jog.TButton", command=lambda: self.queue(self._jog_grip, -1)).grid(
            row=1, column=0)
        ttk.Button(gf, text="G+", style="Jog.TButton", command=lambda: self.queue(self._jog_grip, +1)).grid(
            row=1, column=1)
        self.glabel = ttk.Label(gf, text="—", font=("Consolas", 13))
        self.glabel.grid(row=2, column=0, columnspan=2, pady=4)
        ttk.Button(gf, text="Save as OPEN", command=lambda: self.queue(self._grip_set, "open")).grid(
            row=3, column=0, sticky="ew")
        ttk.Button(gf, text="Save as CLOSED", command=lambda: self.queue(self._grip_set, "closed")).grid(
            row=3, column=1, sticky="ew")
        ttk.Button(gf, text="Open", command=lambda: self.queue(self._grip_go, "open")).grid(
            row=4, column=0, sticky="ew")
        ttk.Button(gf, text="Close", command=lambda: self.queue(self._grip_go, "closed")).grid(
            row=4, column=1, sticky="ew")
        self.gsaved = tk.StringVar(value=self._gripper_saved_text())
        ttk.Label(gf, textvariable=self.gsaved, font=("Segoe UI", 9)).grid(row=5, column=0, columnspan=2)
        ttk.Label(gf, text="width (cm), linear 0 = closed .. 10 = open").grid(row=6, column=0, columnspan=2)
        self.gwidth = tk.DoubleVar(value=5.0)
        tk.Scale(gf, from_=0, to=GRIP_OPEN_CM, resolution=0.5, orient="horizontal", variable=self.gwidth,
                 length=180).grid(row=7, column=0, columnspan=2)
        ttk.Button(gf, text="Go to width", command=lambda: self.queue(self._grip_width)).grid(
            row=8, column=0, columnspan=2, sticky="ew")

        kf = ttk.LabelFrame(mid, text="Calibrate arm base (calib-base)", padding=6)
        kf.pack(side="left", fill="y", padx=8)
        ttk.Label(kf, text="Put the tool tip on a tag centre, then:").grid(row=0, column=0, columnspan=2)
        for i, tag in enumerate((0, 3, 1, 2)):
            ttk.Button(kf, text=f"Touch tag {tag}", command=lambda t=tag: self.queue(self._touch, [str(t)])).grid(
                row=1 + i // 2, column=i % 2, sticky="ew")
        xy = ttk.Frame(kf)
        xy.grid(row=3, column=0, columnspan=2, pady=4)
        self.tx, self.ty = tk.StringVar(), tk.StringVar()
        ttk.Label(xy, text="x cm").pack(side="left")
        ttk.Entry(xy, textvariable=self.tx, width=6).pack(side="left")
        ttk.Label(xy, text="y cm").pack(side="left")
        ttk.Entry(xy, textvariable=self.ty, width=6).pack(side="left")
        ttk.Button(xy, text="Touch point", command=lambda: self.queue(
            self._touch, [self.tx.get(), self.ty.get()])).pack(side="left", padx=4)
        ttk.Button(kf, text="Solve base (≥ 3 touches)", command=lambda: self.queue(self._solve)).grid(
            row=4, column=0, columnspan=2, sticky="ew")
        self.ntouch = tk.StringVar(value="touches: 0")
        ttk.Label(kf, textvariable=self.ntouch).grid(row=5, column=0, columnspan=2)

        mf = ttk.LabelFrame(r, text="Motors", padding=6)
        mf.pack(fill="x", padx=8)
        self.mtext = tk.StringVar(value="—")
        ttk.Label(mf, textvariable=self.mtext, font=("Consolas", 10), justify="left").pack(anchor="w")

        lf = ttk.LabelFrame(r, text="Log", padding=6)
        lf.pack(fill="both", expand=True, padx=8, pady=8)
        self.log = tk.Text(lf, height=10, font=("Consolas", 10), bg="#111", fg="#ddd")
        self.log.pack(fill="both", expand=True)

    # ------------------------------------------------------------------ helpers
    def _gripper_saved_text(self) -> str:
        c = B.load_calib()
        o, k = c.get("gripper_open_rad"), c.get("gripper_closed_rad")
        fmt = lambda v: "—" if v is None else f"{math.degrees(v):+.1f}°"  # noqa: E731
        return f"saved: open {fmt(o)}   closed {fmt(k)}"

    def queue(self, fn, *a) -> None:
        if self.jog is None:
            messagebox.showwarning("reBot", "Connect first.")
            return
        if self.busy:
            print("busy: wait for the current move (or STOP)")
            return
        self.work.put((fn, a))

    def _worker(self) -> None:
        while True:
            fn, a = self.work.get()
            self.busy = True
            try:
                fn(*a)
            except Exception as e:  # noqa: BLE001
                print(f"error: {type(e).__name__}: {e}")
                if self.jog is not None and self.jog.enabled:
                    try:
                        self.jog.hold(self.jog.read_motor())
                    except Exception:  # noqa: BLE001
                        pass
            finally:
                self.busy = False

    def _poller(self) -> None:
        """Read-only status at ~4 Hz, only while no motion holds the bus."""
        while True:
            time.sleep(POLL_S)
            jog = self.jog
            if jog is None or not jog.lock.acquire(blocking=False):
                continue
            try:
                states = B.read_states(self.drv, 1)
            except Exception as e:  # noqa: BLE001
                self.state = {"error": str(e)}
                continue
            finally:
                jog.lock.release()
            st = {"states": states}
            if all(s is not None for _, s in states[:6]):
                m = np.array([s.pos for _, s in states])
                q = self.drv.motor_to_urdf(m[:6])
                st.update(q=q, grip=float(m[6]) if len(m) > 6 else None, pose=self.drv.tool_pose(q))
            self.state = st

    def _tick(self) -> None:
        while not self.logq.empty():
            self.log.insert("end", self.logq.get())
            self.log.see("end")
        st = self.state
        if self.jog is None:
            self.status.set("not connected")
            self.status_lbl.config(fg="#555")
        else:
            on = self.jog.enabled
            self.status.set(("TORQUE ON" if on else "torque off (read-only)") + ("  · moving" if self.busy else ""))
            self.status_lbl.config(fg="#2e7d32" if on else "#555")
        if "q" in st:
            for j, lbl in enumerate(self.jlabels):
                lbl.config(text=f"{math.degrees(st['q'][j]):+7.1f}°")
            p = st["pose"]
            self.pose.set(f"x {p['x'] * 100:+6.1f} cm\ny {p['y'] * 100:+6.1f} cm\nz {p['z'] * 100:+6.1f} cm\n"
                          f"yaw {p['yaw']:+6.1f}°\ntilt {p['tilt_deg']:5.1f}°")
            if st.get("grip") is not None:
                self.glabel.config(text=f"{math.degrees(st['grip']):+7.1f}°")
        if "states" in st:
            lines = []
            for name, s in st["states"]:
                if s is None:
                    lines.append(f"{name:9s} NO FEEDBACK")
                    continue
                code = getattr(s, "status_code", None)
                lines.append(f"{name:9s} {math.degrees(s.pos):+8.1f}°  T {getattr(s, 't_mos', float('nan')):3.0f}C  "
                             f"status {code} {B.DM_STATUS.get(code, '') if isinstance(code, int) else ''}")
            self.mtext.set("\n".join(lines))
        elif "error" in st:
            self.mtext.set(f"read error: {st['error']}")
        if self.jog is not None:
            self.ntouch.set(f"touches: {len(self.jog.touches)}")
        self.root.after(100, self._tick)

    # ------------------------------------------------------------------ actions (UI thread)
    def connect(self) -> None:
        if self.jog is not None:
            return
        self.args.port = self.port.get().strip() or "COM5"
        drv = B.make_driver(self.args)
        print(f"opening {drv.port} @ {drv.baud} (read-only: torque stays OFF)")
        try:
            B.open_bus(drv)
        except SystemExit as e:              # open_bus explains the problem and exits in the CLI
            messagebox.showerror("reBot", str(e))
            return
        self.drv, self.jog = drv, B.Jog(drv, self.args)
        self._set_speed()
        B.print_states(drv, B.read_states(drv))

    def disconnect(self) -> None:
        if self.jog is None:
            return
        if self.jog.enabled and not messagebox.askyesno(
                "reBot", "Torque is ON. Disconnecting turns it OFF and the arm goes LIMP.\n"
                         "Support the arm. Continue?"):
            return
        jog, self.jog = self.jog, None
        jog.abort.set()
        with jog.lock:
            B.close_bus(self.drv, disable=jog.enabled)
        print("disconnected")

    def enable(self) -> None:
        if self.jog is None:
            messagebox.showwarning("reBot", "Connect first.")
            return
        if messagebox.askyesno("Enable torque",
                               "Torque ON: the arm will HOLD its current pose.\n\n"
                               "Area clear? Hand on the power switch?"):
            self.queue(lambda: self.jog.enable(confirm=False))

    def stop(self) -> None:
        if self.jog is not None:
            threading.Thread(target=self.jog.stop, daemon=True).start()   # never waits behind a move

    def disable(self) -> None:
        if self.jog is None:
            return
        if messagebox.askyesno("Disable torque", "Torque OFF: the arm goes LIMP and may drop.\n"
                                                 "Support it first. Disable now?"):
            threading.Thread(target=self.jog.disable, daemon=True).start()

    def _set_speed(self) -> None:
        if self.jog is not None:
            self.jog.speed = math.radians(min(float(self.speed.get()), B.MAX_JOINT_SPEED_DEG))

    # ------------------------------------------------------------------ actions (worker thread)
    def _jog_joint(self, j: int, sign: int) -> None:
        q = self.jog.read_q()
        q[j] += math.radians(sign * float(self.step.get()))
        self.jog.move_q(q)

    def _jog_cart(self, ax: str, sign: int) -> None:
        q0 = self.jog.read_q()
        t = self.drv.tool_pose(q0)
        t[ax] += sign * float(self.mm.get()) / 1000
        self.jog.move_q(self.drv.ik(t["x"], t["y"], t["z"], t["yaw"], q0))

    def _jog_grip(self, sign: int) -> None:
        self.jog.move_grip(float(self.jog.read_motor()[6]) + math.radians(sign * float(self.gstep.get())))

    def _grip_set(self, which: str) -> None:
        key = "gripper_open_rad" if which == "open" else "gripper_closed_rad"
        v = round(float(self.jog.read_motor()[6]), 4)
        B.save_calib(**{key: v})
        print(f"saved gripper {which} = {math.degrees(v):+.1f}° (motor angle now)")
        self.gsaved.set(self._gripper_saved_text())

    def _grip_go(self, which: str) -> None:
        if not B.gripper_known():
            print("save OPEN and CLOSED first (jog the gripper there, then Save as ...)")
            return
        cal = B.load_calib()
        self.jog.move_grip(cal["gripper_open_rad" if which == "open" else "gripper_closed_rad"])

    def _grip_width(self) -> None:
        if not B.gripper_known():
            print("save OPEN and CLOSED first (jog the gripper there, then Save as ...)")
            return
        cal = B.load_calib()
        frac = min(1.0, max(0.0, float(self.gwidth.get()) / GRIP_OPEN_CM))
        o, c = cal["gripper_open_rad"], cal["gripper_closed_rad"]
        print(f"gripper to {frac * GRIP_OPEN_CM:.1f} cm (value {frac:.2f})")
        self.jog.move_grip(c + (o - c) * frac)

    def _home(self) -> None:
        jog = self.jog
        if jog.home_q is None:
            print("no home yet (enable first)")
            return
        q0 = jog.read_q()
        d = jog.home_q - q0
        jog.move_q(q0 + d * min(1.0, jog.cap / max(float(np.max(np.abs(d))), 1e-9)))
        left = float(np.max(np.abs(jog.home_q - jog.read_q())))
        print("at home" if left < 0.03 else f"{math.degrees(left):.0f}° to go: press Home again")

    def _sethome(self) -> None:
        self.jog.home_q = self.jog.read_q()
        print("home = this pose")

    def _touch(self, parts: list[str]) -> None:
        self.jog.touch([p for p in parts if p != ""])

    def _solve(self) -> None:
        self.jog.solve()

    # ------------------------------------------------------------------ shutdown
    def on_close(self) -> None:
        if self.jog is not None:
            if self.jog.enabled:
                ans = messagebox.askyesnocancel(
                    "Quit", "Torque is ON.\n\nYes = turn torque OFF (arm goes LIMP: support it)\n"
                            "No = leave the motors holding their last position\nCancel = stay")
                if ans is None:
                    return
                self.jog.abort.set()
                with self.jog.lock:
                    B.close_bus(self.drv, disable=bool(ans))
            else:
                with self.jog.lock:
                    B.close_bus(self.drv, disable=False)
        sys.stdout = sys.__stdout__
        self.root.destroy()


def run_gui(args) -> None:
    root = tk.Tk()
    JogGui(root, args)
    secs = float(getattr(args, "seconds", 0) or 0)
    if secs > 0:                               # smoke test: open the window, then close it
        root.after(int(secs * 1000), root.destroy)
    root.mainloop()
