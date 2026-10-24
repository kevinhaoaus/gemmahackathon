"""SO-101 as a cell arm: an adapter onto capabilities/so101.py's ArmController, with Cartesian
control from swarmmind.cell.kinematics_so101 (URDF forward kinematics + top-down IK).

Cartesian mode (supports_cartesian = True) turns on at connect() when
  - a calibration JSON is configured (`calibration: path`, the same file capabilities/so101.py
    reads; per arm joint {home, min, max, sign}, or a LeRobot calibration file), or
  - the arm is a mock and `kinematics:` is set (then home = 2048, sign = +1, bounds from the
    controller), so the whole pipeline runs without hardware.
Set `kinematics: false` to force the old named-pose-only behaviour. Without Cartesian mode,
move_to / pick / place are refused before anything moves; home, gripper (1 open .. 0 closed),
stop and named poses ({"action": "named_pose", "name": "ready"}) always work.

Extra config keys (this arm's config, or nested under `so101: {...}`):
    kinematics: {urdf: path, table_z: 0.0, limits_deg: {joint: [lo, hi]}}   (or true / false)
    yaw_offset_deg: 0.0          cell yaw = kinematic yaw + this (kinematic yaw 0 = tool pointing
                                 down with shoulder_pan = wrist_roll = 0)
    max_joint_speed_dps: 60      joint-space speed cap (deg/s) on top of Safety's m/s cap
    waypoint_m: 0.005            joint-space interpolation: at most this much TCP travel ...
    waypoint_deg: 3.0            ... and this much joint travel per waypoint
    final_tolerance_ticks: 3     after the last waypoint, wait for this tick error ...
    final_settle_s: 1.0          ... for at most this long
The arm frame is kinematics_so101's: origin on the pan axis at the base's underside, +x the
reach direction at pan = 0, z up. Without a `workspace`, SO101_WORKSPACE (the measured top-down
reachable box) is used.

If this node also runs the "arm.so101" capability, its controller (and serial port) is reused.
Otherwise this adapter opens its own controller from the same config keys as
capability_config["arm.so101"] (port, mock, calibration, ...; see capabilities/so101.py).
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ...logutil import get_logger
from .. import kinematics_so101 as kin_mod
from ..arm import Arm, wrap_deg

log = get_logger("cell.so101")

# Top-down reachable box, arm frame. Measured with kinematics_so101.top_down_workspace (URDF
# limits, tool pointing down, relative yaw -90/0/+90 deg): every (x, y) here is reachable at
# every z in [0, 0.06] (sampled box x 0.09-0.18, y -0.15..0.15); above z ~0.085 nothing is
# reachable top-down (wrist_flex runs out). x starts at 0.10 to keep the gripper off the base
# body, which the URDF check does not model. tests/test_so101_ik.py re-checks it.
SO101_WORKSPACE = {"x": [0.10, 0.18], "y": [-0.14, 0.14], "z": [0.005, 0.06]}


async def _quiet(_p: dict) -> None:
    return None


class SO101Arm(Arm):
    name = "so101"
    supports_cartesian = False

    def __init__(self, cfg: dict):
        cfg = dict(cfg or {})
        if not cfg.get("workspace"):
            cfg["workspace"] = SO101_WORKSPACE
        super().__init__(cfg)
        self.ctl = None
        self._own = False
        self.poses: list[str] = []
        self.kin: Optional[kin_mod.SO101Kinematics] = None
        self.cal: Optional[dict[str, kin_mod.JointCal]] = None
        self.cart_reason = "not connected"
        self._gen = 0
        c = self._conf()
        self.yaw_offset = float(c.get("yaw_offset_deg", self.cfg.get("yaw_offset_deg", 0.0)))
        self.max_joint_speed = math.radians(float(c.get("max_joint_speed_dps", 60.0)))
        self.waypoint_m = float(c.get("waypoint_m", 0.005))
        self.waypoint_rad = math.radians(float(c.get("waypoint_deg", 3.0)))
        self.final_tol = int(c.get("final_tolerance_ticks", 3))
        self.final_settle_s = float(c.get("final_settle_s", 1.0))
        for k, v in (("max_joint_speed_dps", self.max_joint_speed), ("waypoint_m", self.waypoint_m),
                     ("waypoint_deg", self.waypoint_rad)):
            if not v > 0:
                raise ValueError(f"so101 {k} must be > 0")

    def _conf(self) -> dict:
        nested = self.cfg.get("so101")
        return dict(nested) if isinstance(nested, dict) else dict(self.cfg)

    async def connect(self) -> None:
        from ...capabilities import so101 as so101_cap
        ctl = so101_cap.get_controller()
        if ctl is not None:
            self.ctl, self._own = ctl, False
        else:
            self.ctl = await so101_cap.build_controller(self._conf(), publish=None)
            self._own = True
        self.poses = sorted(self.ctl.poses)
        self._setup_kinematics()

    def _setup_kinematics(self) -> None:
        conf = self._conf()
        kcfg = conf.get("kinematics", self.cfg.get("kinematics"))
        calib = conf.get("calibration") or (self.ctl.conf.get("calibration") if self.ctl is not None else None)
        mock = getattr(self.ctl.drv, "name", "") == "mock"
        self.kin, self.cal, self.supports_cartesian = None, None, False
        if kcfg is False:
            self.cart_reason = "kinematics disabled in config"
            return
        if calib:
            with open(Path(calib), "r", encoding="utf-8") as f:
                raw = json.load(f) or {}
            cal = kin_mod.parse_calibration(raw, bounds=self.ctl.bounds, require_home=True)
            jraw = raw.get("joints") if isinstance(raw.get("joints"), dict) else raw
            unsigned = [j for j in kin_mod.ARM_JOINTS if "sign" not in (jraw.get(j) or {})
                        and "drive_mode" not in (jraw.get(j) or {})]
            if unsigned and not mock:
                log.warning("so101: calibration gives no sign for %s; assuming +1 (ticks increase with "
                            "the URDF angle). Verify on the arm before Cartesian moves.", ", ".join(unsigned))
        elif mock and kcfg:
            cal = kin_mod.parse_calibration({}, bounds=self.ctl.bounds, require_home=False)
        else:
            self.cart_reason = ("no calibration file (needed for joint zero offsets and signs)"
                                if not mock else "mock arm without `kinematics:` config")
            return
        kin = kin_mod.SO101Kinematics.from_config(kcfg if isinstance(kcfg, dict) else None)
        kin.clip_limits(cal)
        self.kin, self.cal = kin, cal
        self.supports_cartesian = True
        self.cart_reason = ""

    async def close(self) -> None:
        if self.ctl is not None and self._own:
            await self.ctl.stop_loops()
            self.ctl.close()
        self.ctl = None

    def _need(self):
        if self.ctl is None:
            raise RuntimeError("so101 adapter is not connected")
        return self.ctl

    async def _do(self, step: dict) -> dict:
        from ...capabilities.so101 import MotionAborted as SoAborted
        try:
            return await self._need().handle(step, _quiet)
        except SoAborted as e:
            raise RuntimeError(f"so101 motion aborted: {e}") from None

    def home_target(self) -> Optional[dict]:
        if self.supports_cartesian and isinstance(self.cfg.get("home"), dict):
            return super().home_target()
        return None

    async def home(self, speed: float) -> None:
        await self._do({"action": "home"})

    # ------------------------------------------------------------------ Cartesian
    def current_joints(self) -> np.ndarray:
        return kin_mod.ticks_to_joints(self.cal, self._need().present)

    def plan(self, x: float, y: float, z: float, yaw_deg: float, speed: float,
             q0: Optional[np.ndarray] = None) -> tuple[list[np.ndarray], float]:
        """Joint-space waypoints (excluding the start) from q0 to the IK solution of the target,
        and the time between waypoints (s). Raises kinematics_so101.Unreachable or RuntimeError
        if the path would take a link under the table."""
        if not self.supports_cartesian or self.kin is None:
            raise RuntimeError(f"the so101 driver has no inverse kinematics configured ({self.cart_reason}), "
                               f"so Cartesian move_to/pick/place are not supported; use home, gripper or named_pose")
        kin = self.kin
        q0 = self.current_joints() if q0 is None else np.asarray(q0, dtype=float)
        sol = kin.ik(x, y, z, wrap_deg(yaw_deg - self.yaw_offset), seed=q0)
        q1 = sol.q
        p0, p1 = kin.fk(q0)[:3, 3], kin.fk(q1)[:3, 3]
        dist = float(np.linalg.norm(p1 - p0))
        dq = float(np.max(np.abs(q1 - q0)))
        n = max(1, math.ceil(max(dist / self.waypoint_m, dq / self.waypoint_rad)))
        path = [q0 + (q1 - q0) * (k / n) for k in range(1, n + 1)]
        bad = self._path_problem(q0, path)
        if bad:
            path = self._cartesian_path(q0, p0, x, y, z, yaw_deg, n)
            if path is None:
                raise RuntimeError(f"so101: no safe path to ({x:.3f}, {y:.3f}, {z:.3f}): the joint-space "
                                   f"path would put the {bad} and a straight Cartesian path is not "
                                   f"reachable top-down either")
            q1 = path[-1]
            dq = float(np.max(np.abs(q1 - q0)))
        duration = max(dist / max(speed, 1e-6), dq / self.max_joint_speed)
        return path, duration / len(path)

    def _path_problem(self, q0: np.ndarray, path: list[np.ndarray]) -> Optional[str]:
        """Check the path (and the midpoints between waypoints) stays above the table."""
        prev = q0
        for q in path:
            for qq in (0.5 * (prev + q), q):
                low = self.kin.check_clear(qq)
                if low:
                    return f"{low} below the table (table_z={self.kin.table_z})"
            prev = q
        return None

    def _cartesian_path(self, q0, p0, x, y, z, yaw_deg, n) -> Optional[list[np.ndarray]]:
        """Fallback: straight-line TCP path, IK at every waypoint (seeded with the previous one)."""
        y0 = self.kin.pose(q0)["yaw"]
        dyaw = wrap_deg(wrap_deg(yaw_deg - self.yaw_offset) - y0)
        out, q = [], q0
        for k in range(1, n + 1):
            f = k / n
            p = p0 + (np.array([x, y, z]) - p0) * f
            try:
                q = self.kin.ik(float(p[0]), float(p[1]), float(p[2]), wrap_deg(y0 + dyaw * f), seed=q).q
            except kin_mod.Unreachable:
                return None
            out.append(q)
        return None if self._path_problem(q0, out) else out

    async def move_to(self, x: float, y: float, z: float, yaw_deg: float, speed: float) -> None:
        path, dt = self.plan(x, y, z, yaw_deg, speed)
        ctl = self._need()
        gen = self._gen
        t0 = time.monotonic()
        goal: dict[str, int] = {}
        for k, q in enumerate(path, start=1):
            if self._gen != gen:
                raise RuntimeError("so101 motion aborted: stopped")
            goal = kin_mod.joints_to_ticks(self.cal, q)
            await self._do({"action": "move_joints", "joints": goal, "units": "ticks", "speed": 1.0})
            wait = t0 + k * dt - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
        goal = {j: ctl.clamp_joint(j, v) for j, v in goal.items()}
        deadline = time.monotonic() + self.final_settle_s
        while time.monotonic() < deadline:
            if self._gen != gen:
                raise RuntimeError("so101 motion aborted: stopped")
            if max(abs(ctl.present.get(j, v) - v) for j, v in goal.items()) <= self.final_tol:
                break
            await asyncio.sleep(1.0 / ctl.rate_hz)

    async def named_pose(self, name: str) -> dict:
        return await self._do({"action": "pose", "name": name})

    async def gripper(self, value: float) -> None:
        await self._do({"action": "gripper", "value": float(value)})
        self.grip = float(value)

    async def stop(self) -> None:
        self._gen += 1
        if self.ctl is not None:
            self.ctl.hold("cell arm stop", estop=False)

    async def pose(self) -> dict:
        ctl = self._need()
        st = ctl.state()
        self.grip = st.get("gripper")
        out = {"x": None, "y": None, "z": None, "yaw": None, "joints": st.get("joints"),
               "gripper": self.grip, "so101_mode": st.get("mode")}
        if self.supports_cartesian and self.kin is not None and st.get("joints"):
            p = self.kin.pose(kin_mod.ticks_to_joints(self.cal, st["joints"]))
            out.update(x=p["x"], y=p["y"], z=p["z"], yaw=wrap_deg(p["yaw"] + self.yaw_offset),
                       tilt_deg=p["tilt"])
        return out

    def info(self) -> dict:
        out = {"driver": self.name, "cartesian": self.supports_cartesian, "named_poses": self.poses,
               "shared_controller": self.ctl is not None and not self._own}
        if self.supports_cartesian and self.kin is not None:
            out["kinematics"] = kin_mod.URDF_SOURCE
            out["joint_limits_deg"] = {j: [round(math.degrees(a), 1), round(math.degrees(b), 1)]
                                       for j, (a, b) in self.kin.limits.items()}
        else:
            out["cartesian_off"] = self.cart_reason
        return out
