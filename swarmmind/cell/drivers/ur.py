"""Universal Robots (CB3 / e-Series) via ur_rtde: rtde_control, rtde_receive, rtde_io.

SDK: pip install ur_rtde   (the robot must be in Remote Control mode on e-Series)

Units: ur_rtde poses are [x, y, z, rx, ry, rz] with x/y/z in METRES and (rx, ry, rz) an axis-angle
ROTATION VECTOR in RADIANS; moveL speed m/s, acceleration m/s^2; moveJ joints in RADIANS, rad/s,
rad/s^2. We build the tool orientation R = Rz(yaw) Ry(pitch) Rx(roll) from degrees and convert it
to a rotation vector through a quaternion (robust at the 180 deg rotation that "tool down" always
is). Poses are in the robot's Base frame as RTDE reports it (the pendant's "Base" view, NOT
"Base link", which is rotated 180 deg about z): calibrate table_to_arm against that frame.

Moves use moveL(..., asynchronous=True) and are polled, so stop() (stopL) interrupts them and the
asyncio task stays cancellable. A protective / emergency stop on the robot fails the task.

Config (besides the common keys in cell/arm.py):
    ip: "192.168.1.xxx"
    tool_roll_deg: 180.0, tool_pitch_deg: 0.0      tool z pointing down
    yaw_offset_deg: 0.0
    acceleration: 0.5                   m/s^2
    stop_decel: 2.0                     m/s^2 (stopL) / rad/s^2 (stopJ)
    tolerance_m: 0.002
    move_timeout_s: 30
    tcp_offset_m: [0, 0, 0.15]          optional: setTcp translation (metres; no rotation)
    home_joints_deg: [..6]              optional: home by joint angles in DEGREES (converted to rad)
    joint_speed_deg_s: 30, joint_accel_deg_s2: 60
    gripper: {type: digital_out | tool_digital_out | none, pin: 0, close_level: true, settle_s: 0.5}
             value < 0.5 -> output = close_level (close), else not close_level (open).
             "none" (default) = configurable no-op that only records the value.
"""
from __future__ import annotations

import asyncio
import math
import threading
import time

from ..arm import Arm, wrap_deg
from ..arm import log as arm_log
from . import missing_sdk

SPEED_CAP = 1.0     # m/s; the robot's own safety config may be lower


# ---------------------------------------------------------------------- rotation math
def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    """Degrees -> R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = (math.radians(v) for v in (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def matrix_to_rotvec(R: list[list[float]]) -> list[float]:
    """Rotation matrix -> axis-angle vector (radians), via a quaternion (Shepperd)."""
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2
        w, x, y, z = (R[2][1] - R[1][2]) / s, 0.25 * s, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2
        w, x, y, z = (R[0][2] - R[2][0]) / s, (R[0][1] + R[1][0]) / s, 0.25 * s, (R[1][2] + R[2][1]) / s
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2
        w, x, y, z = (R[1][0] - R[0][1]) / s, (R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, 0.25 * s
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(n, w)
    return [x / n * angle, y / n * angle, z / n * angle]


def rotvec_to_matrix(v: list[float]) -> list[list[float]]:
    """Axis-angle (radians) -> rotation matrix (Rodrigues)."""
    th = math.sqrt(sum(c * c for c in v))
    if th < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    kx, ky, kz = (c / th for c in v)
    c, s, C = math.cos(th), math.sin(th), 1 - math.cos(th)
    return [[c + kx * kx * C, kx * ky * C - kz * s, kx * kz * C + ky * s],
            [ky * kx * C + kz * s, c + ky * ky * C, ky * kz * C - kx * s],
            [kz * kx * C - ky * s, kz * ky * C + kx * s, c + kz * kz * C]]


def yaw_of(R: list[list[float]]) -> float:
    """Yaw (deg) of R = Rz(yaw) Ry(pitch) Rx(roll): the tool x axis projected on the base xy plane."""
    return math.degrees(math.atan2(R[1][0], R[0][0]))


# ---------------------------------------------------------------------- driver
class URArm(Arm):
    name = "ur"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        c = self.cfg
        self.ip = str(c.get("ip", ""))
        self.roll = float(c.get("tool_roll_deg", 180.0))
        self.pitch = float(c.get("tool_pitch_deg", 0.0))
        self.yaw_offset = float(c.get("yaw_offset_deg", 0.0))
        self.accel = float(c.get("acceleration", 0.5))
        self.stop_decel = float(c.get("stop_decel", 2.0))
        self.tol = float(c.get("tolerance_m", 0.002))
        self.timeout_s = float(c.get("move_timeout_s", 30.0))
        self.tcp_offset = c.get("tcp_offset_m")
        self.home_joints = c.get("home_joints_deg")
        self.joint_speed = math.radians(float(c.get("joint_speed_deg_s", 30.0)))   # deg/s -> rad/s
        self.joint_accel = math.radians(float(c.get("joint_accel_deg_s2", 60.0)))
        g = c.get("gripper") or {}
        if isinstance(g, str):
            g = {"type": g}
        self.g_type = str(g.get("type", "none"))
        if self.g_type not in ("digital_out", "tool_digital_out", "none"):
            raise ValueError("ur gripper.type must be digital_out | tool_digital_out | none")
        self.g_pin = int(g.get("pin", 0))
        self.g_close_level = bool(g.get("close_level", True))
        self.g_settle = float(g.get("settle_s", 0.5))
        self.rc = self.rr = self.rio = None
        self._lock = threading.Lock()
        self._last = "L"                        # last move kind, for stopL vs stopJ

    def tool_pose(self, x: float, y: float, z: float, yaw_deg: float) -> list[float]:
        """Our (m, m, m, deg) -> UR [x, y, z, rx, ry, rz] (m, rad rotation vector)."""
        rv = matrix_to_rotvec(rpy_to_matrix(self.roll, self.pitch, yaw_deg + self.yaw_offset))
        return [float(x), float(y), float(z), *rv]

    def from_tool_pose(self, p: list[float]) -> dict:
        R = rotvec_to_matrix(list(p[3:6]))
        return {"x": p[0], "y": p[1], "z": p[2], "yaw": wrap_deg(yaw_of(R) - self.yaw_offset)}

    def _call(self, fn, *a):
        with self._lock:
            return fn(*a)

    async def _acall(self, fn, *a):
        return await asyncio.to_thread(self._call, fn, *a)

    def _need(self):
        if self.rc is None or self.rr is None:
            raise RuntimeError(f"UR at {self.ip} is not connected")
        return self.rc, self.rr

    # ------------------------------------------------------------------ lifecycle
    def _connect_sync(self) -> None:
        if not self.ip:
            raise ValueError("ur driver needs ip")
        try:
            import rtde_control
            import rtde_receive
        except ImportError as e:
            raise missing_sdk("ur_rtde", "pip install ur_rtde", e) from None
        self.rr = rtde_receive.RTDEReceiveInterface(self.ip)
        self.rc = rtde_control.RTDEControlInterface(self.ip)
        if self.g_type != "none":
            import rtde_io
            self.rio = rtde_io.RTDEIOInterface(self.ip)
        if self.tcp_offset:
            t = [float(v) for v in self.tcp_offset][:3]
            self.rc.setTcp([*t, 0.0, 0.0, 0.0])
        p = self.rr.getActualTCPPose()
        arm_log.info("UR connected at %s, TCP at (%.3f, %.3f, %.3f) m", self.ip, p[0], p[1], p[2])

    async def connect(self) -> None:
        try:
            await asyncio.to_thread(self._connect_sync)
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        for attr in ("rc", "rr", "rio"):
            obj = getattr(self, attr)
            setattr(self, attr, None)
            if obj is None:
                continue
            try:
                if attr == "rc":
                    await asyncio.to_thread(obj.stopScript)
                await asyncio.to_thread(obj.disconnect)
            except Exception as e:  # noqa: BLE001
                arm_log.warning("UR %s disconnect: %s", attr, e)

    def _check_safety(self) -> None:
        _, rr = self._need()
        if rr.isEmergencyStopped():
            raise RuntimeError("UR is emergency-stopped (release the e-stop button and power on)")
        if rr.isProtectiveStopped():
            raise RuntimeError("UR is in a protective stop (unlock it on the pendant)")

    async def _wait_async(self, reached, what: str) -> None:
        rc, _ = self._need()
        t0 = time.monotonic()
        while True:
            await asyncio.sleep(0.05)
            await self._acall(self._check_safety)
            running = (await self._acall(rc.getAsyncOperationProgress)) >= 0
            if not running:
                if not await asyncio.to_thread(reached):
                    if time.monotonic() - t0 < 0.5:          # async op may not have started yet
                        continue
                    raise RuntimeError(f"UR stopped before reaching the {what} target")
                return
            if time.monotonic() - t0 > self.timeout_s:
                await self.stop()
                raise TimeoutError(f"UR {what}: not reached after {self.timeout_s:g}s")

    # ------------------------------------------------------------------ motion
    async def move_to(self, x: float, y: float, z: float, yaw_deg: float, speed: float) -> None:
        rc, rr = self._need()
        await self._acall(self._check_safety)
        pose = self.tool_pose(x, y, z, yaw_deg)
        v = min(float(speed), SPEED_CAP)                                    # m/s
        self._last = "L"
        ok = await self._acall(rc.moveL, pose, v, self.accel, True)        # asynchronous=True
        if ok is False:
            raise RuntimeError("UR moveL was rejected (target unreachable or robot not ready)")

        def reached() -> bool:
            p = self._call(rr.getActualTCPPose)
            yaw = self.from_tool_pose(p)["yaw"]
            return math.dist(p[:3], pose[:3]) <= self.tol and abs(wrap_deg(yaw - yaw_deg)) <= 1.0

        await self._wait_async(reached, "move_to")

    def home_target(self):
        if self.home_joints is not None and not isinstance(self.cfg.get("home"), dict):
            return None
        return super().home_target()

    async def home(self, speed: float) -> None:
        rc, rr = self._need()
        await self._acall(self._check_safety)
        q = [math.radians(float(v)) for v in self.home_joints]               # deg -> rad
        self._last = "J"
        ok = await self._acall(rc.moveJ, q, self.joint_speed, self.joint_accel, True)
        if ok is False:
            raise RuntimeError("UR moveJ (home) was rejected")

        def reached() -> bool:
            cur = self._call(rr.getActualQ)
            return all(abs(a - b) <= math.radians(0.5) for a, b in zip(cur, q))

        await self._wait_async(reached, "home")

    async def gripper(self, value: float) -> None:
        v = float(value)
        close = v < 0.5
        if self.g_type != "none":
            if self.rio is None:
                raise RuntimeError("UR IO interface not connected")
            level = self.g_close_level if close else not self.g_close_level
            fn = self.rio.setStandardDigitalOut if self.g_type == "digital_out" else self.rio.setToolDigitalOut
            ok = await self._acall(fn, self.g_pin, level)
            if ok is False:
                raise RuntimeError(f"UR {self.g_type} {self.g_pin} write failed")
            await asyncio.sleep(self.g_settle)
            v = 0.0 if close else 1.0
        self.grip = v

    async def stop(self) -> None:
        if self.rc is None:
            return
        try:
            if self._last == "J":
                await self._acall(self.rc.stopJ, self.stop_decel)
            else:
                await self._acall(self.rc.stopL, self.stop_decel)
        except Exception as e:  # noqa: BLE001
            arm_log.error("UR stop failed: %s", e)

    async def reset(self) -> None:
        rc, _ = self._need()
        if not await self._acall(rc.isProgramRunning):
            await self._acall(rc.reuploadScript)

    async def pose(self) -> dict:
        _, rr = self._need()
        p = await self._acall(rr.getActualTCPPose)
        return {**self.from_tool_pose(p), "gripper": self.grip}

    def info(self) -> dict:
        return {"driver": self.name, "ip": self.ip, "gripper": self.g_type}
