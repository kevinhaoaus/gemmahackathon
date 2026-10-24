"""UFactory xArm 5/6/7 and Lite 6 via xarm-python-sdk (`from xarm.wrapper import XArmAPI`).

SDK: pip install xarm-python-sdk

Units: the SDK works in MILLIMETRES (position), mm/s (speed), mm/s^2 (mvacc) and, with
is_radian=False, DEGREES (roll/pitch/yaw, joint angles, deg/s). We convert m -> mm (x1000) on the way
out and mm -> m (/1000) on the way back; every SDK call passes is_radian=False explicitly.
Orientation: roll/pitch/yaw with R = Rz(yaw) Ry(pitch) Rx(roll); roll 180, pitch 0 = tool down.

Moves are sent with wait=False and polled, so stop() (set_state(4)) interrupts them and the
asyncio task stays cancellable. After our own stop the next move re-enters state 0. A controller
error (collision, joint limit...) is NOT auto-cleared: the task fails until {"action": "reset"}.

Config (besides the common keys in cell/arm.py):
    ip: "192.168.1.xxx"
    model: xarm | lite6                 (speed cap 1.0 vs 0.5 m/s; default gripper type)
    tool_roll_deg: 180.0, tool_pitch_deg: 0.0
    yaw_offset_deg: 0.0                 SDK yaw = yaw + this
    mvacc: 1000                         mm/s^2
    tolerance_mm: 2.0                   a move must end within this of the target
    move_timeout_s: 30
    home_joints_deg: [..]               optional: home by joint angles (deg) at joint_speed_deg_s
    joint_speed_deg_s: 20
    gripper: xarm | lite6 | vacuum | none
             xarm: xArm Gripper position 0 (closed) .. 850 (open), value * 850
             lite6: Lite 6 gripper, binary (value >= 0.5 open, else close)
             vacuum: suction, value < 0.5 = suction ON (holding)
    gripper_speed: 2000                 xArm Gripper r/min
    gripper_settle_s: 0.8               lite6 / vacuum: wait after switching
"""
from __future__ import annotations

import asyncio
import math
import threading
import time

from ..arm import Arm, wrap_deg
from ..arm import log as arm_log
from . import missing_sdk

SPEED_CAP = {"xarm": 1.0, "lite6": 0.5}
GRIPPER_OPEN_POS = 850.0


class XArm(Arm):
    name = "xarm"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        c = self.cfg
        self.ip = str(c.get("ip", ""))
        self.model = str(c.get("model", "xarm")).lower().replace(" ", "").replace("_", "")
        if self.model not in SPEED_CAP:
            raise ValueError(f"xarm model must be xarm | lite6, got {c.get('model')!r}")
        self.roll = float(c.get("tool_roll_deg", 180.0))
        self.pitch = float(c.get("tool_pitch_deg", 0.0))
        self.yaw_offset = float(c.get("yaw_offset_deg", 0.0))
        self.mvacc = float(c.get("mvacc", 1000.0))
        self.tol_mm = float(c.get("tolerance_mm", 2.0))
        self.timeout_s = float(c.get("move_timeout_s", 30.0))
        self.home_joints = c.get("home_joints_deg")
        self.joint_speed = float(c.get("joint_speed_deg_s", 20.0))
        self.gripper_kind = str(c.get("gripper", "lite6" if self.model == "lite6" else "xarm"))
        if self.gripper_kind not in ("xarm", "lite6", "vacuum", "none"):
            raise ValueError("xarm gripper must be xarm | lite6 | vacuum | none")
        self.gripper_speed = float(c.get("gripper_speed", 2000))
        self.gripper_settle_s = float(c.get("gripper_settle_s", 0.8))
        self.api = None
        self._lock = threading.Lock()          # one SDK call at a time; every call is short
        self._stopped = False

    # ------------------------------------------------------------------ helpers
    def _call(self, fn, *a, **kw):
        with self._lock:
            return fn(*a, **kw)

    async def _acall(self, fn, *a, **kw):
        return await asyncio.to_thread(self._call, fn, *a, **kw)

    def _need(self):
        if self.api is None:
            raise RuntimeError(f"xArm at {self.ip} is not connected")
        return self.api

    @staticmethod
    def _ok(code, what: str) -> None:
        if isinstance(code, (tuple, list)):
            code = code[0]
        if code not in (0, None):
            raise RuntimeError(f"xArm {what} failed with code {code}")

    # ------------------------------------------------------------------ lifecycle
    def _connect_sync(self) -> None:
        if not self.ip:
            raise ValueError("xarm driver needs ip")
        try:
            from xarm.wrapper import XArmAPI
        except ImportError as e:
            raise missing_sdk("xarm-python-sdk", "pip install xarm-python-sdk", e) from None
        api = XArmAPI(self.ip, is_radian=False)
        if not getattr(api, "connected", True):
            raise ConnectionError(f"xArm at {self.ip} did not connect")
        self.api = api
        api.clean_warn()
        api.clean_error()
        self._ok(api.motion_enable(enable=True), "motion_enable")
        self._ok(api.set_mode(0), "set_mode(0)")          # position control
        self._ok(api.set_state(state=0), "set_state(0)")
        if self.gripper_kind == "xarm":
            self._ok(api.set_gripper_mode(0), "set_gripper_mode")
            self._ok(api.set_gripper_enable(True), "set_gripper_enable")
            self._ok(api.set_gripper_speed(self.gripper_speed), "set_gripper_speed")
        code, pos = api.get_position(is_radian=False)
        self._ok(code, "get_position")
        arm_log.info("xArm (%s) connected at %s, tool at %s mm", self.model, self.ip, [round(v, 1) for v in pos[:3]])

    async def connect(self) -> None:
        try:
            await asyncio.to_thread(self._connect_sync)
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        api, self.api = self.api, None
        if api is not None:
            try:
                await asyncio.to_thread(api.disconnect)
            except Exception as e:  # noqa: BLE001
                arm_log.warning("xArm disconnect: %s", e)

    def _ready_sync(self) -> None:
        api = self._need()
        if api.error_code:
            raise RuntimeError(f"xArm controller error {api.error_code} (warn {api.warn_code}); "
                               f"fix the cause, then send {{'action': 'reset'}}")
        if self._stopped or api.state not in (0, 1, 2):
            # our own stop left it in state 4 (stopped): re-arm position mode
            self._ok(api.motion_enable(enable=True), "motion_enable")
            self._ok(api.set_mode(0), "set_mode(0)")
            self._ok(api.set_state(state=0), "set_state(0)")
            self._stopped = False

    def _reset_sync(self) -> None:
        api = self._need()
        api.clean_warn()
        api.clean_error()
        self._stopped = True
        self._ready_sync()

    async def reset(self) -> None:
        await self._acall(self._reset_sync)

    # ------------------------------------------------------------------ motion
    async def _wait_motion(self, reached, what: str) -> None:
        """Poll until the controller stops moving; then `reached()` must be true."""
        api = self._need()
        t0 = time.monotonic()
        started = False
        while True:
            await asyncio.sleep(0.05)
            moving = await self._acall(api.get_is_moving)
            if api.error_code:
                raise RuntimeError(f"xArm error {api.error_code} during {what}")
            started = started or moving
            ok = await asyncio.to_thread(reached)
            elapsed = time.monotonic() - t0
            if not moving and (ok or (started and elapsed > 0.2) or elapsed > 1.5):
                if not ok:
                    raise RuntimeError(f"xArm stopped before reaching the {what} target")
                return
            if elapsed > self.timeout_s:
                await self.stop()
                raise TimeoutError(f"xArm {what}: not reached after {self.timeout_s:g}s")

    def _pos_mm(self) -> list:
        code, pos = self._call(self._need().get_position, is_radian=False)
        self._ok(code, "get_position")
        return list(pos)

    async def move_to(self, x: float, y: float, z: float, yaw_deg: float, speed: float) -> None:
        api = self._need()
        await self._acall(self._ready_sync)
        tgt = (x * 1000.0, y * 1000.0, z * 1000.0)                       # m -> mm
        v_mm = min(float(speed), SPEED_CAP[self.model]) * 1000.0          # m/s -> mm/s
        sdk_yaw = wrap_deg(yaw_deg + self.yaw_offset)
        code = await self._acall(api.set_position, x=tgt[0], y=tgt[1], z=tgt[2], roll=self.roll,
                                 pitch=self.pitch, yaw=sdk_yaw, speed=v_mm, mvacc=self.mvacc,
                                 is_radian=False, wait=False)
        self._ok(code, "set_position")

        def reached() -> bool:
            p = self._pos_mm()
            return math.dist(p[:3], tgt) <= self.tol_mm and abs(wrap_deg(p[5] - sdk_yaw)) <= 1.0

        await self._wait_motion(reached, "move_to")

    def home_target(self):
        if self.home_joints is not None and not isinstance(self.cfg.get("home"), dict):
            return None
        return super().home_target()

    async def home(self, speed: float) -> None:
        api = self._need()
        await self._acall(self._ready_sync)
        joints = [float(v) for v in self.home_joints]
        code = await self._acall(api.set_servo_angle, angle=joints, speed=self.joint_speed,
                                 is_radian=False, wait=False)
        self._ok(code, "set_servo_angle")

        def reached() -> bool:
            code, ang = self._call(api.get_servo_angle, is_radian=False)
            self._ok(code, "get_servo_angle")
            return all(abs(a - b) <= 0.5 for a, b in zip(ang, joints))

        await self._wait_motion(reached, "home")

    # ------------------------------------------------------------------ gripper
    async def gripper(self, value: float) -> None:
        api = self._need()
        v = float(value)
        if self.gripper_kind == "none":
            self.grip = v
            return
        if self.gripper_kind == "xarm":
            code = await self._acall(api.set_gripper_position, v * GRIPPER_OPEN_POS, wait=True,
                                     speed=self.gripper_speed, auto_enable=True)
            self._ok(code, "set_gripper_position")
        elif self.gripper_kind == "lite6":
            fn = api.open_lite6_gripper if v >= 0.5 else api.close_lite6_gripper
            self._ok(await self._acall(fn), "lite6 gripper")
            await asyncio.sleep(self.gripper_settle_s)
            v = 1.0 if v >= 0.5 else 0.0
        else:   # vacuum: closed (holding) = suction on
            on = v < 0.5
            self._ok(await self._acall(api.set_vacuum_gripper, on, wait=False), "set_vacuum_gripper")
            await asyncio.sleep(self.gripper_settle_s)
            v = 0.0 if on else 1.0
        self.grip = v

    # ------------------------------------------------------------------ stop / state
    async def stop(self) -> None:
        if self.api is None:
            return
        self._stopped = True
        try:
            await self._acall(self.api.set_state, state=4)      # 4 = stop motion, clear queue
        except Exception as e:  # noqa: BLE001
            arm_log.error("xArm stop failed: %s", e)

    async def pose(self) -> dict:
        p = await asyncio.to_thread(self._pos_mm)
        return {"x": p[0] / 1000.0, "y": p[1] / 1000.0, "z": p[2] / 1000.0,    # mm -> m
                "yaw": wrap_deg(p[5] - self.yaw_offset), "roll": p[3], "pitch": p[4],
                "gripper": self.grip}

    def info(self) -> dict:
        return {"driver": self.name, "model": self.model, "ip": self.ip, "gripper": self.gripper_kind,
                "speed_cap": SPEED_CAP[self.model]}
