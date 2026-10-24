"""LeRobot SO-101 arm: capability "arm.so101" (tasks) + topic teleop ("teleop.so101").

Hardware: 6 Feetech STS3215 servos on one USB serial bus, IDs 1-6. Positions are raw ticks
0..4095 (4096 ticks / turn). Everything here works in raw ticks so it needs no LeRobot
calibration to run; an optional calibration JSON narrows the bounds.

Only active on a node whose config lists "arm.so101" in `capabilities`. Settings live in
`capability_config["arm.so101"]`:

    port: "/dev/ttyACM0" | "COM5" | ""   ("" = mock)
    mock: true | false                   (unset: try the real bus, fall back to mock with a warning)
    fallback_to_mock: true               (false: a failed real bus is a setup error instead)
    calibration: "path/to/so101.json"    (optional, see below)
    baudrate: 1000000
    max_step_ticks: 15                   (max goal change per control tick per joint)
    rate_hz: 30                          (control loop rate; 15 ticks * 30 Hz = ~40 deg/s)
    telemetry_hz: 10
    tolerance_ticks: 20                  (a joint "arrived" within this many ticks)
    settle_s: 2.0                        (wait this long for the servos to catch up; a gripper
                                          closing on a part never does, which is fine)
    move_timeout_s: 30
    uncalibrated_span: 400               (real arm, no calibration: bounds = start pos +- span)
    torque_off_on_estop: false
    estop_latch: false                   (true: after an e-stop only {"action": "reset"} clears it)
    teleop_preempts_tasks: true          (an operator pressing the deadman aborts a running task)
    mock_speed_ticks_s: 1500             (simulated servo speed)
    teleop: {gain_pan: 1.0, gain_lift: 1.0, gain_elbow: -1.0, gain_roll: 1.0,
             beta_split: 0.5, deadman_ms: 300, release_s: 2.0}
             (gains are joint-degrees per phone-degree; negative inverts)

Calibration JSON (ticks). Either our format:
    {"joints": {"shoulder_pan": {"min": 900, "max": 3200, "home": 2048}, ...},
     "poses": {"ready": {"shoulder_lift": 1700, "elbow_flex": 2300}, "open": {"gripper": 1.0}}}
or a LeRobot calibration file ({"shoulder_pan": {"id": 1, "range_min": .., "range_max": ..}, ...});
home defaults to the middle of the range. Pose values: int ticks, or floats in -1..1 (normalized).

Task args (`_task_id` makes them idempotent):
    {"action": "home"}
    {"action": "move_joints", "joints": {"elbow_flex": 2300, "wrist_flex": -0.2}, "speed": 0.5}
    {"action": "gripper", "value": 0.0..1.0}          (0 = closed = min, 1 = open = max)
    {"action": "pose", "name": "ready"}
    {"action": "sequence", "steps": [{...}, {...}]}
    {"action": "stop"}                                (hold now; aborts the running motion)
    {"action": "reset"}                               (clear a latched e-stop)
Joint values: int ticks, or floats with |v| <= 1 as normalized (-1 = min, +1 = max);
force with "units": "ticks" | "norm". Everything is clamped to the bounds.

Teleop sample data: {alpha, beta, gamma (deg, DeviceOrientation), grip 0..1, enabled, deadman,
source (client id)}. Pressing the deadman captures the phone orientation and the arm pose as zero
("clutch"): motion is relative from there, so pressing never makes the arm jump.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..logutil import get_logger
from ..tasks import register_capability
from ..topics import subscribe

if TYPE_CHECKING:
    from ..agent import Node

log = get_logger("so101")

CAP = "arm.so101"
STATE_TOPIC = "arm.so101.state"
TELEOP_TOPIC = "teleop.so101"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DEFAULT_IDS = {name: i + 1 for i, name in enumerate(JOINTS)}
TICKS = 4096
TICKS_PER_DEG = TICKS / 360.0
ADDR_TORQUE = 40
ADDR_GOAL = 42
ADDR_PRESENT = 56


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _wrap_deg(d: float) -> float:
    """Angle difference folded into [-180, 180) so alpha crossing 0/360 doesn't spin the base."""
    return (d + 180.0) % 360.0 - 180.0


class MotionAborted(RuntimeError):
    pass


# ====================================================================== drivers
class ArmDriver:
    """Sync interface; `blocking` drivers are called through a thread."""
    name = "base"
    blocking = False

    def open(self) -> None: ...
    def close(self) -> None: ...
    def read_positions(self) -> dict[str, int]: raise NotImplementedError
    def write_goals(self, goals: dict[str, int]) -> None: raise NotImplementedError
    def set_torque(self, on: bool) -> None: ...


class MockArm(ArmDriver):
    """Simulated arm: each joint moves toward its goal at `speed` ticks/s while torque is on."""
    name = "mock"

    def __init__(self, start: dict[str, int] | None = None, speed: float = 1500.0, clock=time.monotonic):
        self.speed = float(speed)
        self.clock = clock
        self.present = {j: float((start or {}).get(j, 2048)) for j in JOINTS}
        self.goal = dict(self.present)
        self.torque = True
        self._t = clock()
        self.writes = 0

    def _advance(self) -> None:
        now = self.clock()
        dt, self._t = max(0.0, now - self._t), now
        if not self.torque:
            return
        step = self.speed * dt
        for j in JOINTS:
            d = self.goal[j] - self.present[j]
            self.present[j] += _clamp(d, -step, step)

    def read_positions(self) -> dict[str, int]:
        self._advance()
        return {j: int(round(p)) for j, p in self.present.items()}

    def write_goals(self, goals: dict[str, int]) -> None:
        self._advance()
        for j, v in goals.items():
            self.goal[j] = float(_clamp(int(v), 0, TICKS - 1))
        self.writes += 1

    def set_torque(self, on: bool) -> None:
        self._advance()
        self.torque = bool(on)
        if not on:
            self.goal = dict(self.present)


class LeRobotFeetech(ArmDriver):
    """Real bus via LeRobot's FeetechMotorsBus, raw ticks (normalize=False)."""
    name = "lerobot"
    blocking = True

    def __init__(self, port: str, ids: dict[str, int]):
        try:
            from lerobot.motors import Motor, MotorNormMode
            from lerobot.motors.feetech import FeetechMotorsBus
        except ImportError:
            from lerobot.common.motors import Motor, MotorNormMode            # older LeRobot layout
            from lerobot.common.motors.feetech import FeetechMotorsBus
        motors = {j: Motor(ids[j], "sts3215", MotorNormMode.RANGE_0_100 if j == "gripper"
                           else MotorNormMode.RANGE_M100_100) for j in JOINTS}
        self.bus = FeetechMotorsBus(port=port, motors=motors)

    def open(self) -> None:
        self.bus.connect()
        # Never jump on torque-up: make the goal equal to where the arm is right now.
        self.write_goals(self.read_positions())
        self.bus.enable_torque()

    def close(self) -> None:
        try:
            self.bus.disconnect(disable_torque=False)
        except TypeError:
            self.bus.disconnect()

    def read_positions(self) -> dict[str, int]:
        raw = self.bus.sync_read("Present_Position", normalize=False)
        return {j: int(raw[j]) for j in JOINTS}

    def write_goals(self, goals: dict[str, int]) -> None:
        self.bus.sync_write("Goal_Position", {j: int(v) for j, v in goals.items()}, normalize=False)

    def set_torque(self, on: bool) -> None:
        (self.bus.enable_torque if on else self.bus.disable_torque)()


class ScsServo(ArmDriver):
    """Real bus via feetech-servo-sdk (scservo_sdk): raw group sync read/write."""
    name = "scservo"
    blocking = True

    def __init__(self, port: str, ids: dict[str, int], baudrate: int = 1_000_000):
        import scservo_sdk as scs
        self.scs = scs
        self.ids = ids
        self.port = scs.PortHandler(port)
        self.ph = scs.PacketHandler(0)               # STS series: protocol end 0 (little endian)
        self.baudrate = baudrate
        self.reader = scs.GroupSyncRead(self.port, self.ph, ADDR_PRESENT, 2)
        self.writer = scs.GroupSyncWrite(self.port, self.ph, ADDR_GOAL, 2)

    def open(self) -> None:
        if not self.port.openPort():
            raise OSError(f"cannot open {self.port.getPortName()}")
        if not self.port.setBaudRate(self.baudrate):
            raise OSError(f"cannot set baudrate {self.baudrate}")
        for j in JOINTS:
            if not self.reader.addParam(self.ids[j]):
                raise RuntimeError(f"sync read addParam failed for {j}")
        self.write_goals(self.read_positions())       # goal := present before torque-up
        self.set_torque(True)

    def close(self) -> None:
        self.port.closePort()

    def read_positions(self) -> dict[str, int]:
        res = self.reader.txRxPacket()
        if res != self.scs.COMM_SUCCESS:
            raise IOError(f"sync read failed: {self.ph.getTxRxResult(res)}")
        out = {}
        for j in JOINTS:
            sid = self.ids[j]
            if not self.reader.isAvailable(sid, ADDR_PRESENT, 2):
                raise IOError(f"no position from servo {sid} ({j})")
            out[j] = int(_clamp(self.reader.getData(sid, ADDR_PRESENT, 2) & 0x7FFF, 0, TICKS - 1))
        return out

    def write_goals(self, goals: dict[str, int]) -> None:
        self.writer.clearParam()
        for j, v in goals.items():
            v = int(_clamp(int(v), 0, TICKS - 1))
            self.writer.addParam(self.ids[j], [v & 0xFF, (v >> 8) & 0xFF])
        res = self.writer.txPacket()
        self.writer.clearParam()
        if res != self.scs.COMM_SUCCESS:
            raise IOError(f"sync write failed: {self.ph.getTxRxResult(res)}")

    def set_torque(self, on: bool) -> None:
        for j in JOINTS:
            self.ph.write1ByteTxRx(self.port, self.ids[j], ADDR_TORQUE, 1 if on else 0)


def open_driver(conf: dict) -> tuple[ArmDriver, str | None]:
    """Returns (opened driver, warning or None). Raises only if fallback_to_mock is false."""
    port = str(conf.get("port") or "")
    ids = {**DEFAULT_IDS, **(conf.get("ids") or {})}
    mock_speed = float(conf.get("mock_speed_ticks_s", 1500))
    if conf.get("mock") is True:
        return MockArm(speed=mock_speed), None
    errors = []
    if not port:
        errors.append("no port configured")
    else:
        for cls in (LeRobotFeetech, ScsServo):
            drv = None
            try:
                drv = cls(port, ids) if cls is LeRobotFeetech else cls(port, ids, int(conf.get("baudrate", 1_000_000)))
                drv.open()
                return drv, None
            except Exception as e:  # noqa: BLE001 - ImportError, serial errors, missing motors...
                errors.append(f"{cls.name}: {type(e).__name__}: {e}")
                if drv is not None:
                    try:
                        drv.close()
                    except Exception:  # noqa: BLE001
                        pass
    msg = f"{CAP}: real bus unavailable ({'; '.join(errors)})"
    if conf.get("fallback_to_mock", True) is False:
        raise RuntimeError(msg)
    return MockArm(speed=mock_speed), msg + " -> using MOCK arm"


# ====================================================================== calibration
def load_calibration(path: str | None, start: dict[str, int] | None, span: int,
                     real: bool) -> tuple[dict[str, dict], dict[str, dict], str | None]:
    """Returns (bounds {joint: {min,max,home}}, poses, warning)."""
    warn = None
    raw: dict = {}
    if path:
        with open(Path(path), "r", encoding="utf-8") as f:
            raw = json.load(f) or {}
    jraw = raw.get("joints") if isinstance(raw.get("joints"), dict) else \
        {k: v for k, v in raw.items() if k in JOINTS}
    bounds = {}
    for j in JOINTS:
        c = jraw.get(j) or {}
        lo = c.get("min", c.get("range_min"))
        hi = c.get("max", c.get("range_max"))
        if lo is None or hi is None:
            if real and start:
                # Uncalibrated real arm: only allow a small window around where it booted.
                p = start[j]
                lo, hi = max(0, p - span), min(TICKS - 1, p + span)
                warn = f"{CAP}: no calibration for some joints; limited to +-{span} ticks around the boot pose"
            else:
                lo, hi = 1024, 3072
        lo, hi = int(_clamp(min(lo, hi), 0, TICKS - 1)), int(_clamp(max(lo, hi), 0, TICKS - 1))
        home = c.get("home")
        if home is None:
            home = start[j] if (real and start and "min" not in c and "range_min" not in c) else (lo + hi) // 2
        bounds[j] = {"min": lo, "max": hi, "home": int(_clamp(int(home), lo, hi))}
    poses = raw.get("poses") if isinstance(raw.get("poses"), dict) else {}
    return bounds, poses, warn


# ====================================================================== controller
class ArmController:
    def __init__(self, driver: ArmDriver, conf: dict, bounds: dict[str, dict], poses: dict,
                 publish=None, event=None):
        self.drv = driver
        self.conf = conf
        self.bounds = bounds
        self.poses = poses
        self._publish = publish
        self._event = event or (lambda msg, level="info": log.info(msg))
        self.max_step = max(1, int(conf.get("max_step_ticks", 15)))
        self.rate_hz = float(_clamp(float(conf.get("rate_hz", 30)), 5, 200))
        self.telemetry_hz = float(_clamp(float(conf.get("telemetry_hz", 10)), 0.5, 50))
        self.tol = int(conf.get("tolerance_ticks", 20))
        self.settle_s = float(conf.get("settle_s", 2.0))
        self.move_timeout_s = float(conf.get("move_timeout_s", 30))
        self.torque_off_on_estop = bool(conf.get("torque_off_on_estop", False))
        self.estop_latch = bool(conf.get("estop_latch", False))
        self.teleop_preempts = bool(conf.get("teleop_preempts_tasks", True))
        tc = conf.get("teleop") or {}
        self.gain = {"shoulder_pan": float(tc.get("gain_pan", 1.0)),
                     "shoulder_lift": float(tc.get("gain_lift", 1.0)),
                     "elbow_flex": float(tc.get("gain_elbow", -1.0)),
                     "wrist_roll": float(tc.get("gain_roll", 1.0))}
        self.beta_split = float(_clamp(float(tc.get("beta_split", 0.5)), 0.0, 1.0))
        self.deadman_s = float(tc.get("deadman_ms", 300)) / 1000.0
        self.release_s = float(tc.get("release_s", 2.0))

        self.present: dict[str, int] = {}
        self.cmd: dict[str, float] = {}       # last goal written (rate-limited)
        self.target: dict[str, float] = {}    # where we are heading
        self.speed = 1.0
        self.mode = "idle"                    # idle | task | teleop
        self.estop = False
        self.estop_reason = ""
        self.fault: str | None = None
        self.torque = True
        self._hold_req = False
        self._gen = 0                         # bumped on stop / preempt: running motions abort
        self._abort_reason = ""
        self._io_fail = 0
        self._motion_lock = asyncio.Lock()
        self._loops: list[asyncio.Task] = []
        # teleop
        self.tele_src: str | None = None
        self.tele_last = 0.0                  # monotonic time of the latched source's last sample
        self.tele_enabled = False
        self.tele_active = False              # enabled and deadman held and fresh
        self.tele_zero: tuple[float, float, float] | None = None
        self.tele_base: dict[str, float] = {}
        self.tele_rejected = 0

    # -------------------------------------------------------------- io
    async def _io(self, fn, *args):
        if self.drv.blocking:
            return await asyncio.to_thread(fn, *args)
        return fn(*args)

    async def start(self, loops: bool = True) -> None:
        self.present = await self._io(self.drv.read_positions)
        self.cmd = {j: float(v) for j, v in self.present.items()}
        self.target = dict(self.cmd)
        if loops:
            self._loops = [asyncio.create_task(self._control_loop()),
                           asyncio.create_task(self._telemetry_loop())]

    async def stop_loops(self) -> None:
        for t in self._loops:
            t.cancel()
        for t in self._loops:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._loops = []

    def close(self) -> None:
        try:
            self.drv.close()
        except Exception:  # noqa: BLE001
            log.exception("driver close failed")

    # -------------------------------------------------------------- helpers
    def clamp_joint(self, j: str, v: float) -> int:
        b = self.bounds[j]
        return int(round(_clamp(v, b["min"], b["max"])))

    def to_ticks(self, j: str, v: Any, units: str = "auto") -> int:
        if j not in self.bounds:
            raise ValueError(f"unknown joint {j!r} (want {JOINTS})")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{j}: value must be a number, got {v!r}")
        b = self.bounds[j]
        norm = units == "norm" or (units == "auto" and isinstance(v, float) and -1.0 <= v <= 1.0)
        if norm:
            v = b["min"] + (_clamp(float(v), -1.0, 1.0) + 1.0) / 2.0 * (b["max"] - b["min"])
        return self.clamp_joint(j, float(v))

    def gripper_norm(self, ticks: float) -> float:
        b = self.bounds["gripper"]
        return round(_clamp((ticks - b["min"]) / max(1, b["max"] - b["min"]), 0.0, 1.0), 3)

    def state(self) -> dict:
        return {"joints": dict(self.present), "gripper": self.gripper_norm(self.present.get("gripper", 0)),
                "mode": self.mode, "estop": self.estop, "estop_reason": self.estop_reason,
                "fault": self.fault, "driver": self.drv.name, "torque": self.torque,
                "targets": {j: int(round(v)) for j, v in self.target.items()},
                "limits": {j: [b["min"], b["max"]] for j, b in self.bounds.items()},
                "teleop_source": self.tele_src, "teleop_active": self.tele_active}

    def hold(self, reason: str, estop: bool = False) -> None:
        """Stop now: freeze the setpoint (sync, safe to call from a cancel handler); the control
        loop then snaps it to the measured position on its next tick."""
        self.target = dict(self.cmd)
        self._hold_req = True
        self._gen += 1
        self._abort_reason = reason
        if estop:
            if not self.estop:
                self._event(f"{CAP} E-STOP: {reason}", "warning")
            self.estop, self.estop_reason = True, reason

    def _clear_estop_for_command(self) -> None:
        if self.estop:
            if self.estop_latch:
                raise RuntimeError(f"e-stop latched ({self.estop_reason}); send {{'action': 'reset'}} first")
            self.estop, self.estop_reason = False, ""

    async def _ensure_torque(self) -> None:
        if not self.torque:
            pos = await self._io(self.drv.read_positions)
            self.present = pos
            self.cmd = {j: float(v) for j, v in pos.items()}
            self.target = dict(self.cmd)
            await self._io(self.drv.write_goals, {j: int(v) for j, v in pos.items()})
            await self._io(self.drv.set_torque, True)
            self.torque = True

    async def _resolve_hold(self) -> None:
        """Apply a pending hold now, so it can't overwrite the target of the command that follows."""
        if self._hold_req:
            self._hold_req = False
            self.present = await self._io(self.drv.read_positions)
            self.cmd = {j: float(v) for j, v in self.present.items()}
            self.target = dict(self.cmd)
            if self.torque:
                await self._io(self.drv.write_goals, dict(self.present))

    # -------------------------------------------------------------- control loop
    async def tick(self) -> None:
        now = time.monotonic()
        if self.tele_src is not None:
            silent = now - self.tele_last
            if self.tele_active and silent > self.deadman_s:
                self.tele_active = False
                self.hold("teleop deadman timeout")
            if silent > self.release_s:
                self._event(f"{CAP} teleop source {self.tele_src} released (silent {silent:.1f}s)")
                self.tele_src, self.tele_enabled, self.tele_active = None, False, False
                if self.mode == "teleop":
                    self.mode = "idle"
        try:
            self.present = await self._io(self.drv.read_positions)
            if self._hold_req:
                self._hold_req = False
                self.cmd = {j: float(v) for j, v in self.present.items()}
                self.target = dict(self.cmd)
                if self.torque:
                    await self._io(self.drv.write_goals, dict(self.present))
                if self.estop and self.torque_off_on_estop and self.torque:
                    await self._io(self.drv.set_torque, False)
                    self.torque = False
            elif not self.torque:
                self.cmd = {j: float(v) for j, v in self.present.items()}   # follow a limp arm
                self.target = dict(self.cmd)
            else:
                step = self.max_step * _clamp(self.speed, 0.05, 1.0)
                changed = {}
                for j in JOINTS:
                    c, t = self.cmd[j], self.target[j]
                    if c != t:
                        c = c + _clamp(t - c, -step, step)
                        self.cmd[j] = c
                        changed[j] = int(round(c))
                if changed:
                    await self._io(self.drv.write_goals, changed)
            self._io_fail = 0
            if self.fault and self.fault.startswith("bus"):
                self.fault = None
        except Exception as e:  # noqa: BLE001
            self._io_fail += 1
            if self._io_fail == 1:
                log.warning("%s bus error: %s", CAP, e)
            if self._io_fail >= 5 and not self.estop:
                self.fault = f"bus: {type(e).__name__}: {e}"
                self.hold(f"bus failure: {e}", estop=True)

    async def _control_loop(self) -> None:
        period = 1.0 / self.rate_hz
        while True:
            t0 = time.monotonic()
            await self.tick()
            await asyncio.sleep(max(0.0, period - (time.monotonic() - t0)))

    async def _telemetry_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0 / self.telemetry_hz)
            if self._publish:
                try:
                    await self._publish(STATE_TOPIC, self.state())
                except Exception:  # noqa: BLE001
                    log.exception("telemetry publish failed")

    # -------------------------------------------------------------- task motions
    def _check_can_move(self) -> None:
        if self.mode == "teleop" or self.tele_active:
            raise RuntimeError(f"teleop active (source {self.tele_src}); release the phone deadman/"
                               f"disable teleop first")
        if self.fault:
            raise RuntimeError(f"arm fault: {self.fault}")

    async def move_to(self, goal: dict[str, int], progress, label: str, speed: float = 1.0) -> dict:
        self._check_can_move()
        self._clear_estop_for_command()
        await self._ensure_torque()
        await self._resolve_hold()
        gen = self._gen
        goal = {j: self.clamp_joint(j, v) for j, v in goal.items()}
        self.target = dict(self.cmd)
        self.target.update({j: float(v) for j, v in goal.items()})
        self.speed = _clamp(float(speed), 0.05, 1.0)
        start = {j: self.present.get(j, int(self.cmd[j])) for j in goal}
        dist0 = max([abs(goal[j] - start[j]) for j in goal] + [1])
        t0 = time.monotonic()
        t_cmd_done = None
        last_p = 0.0
        period = 1.0 / self.rate_hz
        try:
            while True:
                await asyncio.sleep(period)
                if self._gen != gen:
                    raise MotionAborted(self._abort_reason or "stopped")
                if self.fault:
                    raise RuntimeError(f"arm fault: {self.fault}")
                now = time.monotonic()
                err = {j: abs(goal[j] - self.present.get(j, goal[j])) for j in goal}
                cmd_done = all(abs(self.cmd[j] - goal[j]) < 0.5 for j in goal)
                if cmd_done and t_cmd_done is None:
                    t_cmd_done = now
                if now - last_p >= 0.25:
                    last_p = now
                    pct = round(100 * _clamp(1 - max(err.values()) / dist0, 0, 1))
                    await progress({"step": label, "pct": pct, "target": goal,
                                    "joints": {j: self.present.get(j) for j in goal}})
                if cmd_done and max(err.values()) <= self.tol:
                    return {"reached": goal, "settled": True, "error_ticks": err}
                if t_cmd_done is not None and now - t_cmd_done > self.settle_s:
                    # e.g. gripper squeezing a part: goal sent, servo blocked. Hold where it is.
                    return {"reached": goal, "settled": False, "error_ticks": err}
                if now - t0 > self.move_timeout_s:
                    self.hold(f"{label}: timeout")
                    raise TimeoutError(f"{label}: not reached after {self.move_timeout_s}s (errors {err})")
        finally:
            self.speed = 1.0

    async def run_action(self, step: dict, progress, depth: int = 0) -> Any:
        action = step.get("action")
        units = str(step.get("units", "auto"))
        speed = float(step.get("speed", 1.0))
        if action == "stop":
            self.hold("stop requested")
            return {"stopped": True}
        if action == "reset":
            self.estop, self.estop_reason = False, ""
            if self.fault and not self.fault.startswith("bus"):
                self.fault = None
            return {"reset": True}
        if action == "home":
            return await self.move_to({j: b["home"] for j, b in self.bounds.items()}, progress, "home", speed)
        if action == "move_joints":
            joints = step.get("joints")
            if not isinstance(joints, dict) or not joints:
                raise ValueError("move_joints needs joints: {name: ticks | -1..1}")
            goal = {j: self.to_ticks(j, v, units) for j, v in joints.items()}
            return await self.move_to(goal, progress, "move_joints", speed)
        if action == "gripper":
            v = step.get("value")
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError("gripper needs value 0..1")
            b = self.bounds["gripper"]
            goal = {"gripper": self.clamp_joint("gripper", b["min"] + _clamp(float(v), 0, 1) * (b["max"] - b["min"]))}
            return await self.move_to(goal, progress, "gripper", speed)
        if action == "pose":
            name = step.get("name")
            pose = self.poses.get(name)
            if not isinstance(pose, dict):
                raise ValueError(f"unknown pose {name!r}; known: {sorted(self.poses)}")
            goal = {j: self.to_ticks(j, v, units) for j, v in pose.items()}
            return await self.move_to(goal, progress, f"pose:{name}", speed)
        if action == "sequence":
            if depth > 0:
                raise ValueError("nested sequences are not allowed")
            steps = step.get("steps")
            if not isinstance(steps, list) or not steps:
                raise ValueError("sequence needs steps: [ {action...}, ... ]")
            for s in steps:                    # validate everything before moving at all
                if not isinstance(s, dict) or s.get("action") not in ACTIONS - {"sequence", "reset"}:
                    raise ValueError(f"bad sequence step {s!r}")
            results = []
            for i, s in enumerate(steps):
                async def sub(p: dict, i=i) -> None:
                    await progress({**p, "seq_index": i, "seq_len": len(steps)})
                await progress({"step": f"sequence {i + 1}/{len(steps)}: {s['action']}",
                                "seq_index": i, "seq_len": len(steps)})
                results.append(await self.run_action(s, sub, depth + 1))
                if s["action"] == "stop":
                    break
            return {"steps": results}
        raise ValueError(f"unknown action {action!r} (want {sorted(ACTIONS)})")

    async def handle(self, args: dict, progress) -> Any:
        action = args.get("action")
        if action in ("stop", "reset"):        # never queued behind a running motion
            return await self.run_action(args, progress)
        self._check_can_move()
        async with self._motion_lock:
            self.mode = "task"
            try:
                return await self.run_action(args, progress)
            except asyncio.CancelledError:
                self.hold("task cancelled", estop=True)
                raise
            finally:
                if self.mode == "task":
                    self.mode = "idle"

    # -------------------------------------------------------------- teleop
    async def on_teleop(self, sample: dict) -> None:
        data = sample.get("data")
        if not isinstance(data, dict):
            return
        now = time.monotonic()
        src = f"{sample.get('origin', '?')}/{data.get('source') or '-'}"
        if self.tele_src is not None and src != self.tele_src:
            if now - self.tele_last <= self.release_s:
                self.tele_rejected += 1
                if self.tele_rejected in (1, 100):
                    self._event(f"{CAP} teleop from {src} ignored: {self.tele_src} has control", "warning")
                return
            self.tele_src = None
        enabled = bool(data.get("enabled"))
        deadman = bool(data.get("deadman"))
        if self.tele_src is None:
            if not enabled:
                return
            if self.mode == "task" and not (self.teleop_preempts and deadman):
                return                         # a task is running and nobody is grabbing control
            self.tele_src, self.tele_rejected = src, 0
            self._event(f"{CAP} teleop latched to {src}")
        self.tele_last = now
        self.tele_enabled = enabled
        if not enabled:
            if self.tele_active:
                self.hold("teleop disabled")
            self.tele_active = False
            self.tele_src = None
            if self.mode == "teleop":
                self.mode = "idle"
            return
        if not deadman or self.fault:
            if self.tele_active:
                self.hold("teleop deadman released")
            self.tele_active = False
            if self.mode != "task":
                self.mode = "teleop"
            return
        try:
            a, b, g = (float(data.get(k) or 0.0) for k in ("alpha", "beta", "gamma"))
            grip = data.get("grip")
            grip = None if grip is None else _clamp(float(grip), 0.0, 1.0)
        except (TypeError, ValueError):
            return
        if not self.tele_active:                # rising edge: take control, capture zero
            if self.mode == "task":
                self.hold("preempted by teleop")   # running task aborts with this reason
            try:
                self._clear_estop_for_command()
            except RuntimeError as e:
                if self.tele_rejected == 0:
                    self._event(f"{CAP} teleop refused: {e}", "warning")
                self.tele_rejected += 1
                return
            await self._ensure_torque()
            await self._resolve_hold()
            self.mode = "teleop"
            self.tele_active = True
            self.tele_zero = (a, b, g)
            self.tele_base = dict(self.cmd)
            self.speed = 1.0
        self.mode = "teleop"
        za, zb, zg = self.tele_zero
        da, db, dg = _wrap_deg(a - za), _wrap_deg(b - zb), _wrap_deg(g - zg)
        base, k = self.tele_base, TICKS_PER_DEG
        tgt = dict(self.target)
        tgt["shoulder_pan"] = base["shoulder_pan"] + self.gain["shoulder_pan"] * da * k
        tgt["shoulder_lift"] = base["shoulder_lift"] + self.gain["shoulder_lift"] * db * self.beta_split * k
        tgt["elbow_flex"] = base["elbow_flex"] + self.gain["elbow_flex"] * db * (1 - self.beta_split) * k
        tgt["wrist_roll"] = base["wrist_roll"] + self.gain["wrist_roll"] * dg * k
        if grip is not None:
            gb = self.bounds["gripper"]
            tgt["gripper"] = gb["min"] + grip * (gb["max"] - gb["min"])
        self.target = {j: float(self.clamp_joint(j, v)) for j, v in tgt.items()}


ACTIONS = {"home", "move_joints", "gripper", "pose", "sequence", "stop", "reset"}

# ====================================================================== module wiring
_CTL: ArmController | None = None
_DONE: dict[str, Any] = {}          # idempotency cache: task_id -> result
_SUBSCRIBED = False


def get_controller() -> ArmController | None:
    return _CTL


async def build_controller(conf: dict, publish=None, event=None, loops: bool = True) -> ArmController:
    drv, warn = await asyncio.to_thread(open_driver, conf)
    if warn:
        (event or (lambda m, level="info": log.warning(m)))(warn, "warning")
    real = drv.name != "mock"
    start = await asyncio.to_thread(drv.read_positions) if drv.blocking else drv.read_positions()
    bounds, poses, cwarn = load_calibration(conf.get("calibration"), start,
                                            int(conf.get("uncalibrated_span", 400)), real)
    if cwarn:
        (event or (lambda m, level="info": log.warning(m)))(cwarn, "warning")
    ctl = ArmController(drv, conf, bounds, poses, publish=publish, event=event)
    await ctl.start(loops=loops)
    return ctl


async def setup(node: "Node") -> None:
    global _CTL, _SUBSCRIBED
    if CAP not in node.cfg.capabilities:
        return
    conf = dict(node.cfg.capability_config.get(CAP, {}) or {})
    _CTL = await build_controller(conf, publish=node.publish, event=node.event)
    node.event(f"{CAP} ready on {node.cfg.node_id}: driver {_CTL.drv.name}, "
               f"{_CTL.max_step} ticks/step @ {_CTL.rate_hz:g} Hz")
    if not _SUBSCRIBED:
        subscribe(TELEOP_TOPIC, _on_teleop_sample)
        _SUBSCRIBED = True


async def shutdown() -> None:
    """Stop loops and close the bus (used by tests; the agent process just exits)."""
    global _CTL
    if _CTL is not None:
        await _CTL.stop_loops()
        _CTL.close()
    _CTL = None
    _DONE.clear()


async def _on_teleop_sample(sample: dict) -> None:
    if _CTL is not None and sample.get("topic") == TELEOP_TOPIC:
        await _CTL.on_teleop(sample)


@register_capability(CAP)
async def so101(args: dict, progress) -> Any:
    tid = args.get("_task_id", "")
    if tid and tid in _DONE:               # at-least-once delivery: never move twice
        return _DONE[tid]
    if _CTL is None:
        raise RuntimeError(f"{CAP} is not set up on this node (add it to capabilities)")
    try:
        result = await _CTL.handle({k: v for k, v in args.items() if k != "_task_id"}, progress)
    except MotionAborted as e:
        raise RuntimeError(f"motion aborted: {e}") from None
    if tid:
        _DONE[tid] = result
        if len(_DONE) > 500:
            _DONE.pop(next(iter(_DONE)))
    return result
