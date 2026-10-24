"""Pluggable Cartesian arm layer for the robot cell (docs/ROBOT_CELL.md).

Units everywhere in this module: metres, metres/second and degrees, in the ARM BASE frame
(the same frame `swarmmind.cell.perception.table_to_arm` produces). Drivers convert to the
vendor's units (mm for xArm, radians/rotation vectors for UR, ...) internally.

Layers:
    Arm        abstract driver (drivers/mock.py, kinova.py, xarm.py, ur.py, so101.py)
    Safety     wraps one Arm. Bounds-checks every target against `workspace`, caps speed,
               owns the stop/e-stop state. Capabilities only ever talk to a Safety.
    pick/place deterministic primitives built only from Safety.move_to / Safety.gripper.

Gripper convention (all drivers): value 1.0 = fully OPEN, 0.0 = fully CLOSED (the same as
capabilities/so101.py). Drivers whose hardware is the other way round (Kinova: 0 = open) invert.

Yaw: rotation of the tool about the base z axis, degrees, tool pointing straight down.
yaw 0 = the driver's tool orientation with `yaw_offset_deg` added (see each driver).

Common config keys (capability_config["arm.<id>"]):
    driver: mock | kinova | xarm | ur | so101
    workspace: {x: [min, max], y: [min, max], z: [min, max]}     metres, arm base frame
               (required for real drivers; mock defaults to DEFAULT_WORKSPACE)
    out_of_bounds: reject | clamp       move_to only; pick/place always reject (default reject)
    max_speed: 0.10                     m/s, hard cap on every move
    speed: 0.05                         m/s, default travel speed
    slow_speed: 0.02                    m/s, pick/place descent
    retreat_speed: 0.03                 m/s, pick/place retreat
    insert_speed: 0.005                 m/s, final insert of place
    insert_step_m: 0.002                metres per insert step
    insert_dwell_s: 0.05                pause after each insert step (lets compliance settle)
    approach_h: 0.08                    metres above the target when not given in the task
    grip_open: 1.0, grip_close: 0.0     gripper values used by pick/place
    home: {x, y, z, yaw}                Cartesian home pose (optional for kinova: its "Home" action)
    estop_latch: false                  true: after an e-stop only {"action": "reset"} clears it
"""
from __future__ import annotations

import abc
import asyncio
import math
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from ..logutil import get_logger

log = get_logger("cell.arm")

Progress = Callable[[dict], Awaitable[None]]

DEFAULT_WORKSPACE = {"x": (0.15, 0.70), "y": (-0.45, 0.45), "z": (0.02, 0.60)}
AXES = ("x", "y", "z")


class MotionAborted(RuntimeError):
    """A motion was interrupted by stop / e-stop."""


class ArmConfigError(ValueError):
    pass


def wrap_deg(a: float) -> float:
    """Fold an angle into [-180, 180)."""
    return (float(a) + 180.0) % 360.0 - 180.0


def num(v: Any, what: str) -> float:
    """Strict finite-number check for task args (bools and strings are refused)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{what} must be a number, got {v!r}")
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"{what} must be finite, got {v!r}")
    return f


async def _noop_progress(_p: dict) -> None:
    return None


# ====================================================================== workspace
@dataclass
class Workspace:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    @classmethod
    def from_cfg(cls, ws: Optional[dict]) -> "Workspace":
        """Accepts {x: [lo, hi], ...} or {x_min, x_max, ...}."""
        ws = ws or {}
        out = {}
        for a in AXES:
            if a in ws:
                pair = ws[a]
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ArmConfigError(f"workspace.{a} must be [min, max], got {pair!r}")
                lo, hi = pair
            elif f"{a}_min" in ws or f"{a}_max" in ws:
                lo, hi = ws.get(f"{a}_min"), ws.get(f"{a}_max")
                if lo is None or hi is None:
                    raise ArmConfigError(f"workspace needs both {a}_min and {a}_max")
            else:
                lo, hi = DEFAULT_WORKSPACE[a]
            lo, hi = num(lo, f"workspace.{a}.min"), num(hi, f"workspace.{a}.max")
            if lo >= hi:
                raise ArmConfigError(f"workspace.{a}: min {lo} must be < max {hi}")
            out[a] = (lo, hi)
        return cls(**out)

    def bounds(self, axis: str) -> tuple[float, float]:
        return getattr(self, axis)

    def violations(self, x: float, y: float, z: float) -> list[str]:
        out = []
        for a, v in zip(AXES, (x, y, z)):
            lo, hi = self.bounds(a)
            if not lo <= v <= hi:
                out.append(f"{a}={v:.4f} outside [{lo}, {hi}]")
        return out

    def clamp(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        res = []
        for a, v in zip(AXES, (x, y, z)):
            lo, hi = self.bounds(a)
            res.append(min(hi, max(lo, v)))
        return tuple(res)  # type: ignore[return-value]

    def center(self) -> tuple[float, float, float]:
        return tuple((lo + hi) / 2.0 for lo, hi in (self.x, self.y, self.z))  # type: ignore[return-value]

    def wire(self) -> dict:
        return {"x": list(self.x), "y": list(self.y), "z": list(self.z)}


# ====================================================================== driver interface
class Arm(abc.ABC):
    """One physical (or simulated) arm. Async; blocking SDKs run their calls in threads.

    Drivers never bounds-check: Safety does that before calling them. Drivers DO convert units
    and must make stop() safe to call while a move_to is in flight (from the same event loop).
    """
    name = "base"
    supports_cartesian = True

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg or {})
        self.workspace = Workspace.from_cfg(self.cfg.get("workspace"))
        self.grip: Optional[float] = None             # last known gripper value, 1 open .. 0 closed

    # ---- lifecycle
    @abc.abstractmethod
    async def connect(self) -> None: ...

    async def close(self) -> None:
        return None

    async def reset(self) -> None:
        """Clear driver/hardware faults that are safe to clear (no motion)."""
        return None

    # ---- motion
    def home_target(self) -> Optional[dict]:
        """Cartesian home {x, y, z, yaw} that Safety bounds-checks and moves to with move_to,
        or None if the driver has its own home motion (then Safety calls home())."""
        h = self.cfg.get("home")
        if isinstance(h, dict):
            return {"x": num(h.get("x"), "home.x"), "y": num(h.get("y"), "home.y"),
                    "z": num(h.get("z"), "home.z"), "yaw": num(h.get("yaw", 0.0), "home.yaw")}
        cx, cy, _ = self.workspace.center()
        zlo, zhi = self.workspace.z
        return {"x": cx, "y": cy, "z": zlo + 0.75 * (zhi - zlo), "yaw": 0.0}

    async def home(self, speed: float) -> None:
        """Only called when home_target() is None."""
        raise NotImplementedError(f"{self.name}: no native home motion")

    @abc.abstractmethod
    async def move_to(self, x: float, y: float, z: float, yaw_deg: float, speed: float) -> None:
        """Straight-line move of the tool to (x, y, z) m, yaw deg, at `speed` m/s. Returns when
        reached; raises if the hardware refuses or aborts."""

    @abc.abstractmethod
    async def gripper(self, value: float) -> None:
        """value 1.0 = open, 0.0 = closed. Returns when the fingers have settled."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop motion now (controlled stop, keeps power). Must not raise."""

    @abc.abstractmethod
    async def pose(self) -> dict:
        """Measured tool pose {x, y, z, yaw} in metres / degrees (values may be None)."""

    def info(self) -> dict:
        return {"driver": self.name}


# ====================================================================== safety wrapper
class Safety:
    """Every command to an arm goes through here."""

    def __init__(self, driver: Arm, cfg: Optional[dict] = None):
        self.drv = driver
        self.cfg = dict(cfg if cfg is not None else driver.cfg)
        c = self.cfg
        self.workspace = driver.workspace
        self.mode = str(c.get("out_of_bounds", "reject"))
        if self.mode not in ("reject", "clamp"):
            raise ArmConfigError(f"out_of_bounds must be reject | clamp, got {self.mode!r}")
        self.max_speed = num(c.get("max_speed", 0.10), "max_speed")
        if self.max_speed <= 0:
            raise ArmConfigError("max_speed must be > 0")
        self.speed = min(num(c.get("speed", 0.05), "speed"), self.max_speed)
        self.slow_speed = min(num(c.get("slow_speed", 0.02), "slow_speed"), self.max_speed)
        self.retreat_speed = min(num(c.get("retreat_speed", 0.03), "retreat_speed"), self.max_speed)
        self.insert_speed = min(num(c.get("insert_speed", 0.005), "insert_speed"), self.max_speed)
        self.insert_step = num(c.get("insert_step_m", 0.002), "insert_step_m")
        self.insert_dwell_s = num(c.get("insert_dwell_s", 0.05), "insert_dwell_s")
        self.approach_h = num(c.get("approach_h", 0.08), "approach_h")
        self.grip_open = min(1.0, max(0.0, num(c.get("grip_open", 1.0), "grip_open")))
        self.grip_close = min(1.0, max(0.0, num(c.get("grip_close", 0.0), "grip_close")))
        self.estop_latch = bool(c.get("estop_latch", False))
        for k, v in (("speed", self.speed), ("slow_speed", self.slow_speed),
                     ("retreat_speed", self.retreat_speed), ("insert_speed", self.insert_speed),
                     ("insert_step_m", self.insert_step)):
            if v <= 0:
                raise ArmConfigError(f"{k} must be > 0")
        self.estop = False
        self.estop_reason = ""
        self.gen = 0                    # bumped by stop/e-stop: in-flight and queued motions abort
        self.abort_reason = ""
        self.target: Optional[dict] = None
        self.last_pose: dict = {"x": None, "y": None, "z": None, "yaw": None}
        self.fault: Optional[str] = None

    # ---- checks
    def cap_speed(self, speed: Optional[float]) -> float:
        if speed is None:
            return self.speed
        s = num(speed, "speed")
        if s <= 0:
            raise ValueError(f"speed must be > 0 m/s, got {s}")
        return min(s, self.max_speed)

    def check(self, x: float, y: float, z: float, yaw: float, allow_clamp: bool = True) -> dict:
        """Validated target {x, y, z, yaw, clamped}. Raises ValueError in reject mode."""
        x, y, z = num(x, "x"), num(y, "y"), num(z, "z")
        yaw = wrap_deg(num(yaw, "yaw"))
        bad = self.workspace.violations(x, y, z)
        clamped = False
        if bad:
            if self.mode == "clamp" and allow_clamp:
                x, y, z = self.workspace.clamp(x, y, z)
                clamped = True
                log.warning("%s: target clamped into workspace (%s)", self.drv.name, "; ".join(bad))
            else:
                raise ValueError("target outside workspace: " + "; ".join(bad))
        return {"x": x, "y": y, "z": z, "yaw": yaw, "clamped": clamped}

    def _require_cartesian(self) -> None:
        if not self.drv.supports_cartesian:
            raise RuntimeError(f"driver {self.drv.name!r} does not support Cartesian moves "
                               f"(no inverse kinematics); use home / gripper / named_pose")

    # ---- e-stop state
    def begin_command(self) -> int:
        """Call at the start of every motion command. Clears a non-latched e-stop; returns the
        generation the command's motions must belong to."""
        if self.estop:
            if self.estop_latch:
                raise RuntimeError(f"e-stop latched ({self.estop_reason}); send {{'action': 'reset'}} first")
            self.estop, self.estop_reason = False, ""
        return self.gen

    def check_gen(self, gen: int) -> None:
        if gen != self.gen:
            raise MotionAborted(self.abort_reason or "stopped")

    async def stop(self, reason: str = "stop requested") -> None:
        """Abort the running motion and any queued ones. Not an e-stop (no latch)."""
        self.gen += 1
        self.abort_reason = reason
        try:
            await self.drv.stop()
        except Exception as e:  # noqa: BLE001 - stop must never raise
            log.error("%s stop() failed: %s", self.drv.name, e)

    async def emergency_stop(self, reason: str) -> None:
        self.estop, self.estop_reason = True, reason
        await self.stop(reason)

    async def reset(self) -> None:
        self.estop, self.estop_reason = False, ""
        self.fault = None
        await self.drv.reset()

    # ---- commands
    async def move_to(self, x: float, y: float, z: float, yaw: float = 0.0, speed: Optional[float] = None,
                      gen: Optional[int] = None, allow_clamp: bool = True) -> dict:
        self._require_cartesian()
        t = self.check(x, y, z, yaw, allow_clamp=allow_clamp)
        s = self.cap_speed(speed)
        gen = self.gen if gen is None else gen
        self.check_gen(gen)
        self.target = {k: t[k] for k in ("x", "y", "z", "yaw")}
        try:
            await self.drv.move_to(t["x"], t["y"], t["z"], t["yaw"], s)
        except MotionAborted:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if gen != self.gen:
                raise MotionAborted(self.abort_reason or "stopped") from None
            raise
        finally:
            self.target = None
        self.check_gen(gen)
        return {**t, "speed": s}

    async def gripper(self, value: float, gen: Optional[int] = None) -> float:
        v = min(1.0, max(0.0, num(value, "gripper value")))
        gen = self.gen if gen is None else gen
        self.check_gen(gen)
        try:
            await self.drv.gripper(v)
        except (MotionAborted, asyncio.CancelledError):
            raise
        except Exception:
            if gen != self.gen:
                raise MotionAborted(self.abort_reason or "stopped") from None
            raise
        self.check_gen(gen)
        return v

    async def home(self, speed: Optional[float] = None, gen: Optional[int] = None) -> dict:
        s = self.cap_speed(speed)
        gen = self.gen if gen is None else gen
        ht = self.drv.home_target()
        if ht is not None:
            r = await self.move_to(ht["x"], ht["y"], ht["z"], ht["yaw"], s, gen=gen, allow_clamp=False)
            return {"home": "pose", **r}
        self.check_gen(gen)
        try:
            await self.drv.home(s)
        except (MotionAborted, asyncio.CancelledError):
            raise
        except Exception:
            if gen != self.gen:
                raise MotionAborted(self.abort_reason or "stopped") from None
            raise
        self.check_gen(gen)
        return {"home": "native", "speed": s}

    async def pose(self) -> dict:
        try:
            p = await self.drv.pose()
            self.last_pose = {k: p.get(k) for k in ("x", "y", "z", "yaw")} | \
                {k: v for k, v in p.items() if k not in ("x", "y", "z", "yaw")}
            if self.fault and self.fault.startswith("pose"):
                self.fault = None
        except Exception as e:  # noqa: BLE001 - telemetry must keep going
            self.fault = f"pose read failed: {type(e).__name__}: {e}"
        return dict(self.last_pose)


# ====================================================================== primitives
async def pick(arm: Safety, x: float, y: float, z: float, yaw: float = 0.0,
               approach_h: Optional[float] = None, speed: Optional[float] = None,
               progress: Optional[Progress] = None, gen: Optional[int] = None) -> dict:
    """open -> approach (lift first if below, then above the part) -> descend slowly -> close ->
    retreat straight up. Every waypoint is validated (always reject) before anything moves."""
    progress = progress or _noop_progress
    gen = arm.gen if gen is None else gen
    arm._require_cartesian()
    ah = arm.approach_h if approach_h is None else num(approach_h, "approach_h")
    if ah <= 0:
        raise ValueError(f"approach_h must be > 0, got {ah}")
    travel = arm.cap_speed(speed)
    grasp = arm.check(x, y, z, yaw, allow_clamp=False)
    above = arm.check(x, y, grasp["z"] + ah, yaw, allow_clamp=False)
    steps = ["open", "approach", "descend", "grip", "retreat"]
    n = len(steps)
    trace: list[dict] = []

    async def mark(i: int, sub: str, **extra) -> None:
        p = {"step": "pick", "sub": sub, "i": i + 1, "n": n, "pct": round(100 * i / n), **extra}
        trace.append({"sub": sub, **extra})
        await progress(p)

    await mark(0, "open", value=arm.grip_open)
    await arm.gripper(arm.grip_open, gen=gen)
    await mark(1, "approach", target=_xyz(above))
    await _approach(arm, above, travel, gen)
    await mark(2, "descend", target=_xyz(grasp), speed=arm.slow_speed)
    await arm.move_to(grasp["x"], grasp["y"], grasp["z"], grasp["yaw"], arm.slow_speed, gen=gen, allow_clamp=False)
    await mark(3, "grip", value=arm.grip_close)
    await arm.gripper(arm.grip_close, gen=gen)
    await mark(4, "retreat", target=_xyz(above), speed=arm.retreat_speed)
    await arm.move_to(above["x"], above["y"], above["z"], above["yaw"], arm.retreat_speed, gen=gen, allow_clamp=False)
    await progress({"step": "pick", "sub": "done", "i": n, "n": n, "pct": 100})
    return {"picked": _xyz(grasp), "approach_h": ah, "steps": [t["sub"] for t in trace]}


async def place(arm: Safety, x: float, y: float, z: float, yaw: float = 0.0,
                approach_h: Optional[float] = None, insert_depth: float = 0.0,
                speed: Optional[float] = None, progress: Optional[Progress] = None,
                gen: Optional[int] = None) -> dict:
    """approach above -> descend slowly to z + insert_depth -> insert the last insert_depth in
    insert_step_m steps at insert_speed -> open -> retreat straight up. (x, y, z) is the SEATED
    position of the held part's grasp point."""
    progress = progress or _noop_progress
    gen = arm.gen if gen is None else gen
    arm._require_cartesian()
    ah = arm.approach_h if approach_h is None else num(approach_h, "approach_h")
    depth = num(insert_depth if insert_depth is not None else 0.0, "insert_depth")
    if depth < 0:
        raise ValueError(f"insert_depth must be >= 0, got {depth}")
    if ah <= 0 or ah < depth:
        raise ValueError(f"approach_h ({ah}) must be > 0 and >= insert_depth ({depth})")
    travel = arm.cap_speed(speed)
    seat = arm.check(x, y, z, yaw, allow_clamp=False)
    entry = arm.check(x, y, seat["z"] + depth, yaw, allow_clamp=False)
    above = arm.check(x, y, seat["z"] + ah, yaw, allow_clamp=False)
    n_ins = max(1, math.ceil(depth / arm.insert_step - 1e-9)) if depth > 0 else 0
    n = 5
    trace: list[str] = []

    async def mark(i: int, sub: str, **extra) -> None:
        trace.append(sub)
        await progress({"step": "place", "sub": sub, "i": i + 1, "n": n, "pct": round(100 * i / n), **extra})

    await mark(0, "approach", target=_xyz(above))
    await _approach(arm, above, travel, gen)
    await mark(1, "descend", target=_xyz(entry), speed=arm.slow_speed)
    await arm.move_to(entry["x"], entry["y"], entry["z"], entry["yaw"], arm.slow_speed, gen=gen, allow_clamp=False)
    await mark(2, "insert", depth=depth, steps=n_ins, speed=arm.insert_speed)
    insert_z = []
    for k in range(1, n_ins + 1):
        zk = entry["z"] - depth * k / n_ins
        if k == n_ins:
            zk = seat["z"]
        await progress({"step": "place", "sub": "insert", "i": 3, "n": n, "insert_i": k, "insert_n": n_ins,
                        "z": round(zk, 5), "pct": round(100 * (2 + k / n_ins) / n)})
        await arm.move_to(seat["x"], seat["y"], zk, seat["yaw"], arm.insert_speed, gen=gen, allow_clamp=False)
        insert_z.append(zk)
        if arm.insert_dwell_s > 0:
            await asyncio.sleep(arm.insert_dwell_s)
            arm.check_gen(gen)
    await mark(3, "release", value=arm.grip_open)
    await arm.gripper(arm.grip_open, gen=gen)
    await mark(4, "retreat", target=_xyz(above), speed=arm.retreat_speed)
    await arm.move_to(above["x"], above["y"], above["z"], above["yaw"], arm.retreat_speed, gen=gen, allow_clamp=False)
    await progress({"step": "place", "sub": "done", "i": n, "n": n, "pct": 100})
    return {"placed": _xyz(seat), "approach_h": ah, "insert_depth": depth,
            "insert_steps": n_ins, "insert_z": [round(v, 5) for v in insert_z], "steps": trace}


async def _approach(arm: Safety, above: dict, travel: float, gen: int) -> None:
    """Go above the target. If the tool is currently lower than the approach height, rise
    vertically first so the lateral move never sweeps low over the table."""
    cur = await arm.pose()
    cx, cy, cz = cur.get("x"), cur.get("y"), cur.get("z")
    if None not in (cx, cy, cz) and cz < above["z"] - 1e-4 and not arm.workspace.violations(cx, cy, above["z"]):
        if math.hypot(cx - above["x"], cy - above["y"]) > 1e-3:
            await arm.move_to(cx, cy, above["z"], above["yaw"] if cur.get("yaw") is None else cur["yaw"], travel, gen=gen, allow_clamp=False)
    await arm.move_to(above["x"], above["y"], above["z"], above["yaw"], travel, gen=gen, allow_clamp=False)


def _xyz(t: dict) -> dict:
    return {"x": round(t["x"], 5), "y": round(t["y"], 5), "z": round(t["z"], 5), "yaw": round(t["yaw"], 3)}


# ====================================================================== registry
def get_driver(name: str, cfg: dict) -> Arm:
    """Instantiate driver `name` (mock | kinova | xarm | ur | so101 | anything registered with
    drivers.register_driver). Does not connect. Vendor SDKs are imported in connect()."""
    from . import drivers
    return drivers.create(name, cfg)


def build_arm(cfg: dict) -> Safety:
    """Driver from cfg["driver"] (default mock) wrapped in Safety."""
    name = str((cfg or {}).get("driver") or "mock")
    if name not in ("mock", "so101") and not (cfg or {}).get("workspace"):
        raise ArmConfigError(f"driver {name!r} needs an explicit workspace {{x: [min, max], y: ..., z: ...}} "
                             f"(metres, arm base frame); refusing to guess bounds for a real arm")
    return Safety(get_driver(name, cfg), cfg)
