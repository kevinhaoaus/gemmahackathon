"""LabMate's shared arm capability "arm.cell" (docs/LABMATE.md, "Arm capability arm.cell", Rules 2-3).

Every arm host that can do LabMate work advertises the same capability name, "arm.cell", so the
leader can hand a task to whichever arm is alive (by node rank, then load). Tasks carry TABLE
coordinates plus an item reference, never arm coordinates: each arm converts with its own `base`.

Active only when "arm.cell" is in node.cfg.capabilities. Config capability_config["arm.cell"]:

    driver: rebot | so101 | mock        (cell/drivers; default mock)
    arm_id: "rebot"                     telemetry topic "arm.cell.<arm_id>.state"
    base: [x, y, yaw_deg, z_table]      this arm's base pose in the TABLE frame (cell/perception.py);
                                        if absent, capability_config.cell.arms["arm.<arm_id>"].base
    workspace: {x: [..], y: [..], z: [..]}   metres, ARM base frame (required for real drivers)
    speed, slow_speed, retreat_speed, approach_h, max_speed, grip_open, grip_close, home ...
                                        common motion keys (cell/arm.py)
    carry_speed: 0.03                   m/s, the level carry between pick and place
    match_radius_m: 0.04                relocation tolerance (Rule 2: 4 cm)
    blobs_max_age_s: 3.0                the latest cell.blobs must be at most this old
    health_interval_s: 1.0              connection check period (unplug -> unavailable in ~1-2 s)
    telemetry_hz: 10
    connect_timeout_s: 15
    ...driver keys (e.g. port: "COM5", see cell/drivers/rebot.py)

Task args (`_task_id` makes them idempotent; cancelling a running task is an e-stop):
    {"action": "move_object", "item": "screwdriver", "blob_id": 3, "table_xy": [x, y], "yaw_deg": 0,
     "height": 0.012, "grasp": "across" | "center", "to": {"table_xy": [x, y], "yaw_deg": 0, "z": 0.02}}
    {"action": "home"}  {"action": "stop"}  {"action": "reset"}

move_object: (a) re-find the item in the latest cell.blobs (same blob id within match_radius_m of
table_xy, else the nearest blob within it; none -> "object not where expected"); this happens
after the task has the arm, right before moving, on every attempt, so a task handed over from a
dead arm never grabs blindly. (b) table -> arm with this arm's base. (c) grasp yaw: "across" =
blob long axis + 90 deg, "center" = blob yaw; folded to the equivalent within +-90 deg of the
tool's current yaw (a parallel gripper is symmetric under 180 deg). (d) pick at `height`, lift to
the carry height, carry level and slowly, place at to.z (table frame), retreat. Every waypoint is
bounds-checked (and IK-checked when the driver can) before anything moves.

Yaw convention: tool pointing down, yaw = direction of the finger (closing) axis in the arm frame.

Availability (Rule 3): advertised only while the driver is connected and answering. A health loop
pings the driver every health_interval_s; a failure marks the arm disconnected (and stops any
motion), and the same loop reconnects when the arm answers again.
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import TYPE_CHECKING, Any, Optional

from ..cell import perception as P
from ..cell.arm import ArmConfigError, MotionAborted, build_arm, num, pick, place, width_to_grip, wrap_deg
from ..logutil import get_logger
from ..tasks import AVAILABILITY, REGISTRY, register_capability, set_availability
from . import cell_arm

if TYPE_CHECKING:
    from ..agent import Node

log = get_logger("arm_cell")

NAME = "arm.cell"
ACTIONS = {"move_object", "home", "stop", "reset"}
NOT_WHERE_EXPECTED = "object not where expected (moved or held by another arm?)"


# ====================================================================== pure helpers
def grasp_yaw_table(blob_yaw_deg: float, grasp: str) -> float:
    """Gripper finger-axis yaw in the table frame for a blob whose long axis is at blob_yaw_deg."""
    if grasp == "across":
        return wrap_deg(blob_yaw_deg + 90.0)
    if grasp == "center":
        return wrap_deg(blob_yaw_deg)
    raise ValueError(f"grasp must be across | center, got {grasp!r}")


def fold_yaw(yaw_deg: float, ref_deg: Optional[float]) -> float:
    """The equivalent of yaw (mod 180, parallel gripper) within +-90 deg of ref (minimal wrist turn)."""
    ref = 0.0 if ref_deg is None else float(ref_deg)
    d = wrap_deg(yaw_deg - ref)
    if d > 90.0:
        d -= 180.0
    elif d < -90.0:
        d += 180.0
    return wrap_deg(ref + d)


def relocate(blobs: list, table_xy, blob_id: Any = None, radius: float = 0.04) -> tuple[dict, float]:
    """(blob, distance) for the item expected at table_xy: the blob with the same id if it is within
    radius, else the nearest blob within radius. Raises RuntimeError if there is none."""
    ex, ey = float(table_xy[0]), float(table_xy[1])
    cands = []
    for b in blobs or []:
        try:
            bx, by = float(b["table_xy"][0]), float(b["table_xy"][1])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        cands.append((math.hypot(bx - ex, by - ey), b))
    if blob_id is not None:
        for d, b in cands:
            if b.get("id") == blob_id and d <= radius:
                return b, d
    near = [(d, b) for d, b in cands if d <= radius]
    if not near:
        nearest = min((d for d, _ in cands), default=None)
        extra = f"; nearest blob {nearest * 100:.1f} cm away" if nearest is not None else "; no blobs seen"
        raise RuntimeError(f"{NOT_WHERE_EXPECTED}: nothing within {radius * 100:.0f} cm of "
                           f"[{ex:.3f}, {ey:.3f}]{extra}")
    d, b = min(near, key=lambda t: t[0])
    return b, d


def _xy(v: Any, what: str) -> tuple[float, float]:
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        raise ValueError(f"{what} must be [x, y] (metres, table frame), got {v!r}")
    return num(v[0], f"{what}[0]"), num(v[1], f"{what}[1]")


# ====================================================================== service
class CellArmService(cell_arm.ArmService):
    """cell_arm.ArmService (connect, telemetry state, e-stop handling) plus move_object, table-frame
    conversion, availability and the reconnecting health loop."""

    def __init__(self, conf: dict, node: "Node"):
        conf = dict(conf or {})
        # arm.cell must advertise only a REAL connected arm; a silent mock stand-in would lie (Rule 3)
        conf.pop("fallback_to_mock", None)
        super().__init__(NAME, conf, node)
        self.arm_id = str(conf.get("arm_id") or self.driver_name)
        self.topic = f"{NAME}.{self.arm_id}.state"
        self.base = self._base(conf, node)
        self.carry_speed = num(conf.get("carry_speed", 0.03), "carry_speed")
        self.radius = num(conf.get("match_radius_m", 0.04), "match_radius_m")
        # gripper closes to (object width from the camera box - squeeze); linear 0 cm = closed .. grip_open_cm = open
        self.grip_open_m = num(conf.get("grip_open_cm", 10.0), "grip_open_cm") / 100.0
        self.squeeze_m = num(conf.get("squeeze_mm", 9.0), "squeeze_mm") / 1000.0
        self.max_age = num(conf.get("blobs_max_age_s", 3.0), "blobs_max_age_s")
        self.health_s = max(0.1, num(conf.get("health_interval_s", 1.0), "health_interval_s"))
        self._health: Optional[asyncio.Task] = None
        self.health_error: Optional[str] = None

    @staticmethod
    def _base(conf: dict, node: "Node") -> list[float]:
        b = conf.get("base")
        if b is None:
            cell = (getattr(node.cfg, "capability_config", {}) or {}).get("cell") or {}
            b = ((cell.get("arms") or {}).get(f"arm.{conf.get('arm_id') or ''}") or {}).get("base")
        if not isinstance(b, (list, tuple)) or len(b) < 3:
            raise ArmConfigError("arm.cell needs base: [x, y, yaw_deg, z_table] (this arm's base pose in the "
                                 "table frame)")
        out = [num(v, "base") for v in b] + [0.0] * (4 - len(b))
        return out[:4]

    # ---------------------------------------------------------------- availability / health
    def available(self) -> bool:
        s = self.safety
        return bool(self.connected and s is not None and s.fault is None and self.health_error is None)

    async def connect(self) -> None:
        """Connect (no mock fallback). Failures are announced once per outage, not every retry."""
        async with self._connect_lock:
            if self.connected:
                return
            if self.safety is None:
                try:
                    self.safety = build_arm(self.conf)
                except Exception as e:  # noqa: BLE001 - bad config is never retried
                    self.connect_error = f"config: {type(e).__name__}: {e}"
                    self._event(f"{NAME} CONFIG ERROR: {e}", "error")
                    return
            s = self.safety
            try:
                await asyncio.wait_for(s.drv.connect(), self.connect_timeout)
            except Exception as e:  # noqa: BLE001 - SDK missing, port absent, motors unpowered...
                err = f"{type(e).__name__}: {e}"
                if err != self.connect_error:
                    self._event(f"{NAME} ({self.arm_id}) not connected: {err}; retrying every "
                                f"{self.health_s:g} s", "warning")
                self.connect_error = err
                try:
                    await s.drv.close()
                except Exception:  # noqa: BLE001
                    pass
                return
            self.connected, self.connect_error, self.health_error = True, None, None
            s.fault = None
            self._event(f"{NAME} ({self.arm_id}) ready: driver {s.drv.name}, base {self.base}, "
                        f"workspace {s.workspace.wire()}")

    async def start(self) -> None:
        await self.connect()
        self._telemetry = asyncio.create_task(self._telemetry_loop())
        self._health = asyncio.create_task(self._health_loop())

    async def _health_loop(self) -> None:
        while True:
            await asyncio.sleep(self.health_s)
            try:
                await self.check_health()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the loop must survive anything
                log.exception("arm.cell health check crashed")

    async def check_health(self) -> None:
        """One health tick: ping when connected (a failure disconnects), reconnect when not."""
        s = self.safety
        if self.connected and s is not None:
            try:
                ping = getattr(s.drv, "ping", None)
                if ping is not None:
                    await asyncio.wait_for(ping(), max(0.5, self.health_s))
                else:
                    await asyncio.wait_for(s.drv.pose(), max(0.5, self.health_s))
                self.health_error = None
                if s.fault and s.fault.startswith("pose"):
                    s.fault = None
                return
            except Exception as e:  # noqa: BLE001
                await self._lost(f"{type(e).__name__}: {e}")
                return
        if s is None and self.connect_error and self.connect_error.startswith("config"):
            return                                   # bad config: never retried
        await self.connect()
        if self.connected:
            self.health_error = None
            if self.safety is not None:
                self.safety.fault = None

    async def _lost(self, err: str) -> None:
        s = self.safety
        self.health_error = err
        self.connected = False
        self.connect_error = err
        self._event(f"{NAME} ({self.arm_id}) lost: {err}; capability withdrawn, reconnecting", "error")
        if s is not None:
            if self.busy:
                await s.stop(f"arm connection lost: {err}")
            try:
                await s.drv.close()
            except Exception:  # noqa: BLE001 - the port may already be gone
                pass

    async def close(self) -> None:
        if self._health:
            self._health.cancel()
            try:
                await self._health
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._health = None
        await super().close()

    # ---------------------------------------------------------------- telemetry
    async def state(self) -> dict:
        st = await super().state()
        st.update({"arm_id": self.arm_id, "base": self.base, "available": self.available(),
                   "health_error": self.health_error})
        return st

    async def _telemetry_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0 / self.hz)
            try:
                await self.node.publish(self.topic, await self.state())
            except Exception:  # noqa: BLE001
                log.exception("%s telemetry publish failed", self.topic)

    # ---------------------------------------------------------------- tasks
    async def handle(self, args: dict, progress) -> Any:
        tid = str(args.get("_task_id") or "")
        if tid and tid in self.done:                 # at-least-once delivery: never move twice
            return self.done[tid]
        action = args.get("action", "move_object")
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r} (want {sorted(ACTIONS)})")
        if not self.connected:
            raise RuntimeError(f"{NAME} ({self.arm_id}): arm not connected ({self.connect_error})")
        s = self.safety
        assert s is not None
        if action == "stop":
            await s.stop("stop requested")
            result: Any = {"stopped": True}
        elif action == "reset":
            await s.reset()
            result = {"reset": True}
        else:
            call = self._prepare_cell(action, args)
            entry_gen = s.gen
            async with self.lock:
                if s.gen != entry_gen:
                    raise RuntimeError(f"motion aborted: {s.abort_reason or 'stopped'} (while queued)")
                if not self.connected:
                    raise RuntimeError(f"{NAME} ({self.arm_id}): arm disconnected while queued")
                self.busy, self.task_id = action, tid or None
                try:
                    gen = s.begin_command()
                    result = await call(gen, progress)
                except asyncio.CancelledError:
                    await s.emergency_stop("task cancelled")
                    raise
                except MotionAborted as e:
                    raise RuntimeError(f"motion aborted: {e}") from None
                finally:
                    self.busy, self.task_id = None, None
        if tid:
            self.done[tid] = result
            if len(self.done) > cell_arm.DONE_MAX:
                self.done.pop(next(iter(self.done)))
        return result

    def _prepare_cell(self, action: str, a: dict):
        if action == "home":
            speed = a.get("speed")
            if speed is not None:
                self.safety.cap_speed(speed)
            return lambda gen, progress: self._home(gen, progress, speed)
        # move_object: parse everything now, relocate later (with the arm held, right before moving)
        item = a.get("item")
        if not isinstance(item, str) or not item:
            raise ValueError("move_object needs item (catalog name)")
        exp = _xy(a.get("table_xy"), "table_xy")
        height = num(a.get("height"), "height")
        grasp = a.get("grasp")
        if grasp not in ("across", "center"):
            raise ValueError(f"move_object grasp must be across | center, got {grasp!r} (grasp none is never picked)")
        to = a.get("to")
        if not isinstance(to, dict):
            raise ValueError("move_object needs to: {table_xy: [x, y], yaw_deg, z}")
        dest = _xy(to.get("table_xy"), "to.table_xy")
        dest_z = num(to.get("z", height), "to.z")
        dest_yaw = None if to.get("yaw_deg") is None else num(to["yaw_deg"], "to.yaw_deg")
        exp_yaw = None if a.get("yaw_deg") is None else num(a["yaw_deg"], "yaw_deg")
        blob_id = a.get("blob_id")
        return lambda gen, progress: self._move_object(gen, progress, item, blob_id, exp, exp_yaw, height,
                                                       grasp, dest, dest_yaw, dest_z)

    # ---------------------------------------------------------------- move_object
    def latest_blobs(self) -> dict:
        """Newest cell.blobs message: the local perception's node.cell_blobs or the topic cache."""
        msgs = []
        local = getattr(self.node, "cell_blobs", None)
        if isinstance(local, dict):
            msgs.append((float(local.get("frame_ts") or 0.0), local))
        topics = getattr(self.node, "topics", None)
        sample = (getattr(topics, "latest", None) or {}).get("cell.blobs") if topics is not None else None
        if isinstance(sample, dict) and isinstance(sample.get("data"), dict):
            d = sample["data"]
            msgs.append((float(d.get("frame_ts") or sample.get("ts") or 0.0), d))
        if not msgs:
            raise RuntimeError(f"{NOT_WHERE_EXPECTED}: no cell.blobs received yet (is vision.cell running?)")
        ts, msg = max(msgs, key=lambda t: t[0])
        age = time.time() - ts
        if self.max_age > 0 and age > self.max_age:
            raise RuntimeError(f"cell.blobs is {age:.1f} s old (max {self.max_age:g} s): camera stalled; "
                               f"refusing to move without a fresh view")
        if msg.get("calibrated") is False:
            raise RuntimeError("cell.blobs is not calibrated (table tags not seen); refusing to move")
        return msg

    def plan(self, blob: dict, exp_yaw: Optional[float], height: float, grasp: str,
             dest: tuple[float, float], dest_yaw: Optional[float], dest_z: float,
             tool_yaw: Optional[float]) -> dict:
        """Arm-frame pick and place poses for a relocated blob (pure; no motion)."""
        bx, by = float(blob["table_xy"][0]), float(blob["table_xy"][1])
        b_yaw = blob.get("yaw_deg")
        b_yaw = float(b_yaw) if isinstance(b_yaw, (int, float)) else (exp_yaw or 0.0)
        px, py, pz = P.table_to_arm((bx, by, height), self.base)
        pick_yaw = fold_yaw(P.yaw_table_to_arm(grasp_yaw_table(b_yaw, grasp), self.base), tool_yaw)
        qx, qy, qz = P.table_to_arm((dest[0], dest[1], dest_z), self.base)
        # turn the part by (destination yaw - its current yaw); blob yaw is an axis (mod 180)
        turn = 0.0 if dest_yaw is None else wrap_deg(dest_yaw - b_yaw)
        place_yaw = fold_yaw(pick_yaw + turn, pick_yaw)
        return {"pick": {"x": px, "y": py, "z": pz, "yaw": pick_yaw},
                "place": {"x": qx, "y": qy, "z": qz, "yaw": place_yaw}}

    async def _move_object(self, gen, progress, item, blob_id, exp, exp_yaw, height, grasp,
                           dest, dest_yaw, dest_z) -> dict:
        s = self.safety
        await progress({"step": "relocate", "item": item, "blob_id": blob_id, "expected": list(exp)})
        msg = self.latest_blobs()
        blob, dist = relocate(msg.get("blobs") or [], exp, blob_id, self.radius)
        cur = await s.pose()
        p = self.plan(blob, exp_yaw, height, grasp, dest, dest_yaw, dest_z, cur.get("yaw"))
        pk, pl = p["pick"], p["place"]
        ah = s.approach_h
        carry_z = max(pk["z"], pl["z"]) + ah
        # validate every waypoint before anything moves (workspace, then the driver's IK if it has one)
        points = [(pk["x"], pk["y"], pk["z"], pk["yaw"]), (pk["x"], pk["y"], pk["z"] + ah, pk["yaw"]),
                  (pk["x"], pk["y"], carry_z, pk["yaw"]), (pl["x"], pl["y"], carry_z, pl["yaw"]),
                  (pl["x"], pl["y"], pl["z"] + ah, pl["yaw"]), (pl["x"], pl["y"], pl["z"], pl["yaw"])]
        for pt in points:
            s.check(*pt, allow_clamp=False)
        reach = getattr(s.drv, "check_reachable", None)
        if reach is not None:
            for pt in points:
                reach(*pt)
        found = {"blob_id": blob.get("id"), "table_xy": [round(float(v), 4) for v in blob["table_xy"][:2]],
                 "yaw_deg": blob.get("yaw_deg"), "moved_m": round(dist, 4)}
        await progress({"step": "relocated", "item": item, **found})

        def sub(phase: str, base_pct: int, span: int):
            async def f(pp: dict) -> None:
                await progress({**pp, "phase": phase, "item": item,
                                "pct": base_pct + round(span * float(pp.get("pct", 0)) / 100)})
            return f

        width = blob.get("width_m")                       # short side of the camera's bounding box
        grip = width_to_grip(float(width), self.grip_open_m, self.squeeze_m) if width else None
        if grip is not None:
            await progress({"step": "grip_width", "item": item, "width_cm": round(float(width) * 100, 1),
                            "gap_cm": round(grip * self.grip_open_m * 100, 1)})
        picked = await pick(s, pk["x"], pk["y"], pk["z"], pk["yaw"], ah, None, sub("pick", 5, 40), gen,
                            grip_value=grip)
        await progress({"step": "carry", "phase": "carry", "item": item, "pct": 50,
                        "target": {k: round(v, 4) for k, v in pl.items()}})
        if carry_z > pk["z"] + ah + 1e-4:
            await s.move_to(pk["x"], pk["y"], carry_z, pk["yaw"], s.slow_speed, gen=gen, allow_clamp=False)
        await s.move_to(pl["x"], pl["y"], carry_z, pl["yaw"], min(self.carry_speed, s.max_speed), gen=gen,
                        allow_clamp=False)
        placed = await place(s, pl["x"], pl["y"], pl["z"], pl["yaw"], carry_z - pl["z"], 0.0, None,
                             sub("place", 55, 40), gen)
        await progress({"step": "done", "item": item, "pct": 100})
        return {"moved": item, "arm_id": self.arm_id, "found": found,
                "pick": picked["picked"], "place": placed["placed"],
                "to": {"table_xy": list(dest), "z": dest_z, "yaw_deg": dest_yaw}}


# ====================================================================== module wiring
_SVC: Optional[CellArmService] = None


def get_service() -> Optional[CellArmService]:
    return _SVC


def _handler(svc: CellArmService):
    async def arm_cell(args: dict, progress) -> Any:
        return await svc.handle(args, progress)
    arm_cell._arm_cell = True  # type: ignore[attr-defined]
    return arm_cell


async def setup(node: "Node") -> None:
    global _SVC
    caps = list(getattr(node.cfg, "capabilities", []) or [])
    if NAME not in caps or _SVC is not None:
        return
    existing = REGISTRY.get(NAME)
    if existing is not None and getattr(existing, "_cell_arm", False):
        # cell_arm's generic "arm.<id>" loader got here first: arm.cell is ours, take it over
        old = cell_arm._ARMS.pop(NAME, None)
        if old is not None:
            await old.close()
    cc = getattr(node.cfg, "capability_config", {}) or {}
    conf = cc.get(NAME)
    if not isinstance(conf, dict):
        node.event(f"{NAME}: no capability_config entry; using the MOCK driver", level="warning")
        conf = {"driver": "mock"}
    try:
        svc = CellArmService(conf, node)
    except Exception as e:  # noqa: BLE001 - bad config: withdraw the capability, keep the node up
        set_availability(NAME, lambda: False)
        node.event(f"{NAME} CONFIG ERROR: {e}", level="error")
        return
    _SVC = svc
    register_capability(NAME)(_handler(svc))       # registered before cell_arm.setup runs: it skips us
    set_availability(NAME, svc.available)
    try:
        await svc.start()
    except Exception:  # noqa: BLE001
        log.exception("%s setup failed", NAME)


async def shutdown() -> None:
    global _SVC
    svc, _SVC = _SVC, None
    if svc is not None:
        try:
            await svc.close()
        except Exception:  # noqa: BLE001
            log.exception("%s shutdown failed", NAME)
    h = REGISTRY.get(NAME)
    if h is not None and getattr(h, "_arm_cell", False):
        REGISTRY.pop(NAME, None)
    AVAILABILITY.pop(NAME, None)
