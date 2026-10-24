"""Robot-cell arms: capability "arm.<id>" for every configured large arm (docs/ROBOT_CELL.md).

At setup, every capability on this node named "arm.<id>" (except "arm.so101", owned by
capabilities/so101.py, and "arm.mock", the builtin demo) gets a handler registered under that name,
driven by capability_config["arm.<id>"]:

    capability_config:
      arm.big:
        driver: mock | kinova | xarm | ur | so101       (default mock)
        workspace: {x: [0.15, 0.70], y: [-0.45, 0.45], z: [0.02, 0.60]}  (metres, arm base frame;
                                                         required for real drivers)
        max_speed: 0.10                                  m/s cap
        fallback_to_mock: false                          true: a real arm that fails to connect is
                                                         replaced by the mock (logged loudly)
        telemetry_hz: 10
        connect_timeout_s: 15
        ...motion keys (cell/arm.py) and driver keys (cell/drivers/<driver>.py)

Task args (actions in arm-base metres and degrees; `_task_id` makes them idempotent):
    {"action": "home"}
    {"action": "move_to", "x": .., "y": .., "z": .., "yaw": 0, "speed": 0.05}
    {"action": "gripper", "value": 0..1}             (1 = open, 0 = closed; or "open" / "close")
    {"action": "pick",  "x", "y", "z", "yaw", "approach_h", "speed"?}
    {"action": "place", "x", "y", "z", "yaw", "approach_h", "insert_depth", "speed"?}
    {"action": "stop"}                               (aborts the running and queued motions)
    {"action": "reset"}                              (clear a latched e-stop / driver faults)
    {"action": "named_pose", "name": ".."}           (so101 driver only)

Cancelling a running task (TASK_CANCEL -> asyncio cancellation) is an e-stop: the arm stops at once.
Motion tasks run one at a time per arm; a stop also aborts tasks still waiting for the arm.
Telemetry: topic "arm.<id>.state" at telemetry_hz:
    {pose: {x, y, z, yaw}, gripper, busy, action, driver, estop, estop_reason, fault, connected,
     fallback_from, workspace, max_speed, task_id}
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Optional

from ..cell.arm import MotionAborted, Safety, build_arm, get_driver, num, pick, place
from ..logutil import get_logger
from ..tasks import REGISTRY, register_capability

if TYPE_CHECKING:
    from ..agent import Node

log = get_logger("cell_arm")

SKIP = {"arm.so101", "arm.mock"}
ACTIONS = {"home", "move_to", "gripper", "pick", "place", "stop", "reset", "named_pose"}
DONE_MAX = 500


class ArmService:
    async def _mock_carry(self, obj: str, x: float, y: float, yaw: float) -> None:
        """Simulator only: after a mock place, move the carried part in the synthetic camera scene
        (topic cell.mock.poses, read by capabilities/cell_vision.py in source: mock) so the scene,
        and Gemma's verify step, reflect what the arm did. Real arms never publish this."""
        arm_cfg = ((self.node.cfg.capability_config.get("cell") or {}).get("arms") or {}).get(self.name)
        if not arm_cfg:
            return
        try:
            from ..cell import perception as P
            tx, ty, _ = P.arm_to_table((x, y, 0.0), arm_cfg)
            tyaw = P.yaw_arm_to_table(yaw, arm_cfg)
            await self.node.publish("cell.mock.poses", {obj: [round(float(tx), 4), round(float(ty), 4), round(float(tyaw), 1)]})
        except Exception as e:  # noqa: BLE001 - cosmetic for the simulator only
            log.warning("%s: mock carry of %s failed: %s", self.name, obj, e)

    def __init__(self, name: str, conf: dict, node: "Node"):
        self.name = name
        self.conf = dict(conf or {})
        self.node = node
        self.driver_name = str(self.conf.get("driver") or "mock")
        self.safety: Optional[Safety] = None
        self.connected = False
        self.connect_error: Optional[str] = None
        self.fallback_from: Optional[str] = None
        self.lock = asyncio.Lock()
        self.done: dict[str, Any] = {}
        self.busy: Optional[str] = None
        self.task_id: Optional[str] = None
        self.hz = max(0.5, min(50.0, float(self.conf.get("telemetry_hz", 10))))
        self.connect_timeout = float(self.conf.get("connect_timeout_s", 15))
        self._telemetry: Optional[asyncio.Task] = None
        self._connect_lock = asyncio.Lock()

    def _event(self, msg: str, level: str = "info") -> None:
        try:
            self.node.event(msg, level=level)
        except Exception:  # noqa: BLE001
            getattr(log, level if level in ("warning", "error") else "info")(msg)

    # ------------------------------------------------------------------ connect
    async def start(self) -> None:
        await self.connect()
        self._telemetry = asyncio.create_task(self._telemetry_loop())

    async def connect(self) -> None:
        async with self._connect_lock:
            if self.connected:
                return
            if self.safety is None:
                try:     # bad config is never papered over by the mock fallback
                    self.safety = build_arm(self.conf)
                except Exception as e:  # noqa: BLE001
                    self.connect_error = f"config: {type(e).__name__}: {e}"
                    self._event(f"{self.name} CONFIG ERROR: {e}", "error")
                    return
            safety = self.safety
            try:
                await asyncio.wait_for(safety.drv.connect(), self.connect_timeout)
                self.connected, self.connect_error = True, None
                self._event(f"{self.name} ready: driver {safety.drv.name}, workspace {safety.workspace.wire()}, "
                            f"max {safety.max_speed:g} m/s")
                return
            except Exception as e:  # noqa: BLE001 - ImportError of the SDK, network, auth...
                err = f"{type(e).__name__}: {e}"
                self.connect_error = err
            if self.driver_name != "mock" and bool(self.conf.get("fallback_to_mock", False)):
                mock_conf = {**self.conf, "driver": "mock"}
                drv = get_driver("mock", mock_conf)
                await drv.connect()
                self.safety = Safety(drv, mock_conf)
                self.connected, self.fallback_from = True, self.driver_name
                msg = (f"!!! {self.name}: {self.driver_name} arm FAILED to connect ({err}). "
                       f"fallback_to_mock is on: running on the MOCK arm. NOTHING PHYSICAL WILL MOVE. !!!")
                log.error(msg)
                self._event(msg, "error")
            else:
                self._event(f"{self.name}: {self.driver_name} arm failed to connect ({err}); tasks will "
                            f"retry the connection (set fallback_to_mock: true to simulate instead)", "error")

    async def close(self) -> None:
        if self._telemetry:
            self._telemetry.cancel()
            try:
                await self._telemetry
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._telemetry = None
        if self.safety is not None:
            if self.busy:
                await self.safety.stop("shutdown")
            try:
                await self.safety.drv.close()
            except Exception:  # noqa: BLE001
                log.exception("%s driver close failed", self.name)
        self.connected = False
        self.done.clear()

    # ------------------------------------------------------------------ telemetry
    async def state(self) -> dict:
        s = self.safety
        pose = await s.pose() if (s is not None and self.connected) else \
            {"x": None, "y": None, "z": None, "yaw": None}
        return {
            "pose": {k: (round(pose[k], 5) if isinstance(pose.get(k), float) else pose.get(k))
                     for k in ("x", "y", "z", "yaw")},
            "gripper": None if s is None or s.drv.grip is None else round(float(s.drv.grip), 3),
            "busy": self.busy is not None,
            "action": self.busy,
            "task_id": self.task_id,
            "driver": s.drv.name if s is not None else self.driver_name,
            "fallback_from": self.fallback_from,
            "connected": self.connected,
            "estop": bool(s and s.estop),
            "estop_reason": s.estop_reason if s else "",
            "fault": (s.fault if s else None) or (None if self.connected else self.connect_error),
            "target": s.target if s else None,
            "workspace": s.workspace.wire() if s else None,
            "max_speed": s.max_speed if s else None,
            "ts": time.time(),
        }

    async def _telemetry_loop(self) -> None:
        topic = f"{self.name}.state"
        while True:
            await asyncio.sleep(1.0 / self.hz)
            try:
                await self.node.publish(topic, await self.state())
            except Exception:  # noqa: BLE001
                log.exception("%s telemetry publish failed", self.name)

    # ------------------------------------------------------------------ tasks
    async def handle(self, args: dict, progress) -> Any:
        tid = str(args.get("_task_id") or "")
        if tid and tid in self.done:            # at-least-once delivery: never move twice
            return self.done[tid]
        action = args.get("action", "move_to")
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r} (want {sorted(ACTIONS)})")
        if not self.connected:
            await self.connect()
            if not self.connected:
                raise RuntimeError(f"{self.name}: {self.driver_name} arm not connected ({self.connect_error})")
        s = self.safety
        assert s is not None
        if action == "stop":
            await s.stop("stop requested")
            result: Any = {"stopped": True}
        elif action == "reset":
            await s.reset()
            result = {"reset": True}
        else:
            call = self._prepare(action, args)          # validate everything before queueing
            entry_gen = s.gen
            async with self.lock:
                if s.gen != entry_gen:
                    raise RuntimeError(f"motion aborted: {s.abort_reason or 'stopped'} (while queued)")
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
            if len(self.done) > DONE_MAX:
                self.done.pop(next(iter(self.done)))
        return result

    def _prepare(self, action: str, a: dict):
        """Parse + bounds-check args now (raises ValueError) and return the coroutine factory."""
        s = self.safety
        speed = a.get("speed")
        if speed is not None:
            s.cap_speed(speed)

        def xyz() -> tuple[float, float, float, float]:
            for k in ("x", "y", "z"):
                if k not in a:
                    raise ValueError(f"{action} needs x, y, z (metres, arm base frame); missing {k}")
            return num(a["x"], "x"), num(a["y"], "y"), num(a["z"], "z"), num(a.get("yaw", 0.0), "yaw")

        if action == "home":
            return lambda gen, progress: self._home(gen, progress, speed)
        if action == "move_to":
            x, y, z, yaw = xyz()
            s._require_cartesian()
            s.check(x, y, z, yaw)
            return lambda gen, progress: self._move(gen, progress, x, y, z, yaw, speed)
        if action == "gripper":
            v = a.get("value")
            if isinstance(v, str):
                v = {"open": 1.0, "opened": 1.0, "close": 0.0, "closed": 0.0}.get(v.lower(), v)
            v = num(v, "gripper value (0 closed .. 1 open)")
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"gripper value must be in 0..1, got {v}")
            return lambda gen, progress: self._grip(gen, progress, v)
        if action == "pick":
            x, y, z, yaw = xyz()
            ah = a.get("approach_h")
            return lambda gen, progress: pick(s, x, y, z, yaw, ah, speed, progress, gen)
        if action == "place":
            x, y, z, yaw = xyz()
            ah = a.get("approach_h")
            depth = a.get("insert_depth", 0.0)
            carry = a.get("carry")

            async def do_place(gen, progress):
                result = await place(s, x, y, z, yaw, ah, depth, speed, progress, gen)
                if carry and s.drv.name == "mock":
                    await self._mock_carry(str(carry), x, y, yaw)
                return result
            return do_place
        if action == "named_pose":
            name = a.get("name")
            if not hasattr(s.drv, "named_pose"):
                raise ValueError(f"driver {s.drv.name!r} has no named poses")
            if not isinstance(name, str) or not name:
                raise ValueError("named_pose needs name")
            return lambda gen, progress: self._named(gen, progress, name)
        raise ValueError(f"unknown action {action!r}")

    async def _home(self, gen, progress, speed) -> dict:
        await progress({"step": "home"})
        r = await self.safety.home(speed, gen=gen)
        return {"homed": True, **r}

    async def _move(self, gen, progress, x, y, z, yaw, speed) -> dict:
        await progress({"step": "move_to", "target": {"x": x, "y": y, "z": z, "yaw": yaw}})
        r = await self.safety.move_to(x, y, z, yaw, speed, gen=gen)
        return {"reached": {k: r[k] for k in ("x", "y", "z", "yaw")}, "clamped": r["clamped"], "speed": r["speed"]}

    async def _grip(self, gen, progress, v) -> dict:
        await progress({"step": "gripper", "value": v})
        await self.safety.gripper(v, gen=gen)
        return {"gripper": v}

    async def _named(self, gen, progress, name) -> dict:
        await progress({"step": "named_pose", "name": name})
        self.safety.check_gen(gen)
        try:
            r = await self.safety.drv.named_pose(name)
        except Exception:
            self.safety.check_gen(gen)
            raise
        self.safety.check_gen(gen)
        return {"pose": name, "result": r}


# ====================================================================== module wiring
_ARMS: dict[str, ArmService] = {}


def get_arm(name: str) -> Optional[ArmService]:
    return _ARMS.get(name)


def arm_names(node: "Node") -> list[str]:
    caps = list(getattr(node.cfg, "capabilities", []) or [])
    return [c for c in caps if isinstance(c, str) and c.startswith("arm.") and c not in SKIP]


def _make_handler(svc: ArmService):
    async def handler(args: dict, progress) -> Any:
        return await svc.handle(args, progress)
    handler.__name__ = f"cell_arm_{svc.name.replace('.', '_')}"
    handler._cell_arm = True  # type: ignore[attr-defined]
    return handler


async def setup(node: "Node") -> None:
    cc = getattr(node.cfg, "capability_config", {}) or {}
    for name in arm_names(node):
        if name in _ARMS:
            continue
        existing = REGISTRY.get(name)
        if existing is not None and not getattr(existing, "_cell_arm", False):
            log.info("%s already has a handler (%s); cell_arm leaves it alone", name, existing.__module__)
            continue
        conf = cc.get(name)
        if not isinstance(conf, dict):
            node.event(f"{name}: no capability_config entry; using the MOCK driver", level="warning")
            conf = {"driver": "mock"}
        svc = ArmService(name, conf, node)
        _ARMS[name] = svc
        register_capability(name)(_make_handler(svc))
        try:
            await svc.start()
        except Exception:  # noqa: BLE001 - one bad arm must not stop the others
            log.exception("%s setup failed", name)


async def shutdown() -> None:
    """Stop telemetry, stop any motion, close drivers, unregister handlers."""
    for name, svc in list(_ARMS.items()):
        try:
            await svc.close()
        except Exception:  # noqa: BLE001
            log.exception("%s shutdown failed", name)
        h = REGISTRY.get(name)
        if h is not None and getattr(h, "_cell_arm", False):
            REGISTRY.pop(name, None)
    _ARMS.clear()
