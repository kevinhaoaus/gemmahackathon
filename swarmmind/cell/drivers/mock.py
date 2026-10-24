"""Simulated Cartesian arm: no hardware, realistic timing.

The tool moves along a straight line with a trapezoidal velocity profile (accelerate, cruise at
min(requested speed, mock_max_speed), decelerate), and yaw turns at mock_yaw_speed. The pose is
updated every tick, so telemetry shows the motion. stop() halts at the current point.

Config (besides the common keys in cell/arm.py):
    mock_max_speed: 0.25        m/s, the simulated arm's own top speed
    mock_accel: 0.5             m/s^2
    mock_yaw_speed: 90          deg/s
    mock_tick_s: 0.02           simulation step
    mock_latency_s: 0.02        per-command overhead (network + planning)
    mock_gripper_s: 0.6         full open <-> closed stroke time
    mock_start: {x, y, z, yaw}  initial pose (default: the home pose)
    mock_fail_connect: false    true: connect() raises (to exercise fallback paths)
"""
from __future__ import annotations

import asyncio
import math
import time

from ..arm import Arm, wrap_deg


class MockArm(Arm):
    name = "mock"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        c = self.cfg
        self.max_speed = float(c.get("mock_max_speed", 0.25))
        self.accel = float(c.get("mock_accel", 0.5))
        self.yaw_speed = float(c.get("mock_yaw_speed", 90.0))
        self.tick_s = float(c.get("mock_tick_s", 0.02))
        self.latency_s = float(c.get("mock_latency_s", 0.02))
        self.gripper_s = float(c.get("mock_gripper_s", 0.6))
        start = c.get("mock_start") or self.home_target()
        self.p = {"x": float(start["x"]), "y": float(start["y"]), "z": float(start["z"]),
                  "yaw": float(start.get("yaw", 0.0))}
        self.grip = 1.0
        self.connected = False
        self.moving = False
        self._stop_gen = 0
        self.log: list[dict] = []          # every command, for tests and debugging

    async def connect(self) -> None:
        if self.cfg.get("mock_fail_connect"):
            raise ConnectionError("mock_fail_connect is set")
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    async def stop(self) -> None:
        self._stop_gen += 1
        self.log.append({"cmd": "stop", "pose": dict(self.p)})

    async def pose(self) -> dict:
        return {**self.p, "gripper": self.grip}

    async def move_to(self, x: float, y: float, z: float, yaw_deg: float, speed: float) -> None:
        gen = self._stop_gen
        self.log.append({"cmd": "move_to", "x": x, "y": y, "z": z, "yaw": yaw_deg, "speed": speed})
        await asyncio.sleep(self.latency_s)
        if gen != self._stop_gen:
            return
        v = max(1e-6, min(float(speed), self.max_speed))
        a = max(1e-6, self.accel)
        s0 = dict(self.p)
        d = math.dist((s0["x"], s0["y"], s0["z"]), (x, y, z))
        dyaw = wrap_deg(yaw_deg - s0["yaw"])
        # trapezoid (or triangle if too short to reach v)
        t_acc = v / a
        if d >= v * t_acc:
            t_lin = d / v + t_acc
        else:
            t_acc = math.sqrt(d / a)
            v = a * t_acc
            t_lin = 2 * t_acc
        t_yaw = abs(dyaw) / max(1e-6, self.yaw_speed)
        total = max(t_lin, t_yaw)

        def frac(t: float) -> float:
            if d <= 1e-9:
                return 1.0
            if t <= 0:
                return 0.0
            if t >= t_lin:
                return 1.0
            if t < t_acc:
                s = 0.5 * a * t * t
            elif t < t_lin - t_acc:
                s = 0.5 * a * t_acc * t_acc + v * (t - t_acc)
            else:
                r = t_lin - t
                s = d - 0.5 * a * r * r
            return min(1.0, s / d)

        t0 = time.monotonic()
        self.moving = True
        try:
            while True:
                t = time.monotonic() - t0
                f = frac(t)
                fy = 1.0 if total <= 0 or t_yaw <= 0 else min(1.0, t / t_yaw)
                self.p = {"x": s0["x"] + (x - s0["x"]) * f, "y": s0["y"] + (y - s0["y"]) * f,
                          "z": s0["z"] + (z - s0["z"]) * f, "yaw": wrap_deg(s0["yaw"] + dyaw * fy)}
                if t >= total:
                    self.p = {"x": x, "y": y, "z": z, "yaw": wrap_deg(yaw_deg)}
                    return
                await asyncio.sleep(self.tick_s)
                if gen != self._stop_gen:
                    return                      # halted where it is; Safety reports the abort
        finally:
            self.moving = False

    async def gripper(self, value: float) -> None:
        gen = self._stop_gen
        self.log.append({"cmd": "gripper", "value": value})
        start = 1.0 if self.grip is None else self.grip
        dur = abs(value - start) * self.gripper_s
        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            f = 1.0 if dur <= 0 else min(1.0, t / dur)
            self.grip = start + (value - start) * f
            if f >= 1.0:
                self.grip = float(value)
                return
            await asyncio.sleep(self.tick_s)
            if gen != self._stop_gen:
                return

    def info(self) -> dict:
        return {"driver": self.name, "max_speed": self.max_speed}
