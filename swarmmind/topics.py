"""Pub/sub topics: high-rate, lossy, latest-value-wins streams (phone IMU -> arm teleop,
arm joint telemetry, sensor readings). Not leader-gated, not persisted, not synced.

Topic names are dotted: "imu.phone1", "teleop.so101", "arm.so101.state".
Each sample: {topic, data, origin, seq, ts}. Receivers drop samples whose seq is not newer
than the last one seen for (origin, topic), so reordering never moves an arm backwards.
In-process consumers (capability modules) use:

    from swarmmind.topics import subscribe
    subscribe("teleop.so101", on_sample)      # async def on_sample(sample: dict)
"""
from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

from .logutil import get_logger

log = get_logger("topics")

Callback = Callable[[dict], Awaitable[None]]
_SUBSCRIBERS: list[tuple[str, Callback]] = []


def subscribe(prefix: str, cb: Callback) -> None:
    """Register an in-process consumer for every topic starting with `prefix`."""
    _SUBSCRIBERS.append((prefix, cb))


class TopicBus:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.latest: dict[str, dict] = {}                # topic -> newest sample
        self._last_seq: dict[tuple[str, str], int] = {}  # (origin, topic) -> seq
        self._prev = 0
        self.ws_clients: dict[object, tuple[str, ...]] = {}   # websocket -> subscribed prefixes

    def make_sample(self, topic: str, data) -> dict:
        # seq is microseconds-since-epoch (bumped if needed) so it stays increasing across restarts
        self._prev = max(self._prev + 1, time.time_ns() // 1000)
        return {"topic": topic, "data": data, "origin": self.node_id,
                "seq": self._prev, "ts": time.time()}

    def accept(self, sample: dict) -> bool:
        """Record a sample (local or remote). False if stale/duplicate."""
        key = (sample.get("origin", ""), sample["topic"])
        seq = int(sample.get("seq", 0))
        if seq <= self._last_seq.get(key, 0):
            return False
        self._last_seq[key] = seq
        self.latest[sample["topic"]] = sample
        return True

    async def deliver(self, sample: dict) -> None:
        topic = sample["topic"]
        for prefix, cb in list(_SUBSCRIBERS):
            if topic.startswith(prefix):
                try:
                    await cb(sample)
                except Exception:  # noqa: BLE001 - one bad consumer must not stop the stream
                    log.exception("topic subscriber %s failed on %s", prefix, topic)
        dead = []
        for ws, prefixes in list(self.ws_clients.items()):
            if any(topic.startswith(p) for p in prefixes):
                try:
                    await asyncio.wait_for(ws.send_json(sample), 0.5)
                except Exception:  # noqa: BLE001
                    dead.append(ws)
        for ws in dead:
            self.ws_clients.pop(ws, None)
