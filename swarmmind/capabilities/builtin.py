"""Built-in demo capabilities: `echo` (smoke test) and `arm.mock` (a simulated robot arm that
shows the shape a real arm handler should have: validate, bounds-check, report progress, be
idempotent by task id, stop on cancel)."""
from __future__ import annotations

import asyncio

from ..tasks import register_capability

_done: dict[str, dict] = {}     # idempotency cache: task_id -> result

# Workspace bounds in metres (a real arm module would load these from its own config).
BOUNDS = {"x": (0.15, 0.70), "y": (-0.45, 0.45), "z": (0.02, 0.60)}


@register_capability("echo")
async def echo(args: dict, progress) -> dict:
    return {"echo": {k: v for k, v in args.items() if not k.startswith("_")}}


@register_capability("arm.mock")
async def mock_arm(args: dict, progress) -> dict:
    tid = args.get("_task_id", "")
    if tid in _done:                       # at-least-once delivery: never move twice
        return _done[tid]
    action = args.get("action", "move_to")
    if action == "home":
        target = {"x": 0.40, "y": 0.0, "z": 0.40}
    elif action == "move_to":
        target = {k: float(args[k]) for k in ("x", "y", "z")}
    else:
        raise ValueError(f"unknown action {action!r} (want move_to | home)")
    for axis, (lo, hi) in BOUNDS.items():
        if not lo <= target[axis] <= hi:
            raise ValueError(f"{axis}={target[axis]} outside safe bounds [{lo}, {hi}]")
    steps = int(args.get("steps", 5))
    for i in range(1, steps + 1):
        await asyncio.sleep(float(args.get("step_s", 0.3)))   # cancellation point = e-stop
        await progress({"step": "moving", "pct": round(100 * i / steps), "target": target})
    result = {"reached": target}
    _done[tid] = result
    return result
