"""Generic capability task bus — the extension point for non-LLM node types (robot arms, sensors).

A node advertises capabilities in its heartbeat. Handlers live in swarmmind/capabilities/*.py:

    from swarmmind.tasks import register_capability

    @register_capability("arm.kinova")
    async def pick(args: dict, progress) -> dict:
        await progress({"step": "moving", "pct": 30})
        return {"ok": True}

Flow: POST /task (any node) -> leader assigns to the best alive capable node -> task.assign ->
assignee runs the handler, emitting task.progress / task.result. If the assignee dies before
reporting, the leader reassigns with attempt+1 (at-least-once: make handlers idempotent by task_id).
"""
from __future__ import annotations

import asyncio
import importlib
import pkgutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .logutil import get_logger
from .protocol import TASK_ASSIGN, TASK_CANCEL, TASK_PROGRESS, TASK_RESULT

if TYPE_CHECKING:
    from .agent import Node

log = get_logger("tasks")

Progress = Callable[[dict], Awaitable[None]]
Handler = Callable[[dict, Progress], Awaitable[Any]]
REGISTRY: dict[str, Handler] = {}
MAX_ATTEMPTS = 3


def register_capability(name: str) -> Callable[[Handler], Handler]:
    def deco(fn: Handler) -> Handler:
        REGISTRY[name] = fn
        return fn
    return deco


def load_capability_modules() -> list:
    """Import every swarmmind/capabilities/*.py. A module may define `async def setup(node)`;
    the agent calls it at startup (open serial ports, subscribe to topics, start telemetry)."""
    from . import capabilities
    loaded = []
    for m in pkgutil.iter_modules(capabilities.__path__):
        try:
            loaded.append(importlib.import_module(f"{capabilities.__name__}.{m.name}"))
        except Exception as e:  # noqa: BLE001 - a broken plugin must not take the node down
            log.warning("capability module %s failed to import: %s", m.name, e)
    return loaded


@dataclass
class Task:
    task_id: str
    capability: str
    args: dict
    state: str = "queued"           # queued | assigned | running | done | failed | cancelled
    assignee: str | None = None
    attempt: int = 0
    result: Any = None
    error: str | None = None
    progress: dict = field(default_factory=dict)
    created_ts: float = 0.0
    updated_ts: float = 0.0

    def wire(self) -> dict:
        return asdict(self)

    @property
    def finished(self) -> bool:
        return self.state in ("done", "failed", "cancelled")


class TaskManager:
    def __init__(self, node: "Node"):
        self.node = node
        self.tasks: dict[str, Task] = {}
        self.running: dict[str, asyncio.Task] = {}      # local executions: task_id -> asyncio task

    def snapshot(self) -> list[dict]:
        return [t.wire() for t in sorted(self.tasks.values(), key=lambda t: t.created_ts)[-50:]]

    def _push(self, kind: str, task: Task) -> None:
        self.node.ui_push(kind, task.wire())

    # ------------------------------------------------------------------ leader side
    def submit(self, capability: str, args: dict, task_id: str | None = None) -> Task:
        """Leader only. Creates (or re-finds, for idempotent resubmits) and assigns a task."""
        tid = task_id or uuid.uuid4().hex[:8]
        if tid in self.tasks:
            return self.tasks[tid]
        now = time.time()
        task = Task(tid, capability, dict(args or {}), created_ts=now, updated_ts=now)
        self.tasks[tid] = task
        self.node.event(f"task {tid} submitted: {capability}")
        self._assign(task)
        return task

    def _candidates(self, capability: str) -> list[tuple[int, int, str]]:
        n = self.node
        busy: dict[str, int] = {}
        for t in self.tasks.values():
            if not t.finished and t.assignee:
                busy[t.assignee] = busy.get(t.assignee, 0) + 1
        out = []
        if capability in n.capabilities():
            out.append((n.cfg.rank, -busy.get(n.cfg.node_id, 0), n.cfg.node_id))
        for nid, ps in n.election.alive_peers().items():
            # LLM health only matters for llm.* work; an arm host with no LLM is still a valid executor
            if capability in ps.hb.capabilities and (ps.hb.healthy or not capability.startswith("llm.")):
                out.append((ps.hb.rank, -busy.get(nid, 0), nid))
        return sorted(out, reverse=True)

    def _assign(self, task: Task, exclude: str | None = None) -> None:
        cands = [c for c in self._candidates(task.capability) if c[2] != exclude] or \
            self._candidates(task.capability)
        if not cands:
            task.state, task.assignee = "queued", None
            self._push("task.queued", task)
            return
        if task.attempt >= MAX_ATTEMPTS:
            task.state, task.error = "failed", f"gave up after {task.attempt} attempts"
            task.updated_ts = time.time()
            self.node.broadcast(TASK_RESULT, task.wire())
            self._push(TASK_RESULT, task)
            return
        task.attempt += 1
        task.assignee = cands[0][2]
        task.state = "assigned"
        task.updated_ts = time.time()
        self.node.broadcast(TASK_ASSIGN, task.wire())
        self._push(TASK_ASSIGN, task)
        self.on_assign(task.wire())         # in case we assigned it to ourselves

    def cancel(self, task_id: str) -> bool:
        task = self.tasks.get(task_id)
        if not task or task.finished:
            return False
        task.state, task.updated_ts = "cancelled", time.time()
        self.node.broadcast(TASK_CANCEL, task.wire())
        self._push(TASK_CANCEL, task)
        self._stop_local(task_id)
        return True

    def watchdog(self) -> None:
        """Leader tick: assign queued tasks, reassign tasks whose assignee died."""
        n = self.node
        for task in list(self.tasks.values()):
            if task.finished:
                continue
            if task.state == "queued":
                self._assign(task)
            elif task.assignee and not n.election.is_alive(task.assignee):
                n.event(f"task {task.task_id}: assignee {task.assignee} died; reassigning", level="warning")
                self._assign(task, exclude=task.assignee)

    # ------------------------------------------------------------------ every node
    def on_assign(self, body: dict) -> None:
        task = self._merge(body)
        if task.assignee != self.node.cfg.node_id or task.finished:
            return
        key = task.task_id
        if key in self.running and not self.running[key].done():
            return
        handler = REGISTRY.get(task.capability)
        if handler is None or task.capability not in self.node.capabilities():
            self._report(task, error=f"{self.node.cfg.node_id} has no handler for {task.capability}")
            return
        self.running[key] = asyncio.create_task(self._execute(task, handler, task.attempt))

    async def _execute(self, task: Task, handler: Handler, attempt: int) -> None:
        n = self.node
        task.state = "running"

        async def progress(p: dict) -> None:
            task.progress = dict(p)
            task.updated_ts = time.time()
            n.broadcast(TASK_PROGRESS, task.wire())
            self._push(TASK_PROGRESS, task)

        await progress({"step": "started", "node": n.cfg.node_id, "attempt": attempt})
        try:
            result = await handler({**task.args, "_task_id": task.task_id}, progress)
            self._report(task, result=result)
        except asyncio.CancelledError:
            log.info("task %s cancelled locally", task.task_id)
            raise
        except (ValueError, PermissionError, RuntimeError) as e:   # expected refusals: one line
            log.warning("task %s refused: %s", task.task_id, e)
            self._report(task, error=f"{type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            log.exception("task %s handler failed", task.task_id)
            self._report(task, error=f"{type(e).__name__}: {e}")
        finally:
            self.running.pop(task.task_id, None)

    def _report(self, task: Task, result: Any = None, error: str | None = None) -> None:
        task.state = "failed" if error else "done"
        task.result, task.error, task.updated_ts = result, error, time.time()
        self.node.broadcast(TASK_RESULT, task.wire())
        self._push(TASK_RESULT, task)
        self.node.event(f"task {task.task_id} {task.state} on {self.node.cfg.node_id}"
                        + (f": {error}" if error else ""))

    def on_progress(self, body: dict) -> None:
        task = self._merge(body)
        self._push(TASK_PROGRESS, task)

    def on_result(self, body: dict) -> None:
        task = self.tasks.get(body["task_id"])
        if task and task.attempt > int(body.get("attempt", 0)):
            return                           # stale result from a superseded attempt
        task = self._merge(body)
        self._push(TASK_RESULT, task)

    def on_cancel(self, body: dict) -> None:
        task = self._merge(body)
        self._stop_local(task.task_id)
        self._push(TASK_CANCEL, task)

    def _stop_local(self, task_id: str) -> None:
        t = self.running.pop(task_id, None)
        if t and not t.done():
            t.cancel()

    def _merge(self, body: dict) -> Task:
        tid = body["task_id"]
        task = self.tasks.get(tid)
        fields = Task.__dataclass_fields__
        incoming = Task(**{k: v for k, v in body.items() if k in fields})
        if task is None or incoming.updated_ts >= task.updated_ts or incoming.attempt > task.attempt:
            self.tasks[tid] = incoming
            return incoming
        return task

    def merge_sync(self, tasks: list[dict]) -> None:
        for body in tasks:
            self._merge(body)
