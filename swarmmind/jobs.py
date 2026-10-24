"""Jobs: replicated job state, token streaming, and mid-sentence handoff.

Every node keeps the full state of every job (question, vision context, frame, tokens).
Only the leader generates. When leadership moves while a job is unfinished, the new
leader continues the SAME answer by raw-prompting its model with the tokens so far.
"""
from __future__ import annotations

import asyncio
import base64
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from .logutil import get_logger
from .protocol import (JOB_DONE, JOB_HANDOFF, JOB_NEW, JOB_TOKEN, REPORT_NEW, SYSTEM_PROMPT)

if TYPE_CHECKING:
    from .agent import Node

log = get_logger("jobs")

MAX_JOBS = 50
VISION_UNAVAILABLE = "[vision unavailable]"
# What a successor that cannot see the frame is told. Phrased so the model keeps going instead of
# announcing it has no vision (which contradicts the first half of the answer).
RESUME_NO_VISION = ("[The camera frame was already described earlier in this answer by another unit; "
                    "it is not attached here. Continue the answer consistently.]")


@dataclass
class Job:
    job_id: str
    question: str
    model: str
    vision_mode: str
    vision_text: str | None
    image_b64: str | None
    state: str = "streaming"                     # streaming | done
    node_id: str = ""                            # node currently generating
    term: int = 0
    tokens: dict[int, tuple[str, str]] = field(default_factory=dict)   # seq -> (token, node_id)
    handoffs: list[dict] = field(default_factory=list)
    started_ts: float = 0.0
    done_ts: float | None = None
    final_seq: int | None = None
    nodes_used: list[str] = field(default_factory=list)
    last_token_mono: float | None = None         # local clock: when we last got a token
    kind: str = "ask"                            # ask | plan | verify (robot cell planner)
    system: str | None = None                    # system prompt; None = SYSTEM_PROMPT (old nodes)
    context: str | None = None                   # extra user-turn text, e.g. the cell's object list

    @property
    def has_image(self) -> bool:
        return bool(self.image_b64)

    def contiguous(self) -> int:
        n = 0
        while n in self.tokens:
            n += 1
        return n

    def text(self) -> str:
        return "".join(self.tokens[i][0] for i in range(self.contiguous()))

    def add_token(self, seq: int, token: str, node_id: str) -> bool:
        if seq in self.tokens:
            return False                         # dedupe by (job_id, seq)
        self.tokens[seq] = (token, node_id)
        self.last_token_mono = time.monotonic()
        if node_id not in self.nodes_used:
            self.nodes_used.append(node_id)
        return True

    def apply_handoff(self, h: dict) -> bool:
        key = (h["at_seq"], h["to_node"])
        if any((x["at_seq"], x["to_node"]) == key for x in self.handoffs):
            return False
        # Tokens at/after at_seq came from the dying leader and may not match the new stream.
        for s in [s for s in self.tokens if s >= h["at_seq"]]:
            del self.tokens[s]
        self.handoffs.append(h)
        self.node_id = h["to_node"]
        if h["to_node"] not in self.nodes_used:
            self.nodes_used.append(h["to_node"])
        return True

    # ---- wire forms
    def meta(self) -> dict[str, Any]:
        return {"job_id": self.job_id, "question": self.question, "model": self.model,
                "vision_mode": self.vision_mode, "vision_text": self.vision_text,
                "has_image": self.has_image, "state": self.state, "node_id": self.node_id,
                "term": self.term, "handoffs": self.handoffs, "started_ts": self.started_ts,
                "done_ts": self.done_ts, "final_seq": self.final_seq, "nodes_used": self.nodes_used,
                "kind": self.kind, "system": self.system, "context": self.context}

    def ui(self) -> dict[str, Any]:
        d = self.meta()
        d["tokens"] = [{"seq": s, "token": t, "node_id": n} for s, (t, n) in sorted(self.tokens.items())]
        return d

    def full(self) -> dict[str, Any]:
        d = self.ui()
        d["image_b64"] = self.image_b64
        return d

    @classmethod
    def from_wire(cls, d: dict[str, Any]) -> "Job":
        job = cls(job_id=d["job_id"], question=d["question"], model=d.get("model", ""),
                  vision_mode=d.get("vision_mode", "none"), vision_text=d.get("vision_text"),
                  image_b64=d.get("image_b64"), state=d.get("state", "streaming"),
                  node_id=d.get("node_id", ""), term=int(d.get("term", 0)),
                  handoffs=list(d.get("handoffs") or []), started_ts=float(d.get("started_ts") or 0),
                  done_ts=d.get("done_ts"), final_seq=d.get("final_seq"),
                  nodes_used=list(d.get("nodes_used") or []), kind=d.get("kind") or "ask",
                  system=d.get("system"), context=d.get("context"))
        for t in d.get("tokens") or []:
            job.tokens[int(t["seq"])] = (t["token"], t.get("node_id", ""))
        return job


class JobManager:
    def __init__(self, node: "Node"):
        self.node = node
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.streaming: dict[str, asyncio.Task] = {}   # job_id -> generation task (leader only)

    # ------------------------------------------------------------------ queries
    def active_job_id(self) -> str | None:
        for jid, t in self.streaming.items():
            if not t.done():
                return jid
        return None

    def has_unfinished(self) -> bool:
        return any(j.state != "done" for j in self.jobs.values())

    def snapshot(self) -> list[dict]:
        return [self.jobs[j].ui() for j in self.order[-20:]]

    def sync_payload(self) -> list[dict]:
        return [self.jobs[j].full() for j in self.order[-20:]]

    def _store(self, job: Job) -> None:
        if job.job_id not in self.jobs:
            self.order.append(job.job_id)
        self.jobs[job.job_id] = job
        while len(self.order) > MAX_JOBS:
            old = self.order.pop(0)
            if self.jobs.get(old) and self.jobs[old].state == "done":
                self.jobs.pop(old, None)
            else:
                self.order.insert(0, old)
                break

    # ------------------------------------------------------------------ leader: new job
    async def create(self, question: str, kind: str = "ask", system: str | None = None,
                     image: bytes | None = None, context: str | None = None,
                     job_id: str | None = None) -> Job:
        """Start a job. kind/system/context are replicated so a successor resumes with the same
        prompt; `image` (e.g. the annotated cell frame) replaces the raw camera frame; a given
        `job_id` makes creation idempotent (the planner derives it from plan/step/attempt)."""
        n = self.node
        if job_id and job_id in self.jobs:
            return self.jobs[job_id]
        frame = image if image is not None else await n.camera_frame()
        image_b64 = base64.b64encode(frame).decode() if frame else None
        vision_text: str | None = None
        if frame:
            if n.backend.supports_images:
                vision_mode = "native"
                if n.vision.has_detector:   # bonus: labels help a text-only successor
                    vision_text = await n.vision.detect_text(frame)
            elif n.vision.has_detector:
                vision_mode = n.vision.mode
                vision_text = await n.vision.detect_text(frame)
            else:
                vision_mode = "none"
                vision_text = VISION_UNAVAILABLE
        else:
            vision_mode = "none"
        job = Job(job_id=job_id or uuid.uuid4().hex[:8], question=question, model=n.backend.model,
                  vision_mode=vision_mode, vision_text=vision_text, image_b64=image_b64,
                  node_id=n.cfg.node_id, term=n.election.term, started_ts=time.time(),
                  nodes_used=[n.cfg.node_id], kind=kind, system=system or SYSTEM_PROMPT,
                  context=context)
        self._store(job)
        n.broadcast(JOB_NEW, job.full())
        n.ui_push(JOB_NEW, job.meta())
        n.event(f"{'job' if kind == 'ask' else kind + ' job'} {job.job_id} started on {n.cfg.node_id} "
                f"({job.model}, vision={vision_mode}): {question}")
        self.streaming[job.job_id] = asyncio.create_task(self._generate(job, resume=False))
        return job

    # ------------------------------------------------------------------ leader: resume
    def resume_unfinished(self) -> None:
        for job in list(self.jobs.values()):
            if job.state != "done" and job.job_id not in self.streaming:
                self.streaming[job.job_id] = asyncio.create_task(self._generate(job, resume=True))

    def cancel_all(self, why: str) -> None:
        for jid, t in list(self.streaming.items()):
            if not t.done():
                log.info("cancelling generation of %s: %s", jid, why)
                t.cancel()
        self.streaming.clear()

    async def _vision_for_resume(self, job: Job) -> tuple[str | None, str | None, str]:
        """Vision context for the successor. Returns (text, image_b64_to_attach, note)."""
        n = self.node
        labels = job.vision_text if job.vision_text != VISION_UNAVAILABLE else None
        if not job.has_image:
            return labels, None, "no image"
        if n.backend.supports_images:
            return labels, job.image_b64, "native: re-attached frame"
        if labels:
            return labels, None, "reused labels"
        if n.vision.has_detector:
            detected = await n.vision.detect_text(base64.b64decode(job.image_b64))
            if detected:
                return detected, None, f"re-ran {n.vision.mode} detector"
        return RESUME_NO_VISION, None, "vision unavailable"

    @staticmethod
    def _user_turn(question: str, vision_text: str | None, context: str | None = None) -> str:
        return "\n\n".join(p for p in (vision_text, context, question) if p)

    async def _generate(self, job: Job, resume: bool) -> None:
        n = self.node
        me = n.cfg.node_id
        term = n.election.term
        try:
            if resume:
                at_seq = job.contiguous()
                prefix = job.text()
                vision_text, image, note = await self._vision_for_resume(job)
                from_node = job.node_id or "?"
                ref = job.last_token_mono or n.leader_since
                log.info("resuming job %s at seq %d (%s); prefix=%r", job.job_id, at_seq, note, prefix[-60:])
                budget = max(32, n.cfg.num_predict - at_seq)    # same total length as an uninterrupted answer
                stream = n.backend.stream_continue(job.system or SYSTEM_PROMPT,
                                                   self._user_turn(job.question, vision_text, job.context),
                                                   prefix, image_b64=image, max_tokens=budget)
            else:
                at_seq = 0
                use_image = job.image_b64 if (n.backend.supports_images and job.vision_mode == "native") else None
                stream = n.backend.stream_chat(job.system or SYSTEM_PROMPT,
                                               self._user_turn(job.question, job.vision_text, job.context),
                                               use_image)
            seq = at_seq
            first = True
            async for tok in stream:
                if not n.election.is_leader or n.election.term != term:
                    log.info("lost leadership mid-stream of %s; stopping", job.job_id)
                    return
                if first and resume:
                    gap_ms = round((time.monotonic() - ref) * 1000)
                    h = {"job_id": job.job_id, "from_node": from_node, "to_node": me, "at_seq": at_seq,
                         "gap_ms": gap_ms, "vision_note": note, "term": term}
                    job.apply_handoff(h)
                    n.broadcast(JOB_HANDOFF, h)
                    n.ui_push(JOB_HANDOFF, h)
                    n.event(f"handoff {job.job_id}: {from_node} -> {me} at seq {at_seq}, gap {gap_ms} ms ({note})")
                first = False
                if not tok:
                    continue
                job.add_token(seq, tok, me)
                job.node_id = me
                body = {"job_id": job.job_id, "seq": seq, "token": tok, "node_id": me, "term": term}
                n.broadcast(JOB_TOKEN, body)
                n.ui_push(JOB_TOKEN, body)
                seq += 1
            if not n.election.is_leader or n.election.term != term:
                return
            if first and resume:
                # Model produced nothing new (prefix was already complete): still record handoff.
                h = {"job_id": job.job_id, "from_node": from_node, "to_node": me, "at_seq": at_seq,
                     "gap_ms": round((time.monotonic() - ref) * 1000), "vision_note": note, "term": term}
                job.apply_handoff(h)
                n.broadcast(JOB_HANDOFF, h)
                n.ui_push(JOB_HANDOFF, h)
            self._finish(job, seq)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("generation failed for %s", job.job_id)
            n.event(f"generation error on {me} for {job.job_id}: {e}", level="error")
        finally:
            if self.streaming.get(job.job_id) is asyncio.current_task():
                self.streaming.pop(job.job_id, None)

    def _finish(self, job: Job, final_seq: int) -> None:
        n = self.node
        job.state = "done"
        job.final_seq = final_seq
        job.done_ts = time.time()
        answer = job.text()
        body = {"job_id": job.job_id, "final_seq": final_seq, "answer": answer, "term": n.election.term,
                "nodes_used": job.nodes_used, "done_ts": job.done_ts, "kind": job.kind}
        n.broadcast(JOB_DONE, body)
        n.ui_push(JOB_DONE, body)
        if job.kind != "ask":          # plan/verify output is consumed by the planner, not uplinked
            n.event(f"{job.kind} job {job.job_id} done: {final_seq} tokens, nodes {job.nodes_used}, "
                    f"{len(job.handoffs)} handoff(s)")
            return
        report = {"job_id": job.job_id, "question": job.question, "answer": answer,
                  "nodes_used": job.nodes_used, "handoffs": job.handoffs,
                  "vision": job.vision_text or job.vision_mode, "timestamp": job.done_ts}
        if n.uplink.add_report(report):
            n.broadcast(REPORT_NEW, report)
            n.ui_push(REPORT_NEW, {**report, "sent": False})
        n.event(f"job {job.job_id} done: {final_seq} tokens, nodes {job.nodes_used}, "
                f"{len(job.handoffs)} handoff(s)")

    # ------------------------------------------------------------------ followers
    def on_job_new(self, body: dict) -> None:
        if body["job_id"] in self.jobs:
            return
        job = Job.from_wire(body)
        job.last_token_mono = time.monotonic()
        self._store(job)
        self.node.ui_push(JOB_NEW, job.meta())

    def on_token(self, body: dict) -> None:
        job = self.jobs.get(body["job_id"])
        if job is None or job.state == "done":
            return
        if job.add_token(int(body["seq"]), body["token"], body["node_id"]):
            job.node_id = body["node_id"]
            self.node.ui_push(JOB_TOKEN, body)

    def on_handoff(self, body: dict) -> None:
        job = self.jobs.get(body["job_id"])
        if job and job.apply_handoff(body):
            self.node.ui_push(JOB_HANDOFF, body)

    def on_done(self, body: dict) -> None:
        job = self.jobs.get(body["job_id"])
        if job is None or job.state == "done":
            return
        job.state = "done"
        job.final_seq = body.get("final_seq")
        job.done_ts = body.get("done_ts") or time.time()
        if body.get("nodes_used"):
            job.nodes_used = list(body["nodes_used"])
        self.node.ui_push(JOB_DONE, body)

    def merge_sync(self, jobs: list[dict]) -> None:
        for d in jobs:
            remote = Job.from_wire(d)
            mine = self.jobs.get(remote.job_id)
            if mine is None:
                remote.last_token_mono = time.monotonic()
                self._store(remote)
                continue
            for h in remote.handoffs:
                if all((x["at_seq"], x["to_node"]) != (h["at_seq"], h["to_node"]) for x in mine.handoffs):
                    mine.apply_handoff(h)
            for s, (t, nid) in remote.tokens.items():
                mine.tokens.setdefault(s, (t, nid))
            if remote.state == "done" and mine.state != "done":
                mine.state, mine.final_seq, mine.done_ts = "done", remote.final_seq, remote.done_ts
                mine.nodes_used = remote.nodes_used or mine.nodes_used
