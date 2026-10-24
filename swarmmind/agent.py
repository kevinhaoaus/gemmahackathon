"""SwarmMind node entrypoint: `python -m swarmmind.agent --config configs/laptop.yaml`.

Wires membership (UDP heartbeats), election, the WS peer bus, jobs, tasks, topics, vision,
uplink and the HTTP/WS API + dashboard into one asyncio process.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from .cell.labmate import LabError, LabMate
from .cell.planner import PlanBusy, PlanError, PlanManager
from .config import NodeConfig, load_config
from .election import Election
from .inference import make_backend
from .jobs import JobManager
from .logutil import get_logger, setup_logging
from .protocol import (EVENT, JOB_DONE, JOB_HANDOFF, JOB_NEW, JOB_TOKEN, LEADER_ONLY, PLAN_NEW,
                       PLAN_UPDATE, REPORT_NEW, REPORT_SENT, SYNC_REQ, SYNC_RESP, TASK_ASSIGN,
                       TASK_CANCEL, TASK_PROGRESS, TASK_RESULT, TASK_SUBMIT, TOPIC_PUB, Envelope,
                       Heartbeat)
from .tasks import REGISTRY, TaskManager, is_available, load_capability_modules
from .topics import TopicBus
from .transport import HeartbeatChannel, PeerBus
from .uplink import Uplink
from .vision import Vision

log = get_logger("agent")
ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT / "dashboard"
FWD_HEADER = "x-swarm-forwarded"


class Node:
    def __init__(self, cfg: NodeConfig):
        self.cfg = cfg
        self.boot_id = uuid.uuid4().hex[:8]
        self.backend = make_backend(cfg)
        self.vision = Vision(cfg)
        self.jobs = JobManager(self)
        self.tasks = TaskManager(self)
        self.topics = TopicBus(cfg.node_id)
        self.plans = PlanManager(self)
        # robot cell seams (docs/ROBOT_CELL.md): set by the perception capability on the camera node
        self.cell_state: dict | None = None
        self.cell_annotated_jpeg: bytes | None = None
        # LabMate seams (docs/LABMATE.md): latest cell.blobs dict + the frame with numbered marks
        self.cell_blobs: dict | None = None
        self.cell_marked_jpeg: bytes | None = None
        self.lab = LabMate(self, self.plans)     # LabMate plans ride on the PlanManager
        # never preempt a leader mid-answer or mid-plan (an arm may be moving)
        self.election = Election(cfg, lambda: self.jobs.has_unfinished() or self.plans.has_active(),
                                 self._on_leader_change)
        self.bus = PeerBus(cfg, self._on_message, self._on_link_up)
        self.hb = HeartbeatChannel(cfg, self._make_hb, self._on_hb)
        self.uplink = Uplink(cfg, self._uplink_broadcast, lambda: self.election.is_leader)
        self.events: collections.deque = collections.deque(maxlen=200)
        self.ui_clients: dict[WebSocket, asyncio.Queue] = {}
        self.hb_seq = 0
        self.leader_since = time.monotonic()
        self._last_health_ok = 0.0
        self._was_alive: set[str] = set()
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=1.5))

    # ------------------------------------------------------------------ helpers
    def capabilities(self) -> list[str]:
        caps = []
        if self.backend.name != "none":
            caps.append("llm.text")
            if self.backend.supports_images:
                caps.append("llm.vision")
        if self.vision.has_detector:
            caps.append("vision.detect")
        if self.cfg.camera_url:
            caps.append("camera")
        for c in self.cfg.capabilities:
            if c not in caps and is_available(c):
                caps.append(c)
        return caps

    def broadcast(self, type_: str, body: dict) -> None:
        self.bus.broadcast(Envelope(type_, self.cfg.node_id, self.election.term, body))

    def ui_push(self, type_: str, body: Any) -> None:
        msg = {"type": type_, "body": body}
        for q in list(self.ui_clients.values()):
            if q.qsize() < 5000:
                q.put_nowait(msg)

    def event(self, msg: str, level: str = "info", origin: str | None = None,
              ts: float | None = None, term: int | None = None) -> None:
        ev = {"ts": ts or time.time(), "level": level, "node_id": origin or self.cfg.node_id,
              "term": self.election.term if term is None else term, "msg": msg}
        self.events.append(ev)
        self.ui_push(EVENT, ev)
        if origin is None:
            getattr(log, "warning" if level == "warning" else "error" if level == "error" else "info")(msg)
            self.broadcast(EVENT, ev)

    async def _uplink_broadcast(self, type_: str, body: dict) -> None:
        self.broadcast(type_, body)
        self.ui_push(type_, body)

    def _make_hb(self) -> Heartbeat:
        self.hb_seq += 1
        return Heartbeat(node_id=self.cfg.node_id, rank=self.cfg.rank, ip=self.cfg.host_ip,
                         http_port=self.cfg.http_port, model=self.backend.model,
                         vision_mode=self.vision.mode, healthy=self.election.self_healthy,
                         leader_id=self.election.leader_id, term=self.election.term,
                         active_job_id=self.jobs.active_job_id(), seq=self.hb_seq,
                         capabilities=self.capabilities(), tps=round(self.backend.tps, 1),
                         boot_id=self.boot_id)

    def node_view(self) -> list[dict]:
        now = time.monotonic()
        e = self.election
        me = {"node_id": self.cfg.node_id, "rank": self.cfg.rank, "ip": self.cfg.host_ip,
              "http_port": self.cfg.http_port, "model": self.backend.model,
              "vision_mode": self.vision.mode, "alive": True, "healthy": e.self_healthy,
              "is_leader": e.is_leader, "tps": round(self.backend.tps, 1),
              "capabilities": self.capabilities(), "active_job_id": self.jobs.active_job_id(),
              "last_seen_ms": 0, "self": True}
        out = [me]
        for nid, ps in sorted(self.election.peers.items()):
            hb = ps.hb
            out.append({"node_id": nid, "rank": hb.rank, "ip": hb.ip, "http_port": hb.http_port,
                        "model": hb.model, "vision_mode": hb.vision_mode,
                        "alive": e.is_alive(nid), "healthy": hb.healthy and e.is_alive(nid),
                        "is_leader": nid == e.leader_id, "tps": hb.tps,
                        "capabilities": hb.capabilities, "active_job_id": hb.active_job_id,
                        "last_seen_ms": round((now - ps.last_seen) * 1000), "self": False})
        return sorted(out, key=lambda n: (-n["rank"], n["node_id"]))

    def nodes_body(self) -> dict:
        nodes = self.node_view()
        top = max(nodes, key=lambda n: (n["rank"], n["node_id"]))["node_id"]
        return {"self": self.cfg.node_id, "leader_id": self.election.leader_id,
                "term": self.election.term, "top_node_id": top, "nodes": nodes}

    def snapshot(self) -> dict:
        body = self.nodes_body()
        body.update(jobs=self.jobs.snapshot(), reports=self.uplink.all_reports()[-50:],
                    events=list(self.events), tasks=self.tasks.snapshot(),
                    topics=list(self.topics.latest.values())[-50:], plans=self.plans.snapshot())
        return body

    async def cell_frame(self, forwarded: bool = False) -> bytes | None:
        """Newest annotated cell frame: local perception, else fetched from a camera node."""
        return await self._cell_image(self.cell_annotated_jpeg, "/cell/frame.jpg", forwarded)

    async def cell_marked(self, forwarded: bool = False) -> bytes | None:
        """Newest frame with LabMate's numbered blob marks: local, else from a camera node."""
        return await self._cell_image(self.cell_marked_jpeg, "/cell/marked.jpg", forwarded)

    async def _cell_image(self, local: bytes | None, path: str, forwarded: bool) -> bytes | None:
        if local:
            return local
        if forwarded:                       # already a peer's fetch: never chain further
            return None
        for nid, ps in self.election.alive_peers().items():
            if "vision.cell" not in (ps.hb.capabilities or []):
                continue
            try:
                r = await self.http.get(f"http://{ps.hb.ip}:{ps.hb.http_port}{path}",
                                        headers={FWD_HEADER: self.cfg.node_id}, timeout=2.0)
                if r.status_code == 200 and r.content:
                    return r.content
            except httpx.HTTPError as e:
                log.debug("%s from %s failed: %s", path, nid, e)
        return None

    # ------------------------------------------------------------------ membership / election
    def _on_hb(self, hb: Heartbeat, addr: tuple) -> None:
        returning, rebooted = self.election.observe(hb, addr)
        if rebooted:
            self.bus.drop_link(hb.node_id)
        if returning or rebooted:
            self.bus.ensure_link(hb.node_id, hb.ip, hb.http_port)
            if hb.node_id not in self._was_alive:
                self._was_alive.add(hb.node_id)
                self.event(f"node {hb.node_id} up (rank {hb.rank}, {hb.model}, vision {hb.vision_mode})")
            elif rebooted:
                self.event(f"node {hb.node_id} rebooted")

    def _on_leader_change(self, old: str | None, new: str | None, term: int) -> None:
        self.hb.kick()
        me = self.cfg.node_id
        if new == me:
            self.leader_since = time.monotonic()
            self.event(f"{me} is now LEADER (term {term})")
            self.jobs.resume_unfinished()
            self.plans.resume_all()
        elif old == me:
            self.jobs.cancel_all(f"leadership moved to {new}")
            self.plans.stop_drivers()
            self.event(f"{me} stepped down; leader is {new} (term {term})")
        elif new is None:
            self.event(f"leader {old} lost; electing (term {term})", level="warning")
        self.ui_push("ui.nodes", self.nodes_body())

    async def _membership_loop(self) -> None:
        n = 0
        while True:
            self.election.tick()
            for nid in list(self._was_alive):
                if not self.election.is_alive(nid):
                    self._was_alive.discard(nid)
                    self.bus.drop_link(nid)
                    self.event(f"node {nid} DOWN (no heartbeat for {self.cfg.peer_timeout * 1000:.0f} ms)",
                               level="warning")
            n += 1
            if self.election.is_leader and n % 4 == 0:
                self.tasks.watchdog()
            await asyncio.sleep(0.05)

    async def _health_loop(self) -> None:
        while True:
            try:
                ok = await asyncio.wait_for(self.backend.check_health(), 1.5)
            except Exception:  # noqa: BLE001 - timeout or backend error = not ok
                ok = False
            now = time.monotonic()
            if ok:
                self._last_health_ok = now
            healthy = self._last_health_ok > 0 and now - self._last_health_ok <= self.cfg.health_window
            if healthy != self.election.self_healthy:
                self.event(f"{self.cfg.node_id} LLM {'healthy' if healthy else 'UNHEALTHY'} ({self.backend.model})",
                           level="info" if healthy else "warning")
            self.election.set_self_health(healthy)
            await asyncio.sleep(self.cfg.health_interval)

    async def _warmup(self) -> None:
        """Load the model into (V)RAM now and pin it (keep_alive -1). A standby node that loads
        lazily would add 5-20 s to its first handoff."""
        if self.backend.name != "ollama":
            return
        for attempt in range(30):
            try:
                t0 = time.monotonic()
                r = await self.http.post(f"{self.cfg.ollama_url}/api/generate",
                                         json={"model": self.cfg.model, "prompt": "", "keep_alive": -1},
                                         timeout=httpx.Timeout(300.0, connect=2.0))
                if r.status_code == 200:
                    log.info("model %s loaded and pinned in %.1f s", self.cfg.model, time.monotonic() - t0)
                    return
                log.warning("warm-up got HTTP %d: %s", r.status_code, r.text[:200])
            except Exception as e:  # noqa: BLE001 - Ollama may still be starting
                log.warning("warm-up attempt %d failed: %s", attempt + 1, e)
            await asyncio.sleep(5)

    async def _ui_nodes_loop(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            self.ui_push("ui.nodes", self.nodes_body())

    async def _frame_loop(self) -> None:
        if not self.cfg.camera_url:
            return
        while True:
            await self.vision.capture_frame()
            await asyncio.sleep(2.0)

    # ------------------------------------------------------------------ peer bus
    async def _on_link_up(self, peer: str) -> None:
        self.bus.send_to(peer, Envelope(SYNC_REQ, self.cfg.node_id, self.election.term, {}))

    async def _on_message(self, env: Envelope) -> None:
        t, b = env.type, env.body
        if t in LEADER_ONLY and not self.election.accept_leader_msg(env.src, env.term):
            log.debug("dropped %s from %s at stale term %d", t, env.src, env.term)
            return
        try:
            if t == JOB_TOKEN:
                self.jobs.on_token(b)
            elif t == JOB_NEW:
                self.jobs.on_job_new(b)
            elif t == JOB_HANDOFF:
                self.jobs.on_handoff(b)
            elif t == JOB_DONE:
                self.jobs.on_done(b)
            elif t == TOPIC_PUB:
                if self.topics.accept(b):
                    await self.topics.deliver(b)
            elif t == REPORT_NEW:
                if self.uplink.add_report(b):
                    self.ui_push(REPORT_NEW, {**b, "sent": False})
            elif t == REPORT_SENT:
                self.uplink.mark_sent(list(b.get("job_ids") or []))
                self.ui_push(REPORT_SENT, b)
            elif t == TASK_SUBMIT:
                if self.election.is_leader:
                    self.tasks.submit(b["capability"], b.get("args") or {}, b.get("task_id"))
            elif t == TASK_ASSIGN:
                self.tasks.on_assign(b)
            elif t == TASK_PROGRESS:
                self.tasks.on_progress(b)
            elif t == TASK_RESULT:
                self.tasks.on_result(b)
            elif t == TASK_CANCEL:
                self.tasks.on_cancel(b)
            elif t == PLAN_NEW:
                self.plans.on_new(b)
            elif t == PLAN_UPDATE:
                self.plans.on_update(b)
            elif t == EVENT:
                self.event(b.get("msg", ""), b.get("level", "info"), origin=b.get("node_id", env.src),
                           ts=b.get("ts"), term=b.get("term"))
            elif t == SYNC_REQ:
                body = {"jobs": self.jobs.sync_payload(), "reports": self.uplink.all_reports(),
                        "tasks": self.tasks.snapshot(), "plans": self.plans.sync_payload()}
                self.bus.send_to(env.src, Envelope(SYNC_RESP, self.cfg.node_id, self.election.term, body))
            elif t == SYNC_RESP:
                self.jobs.merge_sync(b.get("jobs") or [])
                self.tasks.merge_sync(b.get("tasks") or [])
                self.plans.merge_sync(b.get("plans") or [])
                sent =[r["job_id"] for r in b.get("reports") or [] if r.get("sent")]
                for r in b.get("reports") or []:
                    self.uplink.add_report({k: v for k, v in r.items() if k != "sent"})
                if sent:
                    self.uplink.mark_sent(sent)
                log.info("synced from %s: %d jobs, %d reports, %d tasks", env.src,
                         len(b.get("jobs") or []), len(b.get("reports") or []), len(b.get("tasks") or []))
            else:
                log.debug("unknown message type %s from %s", t, env.src)
        except Exception:  # noqa: BLE001 - one bad message must never kill the bus
            log.exception("error handling %s from %s", t, env.src)

    # ------------------------------------------------------------------ leader forwarding
    async def forward_to_leader(self, path: str, payload: dict, request: Request) -> Response:
        e = self.election
        if request.headers.get(FWD_HEADER) or not e.leader_id:
            return JSONResponse({"error": "no leader (election in progress)", "leader_id": e.leader_id},
                                status_code=503)
        ps = e.peers.get(e.leader_id)
        if ps is None or not e.is_alive(e.leader_id):
            return JSONResponse({"error": "leader unreachable"}, status_code=503)
        url = f"http://{ps.hb.ip}:{ps.hb.http_port}{path}"
        try:
            r = await self.http.post(url, json=payload, headers={FWD_HEADER: self.cfg.node_id})
            return JSONResponse(r.json(), status_code=r.status_code)
        except Exception as ex:  # noqa: BLE001
            return JSONResponse({"error": f"forward to leader failed: {ex}"}, status_code=503)

    # ------------------------------------------------------------------ topics
    async def publish(self, topic: str, data: Any) -> dict:
        sample = self.topics.make_sample(topic, data)
        self.topics.accept(sample)
        self.broadcast(TOPIC_PUB, sample)
        await self.topics.deliver(sample)
        return sample

    # ------------------------------------------------------------------ lifecycle
    async def run(self) -> None:
        loaded = load_capability_modules()
        self._modules = loaded
        for mod in loaded:
            if hasattr(mod, "setup"):
                try:
                    await mod.setup(self)
                except Exception:  # noqa: BLE001 - a broken driver must not take the node down
                    log.exception("capability module %s setup failed", mod.__name__)
        # checked after setup: some modules (cell_arm) register handlers dynamically there
        missing = [c for c in self.cfg.capabilities if c not in REGISTRY]
        log.info("node %s rank %d | backend %s model %s images=%s | vision %s | caps %s | handlers %s",
                 self.cfg.node_id, self.cfg.rank, self.backend.name, self.backend.model,
                 self.backend.supports_images, self.vision.mode, self.capabilities(), sorted(REGISTRY))
        if missing:
            log.warning("capabilities without handlers: %s", missing)
        await self.hb.start()
        app = build_app(self)
        server = uvicorn.Server(uvicorn.Config(app, host=self.cfg.bind_host, port=self.cfg.http_port,
                                               log_level="warning", ws="websockets-sansio", lifespan="off"))
        servers = [server.serve()]
        if self.cfg.ssl_certfile and self.cfg.ssl_keyfile:
            tls = uvicorn.Server(uvicorn.Config(app, host=self.cfg.bind_host, port=self.cfg.https_port,
                                                log_level="warning", ws="websockets-sansio", lifespan="off",
                                                ssl_certfile=self.cfg.ssl_certfile,
                                                ssl_keyfile=self.cfg.ssl_keyfile))
            servers.append(tls.serve())
            log.info("HTTPS for phones on :%d (teleop: https://%s:%d/teleop)",
                     self.cfg.https_port, self.cfg.host_ip, self.cfg.https_port)
        bg = [asyncio.create_task(c) for c in (self.hb.run(), self._membership_loop(), self._health_loop(),
                                                self._ui_nodes_loop(), self._frame_loop(), self.uplink.run(),
                                                self._warmup())]
        try:
            await asyncio.gather(*servers)
        finally:
            for t in bg:
                t.cancel()
            self.jobs.cancel_all("shutdown")
            for mod in getattr(self, "_modules", []):
                if hasattr(mod, "shutdown"):       # e.g. so101: hold position, close the serial port
                    try:
                        r = mod.shutdown()
                        if asyncio.iscoroutine(r):
                            await r
                    except Exception:  # noqa: BLE001
                        log.exception("capability module %s shutdown failed", mod.__name__)
            self.bus.close()
            self.hb.close()
            await self.backend.aclose()
            await self.vision.aclose()
            await self.http.aclose()


def build_app(node: Node) -> FastAPI:
    app = FastAPI(title=f"swarmmind {node.cfg.node_id}")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/")
    async def index() -> Response:
        return FileResponse(DASHBOARD / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/teleop")
    async def teleop() -> Response:
        p = DASHBOARD / "teleop.html"
        if not p.is_file():
            return JSONResponse({"error": "teleop page not installed"}, status_code=404)
        return FileResponse(p, headers={"Cache-Control": "no-store"})

    @app.get("/peers")
    async def peers() -> dict:
        b = node.nodes_body()
        return {"self": b["self"], "leader_id": b["leader_id"], "term": b["term"],
                "top_node_id": b["top_node_id"], "peers": b["nodes"]}

    @app.get("/state")
    async def state() -> dict:
        return node.snapshot()

    @app.get("/frame.jpg")
    async def frame() -> Response:
        jpeg = node.vision.latest_jpeg
        if jpeg is None and node.cfg.camera_url:
            jpeg = await node.vision.capture_frame()
        if jpeg is None:
            for j in reversed(node.jobs.order):    # fall back to the frame of the latest job
                job = node.jobs.jobs.get(j)
                if job and job.image_b64:
                    jpeg = base64.b64decode(job.image_b64)
                    break
        if jpeg is None:
            return Response(status_code=404)
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/ask")
    async def ask(request: Request) -> Response:
        payload = await request.json()
        q = str(payload.get("question", "")).strip()
        if not q:
            return JSONResponse({"error": "question is required"}, status_code=400)
        if node.election.is_leader:
            job = await node.jobs.create(q)
            return JSONResponse({"job_id": job.job_id, "leader_id": node.cfg.node_id})
        return await node.forward_to_leader("/ask", {"question": q}, request)

    @app.post("/task")
    async def task(request: Request) -> Response:
        payload = await request.json()
        cap = str(payload.get("capability", "")).strip()
        if not cap:
            return JSONResponse({"error": "capability is required"}, status_code=400)
        if node.election.is_leader:
            t = node.tasks.submit(cap, payload.get("args") or {}, payload.get("task_id"))
            return JSONResponse({"task_id": t.task_id, "assignee": t.assignee, "state": t.state})
        return await node.forward_to_leader("/task", payload, request)

    @app.post("/task/{task_id}/cancel")
    async def cancel(task_id: str, request: Request) -> Response:
        if node.election.is_leader:
            return JSONResponse({"cancelled": node.tasks.cancel(task_id)})
        return await node.forward_to_leader(f"/task/{task_id}/cancel", {}, request)

    @app.get("/tasks")
    async def tasks() -> dict:
        return {"tasks": node.tasks.snapshot()}

    # ---- plans (LabMate runs on the shared plan manager; docs/LABMATE.md)
    @app.get("/cell/plans")
    async def cell_plans() -> dict:
        return {"plans": node.plans.snapshot()}

    @app.post("/cell/plans/{plan_id}/cancel")
    async def cell_cancel(plan_id: str, request: Request) -> Response:
        if node.election.is_leader:
            return JSONResponse({"cancelled": node.plans.cancel(plan_id)})
        return await node.forward_to_leader(f"/cell/plans/{plan_id}/cancel", {}, request)

    @app.get("/cell/frame.jpg")
    async def cell_frame(request: Request) -> Response:
        jpeg = await node.cell_frame(forwarded=bool(request.headers.get(FWD_HEADER)))
        if jpeg is None:
            return Response(status_code=404)
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/cell/marked.jpg")
    async def cell_marked(request: Request) -> Response:
        jpeg = await node.cell_marked(forwarded=bool(request.headers.get(FWD_HEADER)))
        if jpeg is None:
            return Response(status_code=404)
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # ---- LabMate (docs/LABMATE.md); POST routes run on the leader (forwarded from any node)
    async def lab_call(path: str, payload: dict, request: Request, fn) -> Response:
        if not node.election.is_leader:
            return await node.forward_to_leader(path, payload, request)
        try:
            plan = await fn()
        except PlanBusy as e:
            return JSONResponse({"error": str(e), "plan_id": e.plan_id}, status_code=409)
        except LabError as e:
            return JSONResponse({"error": str(e)}, status_code=e.status)
        except PlanError as e:
            return JSONResponse({"error": str(e)}, status_code=503)
        return JSONResponse({"plan_id": plan.plan_id, "state": plan.state, "leader_id": node.cfg.node_id})

    @app.post("/lab/inspect")
    async def lab_inspect(request: Request) -> Response:
        return await lab_call("/lab/inspect", {}, request, node.lab.inspect)

    @app.post("/lab/answer")
    async def lab_answer(request: Request) -> Response:
        payload = await request.json()
        text = str(payload.get("text", "")).strip()
        if not text:
            return JSONResponse({"error": "text is required"}, status_code=400)
        return await lab_call("/lab/answer", {"text": text}, request, lambda: node.lab.answer(text))

    @app.post("/lab/fetch")
    async def lab_fetch(request: Request) -> Response:
        payload = await request.json()
        item = str(payload.get("item", "")).strip()
        if not item:
            return JSONResponse({"error": "item is required"}, status_code=400)
        return await lab_call("/lab/fetch", {"item": item}, request, lambda: node.lab.fetch(item))

    @app.get("/lab/state")
    async def lab_state() -> dict:
        return node.lab.state()

    @app.post("/pub/{topic}")
    async def pub(topic: str, request: Request) -> dict:
        return await node.publish(topic, await request.json())

    @app.get("/topics")
    async def topics() -> dict:
        return {"topics": node.topics.latest}

    @app.websocket("/ws/topics")
    async def ws_topics(ws: WebSocket) -> None:
        """Bidirectional: send {"topic","data"} to publish; receive samples matching ?sub=a.,b."""
        await ws.accept()
        subs = tuple(s for s in (ws.query_params.get("sub") or "").split(",") if s)
        node.topics.ws_clients[ws] = subs
        try:
            while True:
                msg = await ws.receive_json()
                if isinstance(msg, dict) and msg.get("topic"):
                    await node.publish(str(msg["topic"]), msg.get("data"))
        except (WebSocketDisconnect, RuntimeError, ValueError):
            pass
        finally:
            node.topics.ws_clients.pop(ws, None)

    @app.websocket("/ws/peer")
    async def ws_peer(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                await node.bus.handle_inbound(await ws.receive_text())
        except (WebSocketDisconnect, RuntimeError):
            pass

    @app.websocket("/ws/ui")
    async def ws_ui(ws: WebSocket) -> None:
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue()
        node.ui_clients[ws] = q
        q.put_nowait({"type": "ui.snapshot", "body": node.snapshot()})

        async def sender() -> None:
            while True:
                await ws.send_json(await q.get())

        async def receiver() -> None:
            while True:
                await ws.receive_text()      # clients only send keepalives; this detects close

        tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
            node.ui_clients.pop(ws, None)

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="SwarmMind node")
    ap.add_argument("--config", required=True, help="path to node YAML")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.node_id, cfg.log_level)
    try:
        asyncio.run(Node(cfg).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
