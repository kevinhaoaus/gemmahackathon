"""Membership view + bully leader election with terms and hysteresis.

Rules (docs/PROTOCOL.md §3):
  * eligible = alive (heartbeat < peer_timeout old) and healthy (own LLM answered < 2 s ago)
  * leader   = max(rank, node_id) over eligible nodes
  * only the winner self-promotes: term = max_term_seen + 1, announced in its heartbeat
  * followers adopt the claimant with the highest (term, rank, node_id)
  * failover is immediate; preempting a healthy leader needs 3 s stability and no active job
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .config import NodeConfig
from .logutil import get_logger, set_term
from .protocol import Heartbeat

log = get_logger("election")


@dataclass
class PeerState:
    hb: Heartbeat
    addr: tuple
    last_seen: float                  # monotonic
    first_seen: float
    stable_since: float | None        # monotonic since continuously alive+healthy
    boot_id: str = ""


class Election:
    def __init__(self, cfg: NodeConfig, has_unfinished_job: Callable[[], bool],
                 on_change: Callable[[str | None, str | None, int], None]):
        self.cfg = cfg
        self.has_unfinished_job = has_unfinished_job
        self.on_change = on_change
        self.peers: dict[str, PeerState] = {}
        self.term = 0
        self.max_term_seen = 0
        self.leader_id: str | None = None
        self.self_healthy = False
        self.self_stable_since: float | None = None
        self.started = time.monotonic()

    # ------------------------------------------------------------------ membership
    def observe(self, hb: Heartbeat, addr: tuple) -> tuple[bool, bool]:
        """Record a heartbeat. Returns (is_new_or_returning, rebooted)."""
        now = time.monotonic()
        ps = self.peers.get(hb.node_id)
        returning = ps is None or not self._alive(ps, now)
        rebooted = ps is not None and bool(hb.boot_id) and ps.boot_id != hb.boot_id
        if ps is None or returning:
            ps = PeerState(hb, addr, now, now, None, hb.boot_id)
            self.peers[hb.node_id] = ps
        ps.hb, ps.addr, ps.last_seen, ps.boot_id = hb, addr, now, hb.boot_id
        if hb.healthy:
            if ps.stable_since is None:
                ps.stable_since = now
        else:
            ps.stable_since = None
        self.max_term_seen = max(self.max_term_seen, hb.term)
        return returning, rebooted

    def _alive(self, ps: PeerState, now: float) -> bool:
        return now - ps.last_seen <= self.cfg.peer_timeout

    def alive_peers(self) -> dict[str, PeerState]:
        now = time.monotonic()
        return {n: p for n, p in self.peers.items() if self._alive(p, now)}

    def is_alive(self, node_id: str) -> bool:
        if node_id == self.cfg.node_id:
            return True
        ps = self.peers.get(node_id)
        return ps is not None and self._alive(ps, time.monotonic())

    def set_self_health(self, healthy: bool) -> None:
        now = time.monotonic()
        if healthy and not self.self_healthy:
            self.self_stable_since = now
        if not healthy:
            self.self_stable_since = None
        self.self_healthy = healthy

    @property
    def is_leader(self) -> bool:
        return self.leader_id == self.cfg.node_id

    # ------------------------------------------------------------------ term guard
    def accept_leader_msg(self, src: str, term: int) -> bool:
        """Gate for LEADER_ONLY bus messages (split-brain protection)."""
        if term < self.term:
            return False
        if term > self.term:
            # A newer leader exists; its heartbeat may simply not have arrived yet.
            self._set(src, term, why=f"bus message at newer term from {src}")
            return True
        return self.leader_id in (None, src)

    # ------------------------------------------------------------------ the tick
    def tick(self) -> None:
        now = time.monotonic()
        me = self.cfg.node_id
        alive = self.alive_peers()

        # 1. Adopt the strongest live claimant (a node whose heartbeat says it is leader).
        claimants = [p.hb for p in alive.values() if p.hb.leader_id == p.hb.node_id and p.hb.healthy]
        if claimants:
            best = max(claimants, key=lambda h: (h.term, h.rank, h.node_id))
            mine = (self.term, self.cfg.rank, me) if self.is_leader else None
            cur = self._current_claim(alive)
            if mine is not None:
                if (best.term, best.rank, best.node_id) > mine:
                    self._set(best.node_id, best.term, why=f"stepping down to {best.node_id}")
            elif best.term > self.term or (best.term == self.term and (
                    cur is None or (best.rank, best.node_id) > (cur.rank, cur.node_id))):
                if self.leader_id != best.node_id or self.term != best.term:
                    self._set(best.node_id, best.term, why="following claimant")

        # 2. Drop a dead / unhealthy / abdicated leader.
        if self.leader_id and self.leader_id != me:
            lp = alive.get(self.leader_id)
            if lp is None or not lp.hb.healthy or (lp.hb.leader_id != lp.hb.node_id
                                                   and lp.hb.term >= self.term):
                why = "dead" if lp is None else ("unhealthy" if not lp.hb.healthy else "abdicated")
                self._set(None, self.term, why=f"leader {self.leader_id} {why}")
        if self.is_leader and not self.self_healthy:
            self._set(None, self.term, why="I am unhealthy; abdicating")

        # 3. Should I claim?
        if now - self.started < self.cfg.startup_grace_s or not self.self_healthy:
            return
        if self.is_leader:
            return
        eligible = [(p.hb.rank, n, p) for n, p in alive.items() if p.hb.healthy]
        eligible.append((self.cfg.rank, me, None))
        if self.leader_id is None:
            # Prefer nodes that have been stable for takeover_stable_s (a flapping or freshly
            # returned node should not grab leadership); fall back to anyone eligible.
            stable = [e for e in eligible if self._stable_for(e[2], now) >= self.cfg.takeover_stable_s]
            pool = stable or eligible
            top = max(pool, key=lambda e: (e[0], e[1]))
            if top[1] == me:
                self._claim("no live leader")
            return
        # Preemption of a healthy, alive, lower-ranked leader (hysteresis).
        lp = alive.get(self.leader_id)
        if lp is None:
            return
        if (self.cfg.rank, me) <= (lp.hb.rank, lp.hb.node_id):
            return
        top = max(eligible, key=lambda e: (e[0], e[1]))
        if top[1] != me:
            return
        if self._stable_for(None, now) < self.cfg.takeover_stable_s:
            return
        if lp.hb.active_job_id or self.has_unfinished_job():
            return   # never preempt a job that is still streaming
        self._claim(f"preempting lower-ranked {self.leader_id} (idle, I am stable)")

    def _current_claim(self, alive: dict[str, PeerState]) -> Heartbeat | None:
        if self.leader_id and self.leader_id in alive:
            return alive[self.leader_id].hb
        return None

    def _stable_for(self, ps: PeerState | None, now: float) -> float:
        since = self.self_stable_since if ps is None else ps.stable_since
        return 0.0 if since is None else now - since

    def _claim(self, why: str) -> None:
        new_term = max(self.max_term_seen, self.term) + 1
        self._set(self.cfg.node_id, new_term, why=f"claiming leadership: {why}")

    def _set(self, leader: str | None, term: int, why: str) -> None:
        old = self.leader_id
        self.leader_id = leader
        self.term = max(self.term, term)
        self.max_term_seen = max(self.max_term_seen, self.term)
        set_term(self.term)
        if old != leader:
            log.info("leader %s -> %s (term %d): %s", old, leader, self.term, why)
            self.on_change(old, leader, self.term)
