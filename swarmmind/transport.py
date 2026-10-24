"""Transport: UDP heartbeats (discovery + liveness) and the WebSocket peer bus.

Peer bus topology: full mesh. Each node opens ONE outbound WebSocket to every alive peer and
only sends on its outbound links; inbound links (served at /ws/peer) are receive-only.
Nothing is relayed, so there are no loops; `mid` dedupe guards against reconnect replays.
"""
from __future__ import annotations

import asyncio
import os
import socket
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Awaitable, Callable

import websockets

from .config import NodeConfig
from .logutil import get_logger
from .protocol import Envelope, Heartbeat

log = get_logger("transport")

MAX_QUEUE = 20000          # per-peer outbound backlog before we drop the link and rely on sync
MAX_WS_MESSAGE = 32 * 1024 * 1024


def parse_peer(entry: str, default_udp: int) -> tuple[str, int, int]:
    """'host:http_port' or 'host:http_port:udp_port' -> (host, http_port, udp_port)."""
    parts = entry.strip().split(":")
    if len(parts) == 2:
        return parts[0], int(parts[1]), default_udp
    if len(parts) == 3:
        return parts[0], int(parts[1]), int(parts[2])
    raise ValueError(f"bad static peer {entry!r} (want host:http_port[:udp_port])")


# --------------------------------------------------------------------------- UDP heartbeats
class _UdpProto(asyncio.DatagramProtocol):
    def __init__(self, on_datagram: Callable[[bytes, tuple], None]):
        self.on_datagram = on_datagram

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        self.on_datagram(data, addr)

    def error_received(self, exc: Exception) -> None:
        # Windows reports ICMP port-unreachable for dead static peers here; harmless.
        pass


class HeartbeatChannel:
    """Sends our heartbeat every interval (broadcast + unicast) and reports received ones."""

    def __init__(self, cfg: NodeConfig, make_hb: Callable[[], Heartbeat],
                 on_hb: Callable[[Heartbeat, tuple], None]):
        self.cfg = cfg
        self.make_hb = make_hb
        self.on_hb = on_hb
        self.transport: asyncio.DatagramTransport | None = None
        self.targets: list[tuple[str, int]] = []
        for p in cfg.static_peers:
            host, _, udp = parse_peer(p, cfg.udp_port)
            self.targets.append((host, udp))
        if cfg.broadcast_addr:
            self.targets.append((cfg.broadcast_addr, cfg.udp_port))
        self._kick = asyncio.Event()

    async def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if os.name != "nt":
            # Linux: allow quick restarts. Windows: SO_REUSEADDR lets two processes share the
            # port silently (heartbeats would be split between them), so leave it off there.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        bind_host = "" if self.cfg.bind_host in ("0.0.0.0", "") else self.cfg.bind_host
        if self.cfg.broadcast_addr:
            bind_host = ""   # broadcasts only arrive on a wildcard bind
        sock.bind((bind_host, self.cfg.udp_port))
        sock.setblocking(False)
        loop = asyncio.get_running_loop()
        self.transport, _ = await loop.create_datagram_endpoint(
            lambda: _UdpProto(self._on_datagram), sock=sock)
        log.info("udp heartbeat on :%d -> %s", self.cfg.udp_port,
                 ", ".join(f"{h}:{p}" for h, p in self.targets) or "(no targets)")

    def kick(self) -> None:
        """Send a heartbeat right now (used on leadership change)."""
        self._kick.set()

    async def run(self) -> None:
        while True:
            self._send_once()
            try:
                await asyncio.wait_for(self._kick.wait(), self.cfg.heartbeat_interval)
            except asyncio.TimeoutError:
                pass
            self._kick.clear()

    def _send_once(self) -> None:
        if not self.transport:
            return
        data = self.make_hb().to_bytes()
        for target in self.targets:
            try:
                self.transport.sendto(data, target)
            except OSError:
                pass   # network down / unreachable: keep trying next tick

    def _on_datagram(self, data: bytes, addr: tuple) -> None:
        try:
            hb = Heartbeat.from_bytes(data)
        except Exception:  # noqa: BLE001 - garbage on the port is ignored
            return
        if hb.node_id == self.cfg.node_id:
            return
        self.on_hb(hb, addr)

    def close(self) -> None:
        if self.transport:
            self.transport.close()


# --------------------------------------------------------------------------- peer bus
@dataclass
class _Link:
    node_id: str
    url: str
    queue: asyncio.Queue
    task: asyncio.Task | None = None
    connected: bool = False


class PeerBus:
    """Outbound WS links to peers + inbound dedupe. App code calls broadcast()/send_to()."""

    def __init__(self, cfg: NodeConfig, on_message: Callable[[Envelope], Awaitable[None]],
                 on_link_up: Callable[[str], Awaitable[None]]):
        self.cfg = cfg
        self.on_message = on_message
        self.on_link_up = on_link_up
        self.links: dict[str, _Link] = {}
        self._seen: OrderedDict[str, None] = OrderedDict()

    # ---- link management (driven by membership)
    def ensure_link(self, node_id: str, ip: str, http_port: int) -> None:
        url = f"ws://{ip}:{http_port}/ws/peer?src={self.cfg.node_id}"
        link = self.links.get(node_id)
        if link and link.url == url and link.task and not link.task.done():
            return
        if link:
            self.drop_link(node_id)
        link = _Link(node_id, url, asyncio.Queue())
        link.task = asyncio.create_task(self._link_loop(link), name=f"link-{node_id}")
        self.links[node_id] = link

    def drop_link(self, node_id: str) -> None:
        link = self.links.pop(node_id, None)
        if link and link.task:
            link.task.cancel()

    def connected_peers(self) -> list[str]:
        return [n for n, l in self.links.items() if l.connected]

    async def _link_loop(self, link: _Link) -> None:
        while True:
            try:
                async with websockets.connect(link.url, open_timeout=1.0, ping_interval=2.0,
                                              ping_timeout=2.0, max_size=MAX_WS_MESSAGE,
                                              close_timeout=0.5) as ws:
                    link.connected = True
                    log.info("peer link up -> %s", link.node_id)
                    asyncio.create_task(self.on_link_up(link.node_id))
                    while True:
                        env: Envelope = await link.queue.get()
                        await ws.send(env.to_json())
            except asyncio.CancelledError:
                link.connected = False
                raise
            except Exception as e:  # noqa: BLE001 - any failure: back off and retry
                if link.connected:
                    log.warning("peer link down -> %s (%s)", link.node_id, type(e).__name__)
                link.connected = False
                await asyncio.sleep(0.2)

    # ---- sending
    def broadcast(self, env: Envelope) -> None:
        self._remember(env.mid)
        for link in list(self.links.values()):
            self._enqueue(link, env)

    def send_to(self, node_id: str, env: Envelope) -> None:
        link = self.links.get(node_id)
        if link:
            self._enqueue(link, env)

    def _enqueue(self, link: _Link, env: Envelope) -> None:
        if link.queue.qsize() >= MAX_QUEUE:
            log.warning("backlog to %s overflowed; resetting link (peer will resync)", link.node_id)
            self.drop_link(link.node_id)
            return
        link.queue.put_nowait(env)

    # ---- receiving (called by the /ws/peer endpoint)
    async def handle_inbound(self, raw: str) -> None:
        try:
            env = Envelope.from_json(raw)
        except Exception:  # noqa: BLE001
            return
        if env.src == self.cfg.node_id or self._is_dup(env.mid):
            return
        await self.on_message(env)

    def _is_dup(self, mid: str) -> bool:
        if mid in self._seen:
            return True
        self._remember(mid)
        return False

    def _remember(self, mid: str) -> None:
        self._seen[mid] = None
        if len(self._seen) > 50000:
            for _ in range(10000):
                self._seen.popitem(last=False)

    def close(self) -> None:
        for nid in list(self.links):
            self.drop_link(nid)


def monotonic_ms() -> float:
    return time.monotonic() * 1000.0
