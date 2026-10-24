"""Store-and-forward of completed-job reports to LAN-only HTTP sinks (no cloud).

Reports are kept in <data_dir>/<node_id>/reports.jsonl. The leader periodically checks whether
a base station (scripts/uplink_sink.py) is reachable and POSTs every unsent report to it.
Never double-posts: local dedupe by job_id, and the sink is idempotent by job_id too.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import urlsplit, urlunsplit

import httpx

from .config import NodeConfig
from .logutil import get_logger

log = get_logger("uplink")

CONNECT_TIMEOUT_S = 0.5
POST_TIMEOUT_S = 3.0


def _sink_endpoint(url: str) -> str:
    """A bare base URL (http://host:8800 or .../) posts to /reports."""
    parts = urlsplit(url)
    if parts.path in ("", "/"):
        parts = parts._replace(path="/reports")
    return urlunsplit(parts)


def _host_port(url: str) -> tuple[str, int] | None:
    parts = urlsplit(url)
    if not parts.hostname:
        return None
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return parts.hostname, port


class Uplink:
    def __init__(self, cfg: NodeConfig, broadcast: Callable[[str, dict], Awaitable[None]],
                 is_leader: Callable[[], bool]):
        self.cfg = cfg
        self.broadcast = broadcast
        self.is_leader = is_leader
        self.path: Path = cfg.node_data_dir / "reports.jsonl"
        self._reports: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._load()

    # ---- persistence
    def _load(self) -> None:
        if not self.path.is_file():
            return
        bad = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if not isinstance(rec, dict) or not rec.get("job_id"):
                    bad += 1
                    continue
                jid = str(rec["job_id"])
                rec["sent"] = bool(rec.get("sent", False))
                self._reports.pop(jid, None)   # last record wins, keep latest order
                self._reports[jid] = rec
        log.info("uplink: loaded %d reports (%d pending) from %s%s", len(self._reports),
                 self.pending_count(), self.path, f", skipped {bad} bad lines" if bad else "")

    def _append(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _rewrite(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in self._reports.values():
                f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    # ---- public API
    def add_report(self, report: dict) -> bool:
        jid = report.get("job_id") if isinstance(report, dict) else None
        if not jid:
            log.warning("uplink: ignoring report without job_id")
            return False
        jid = str(jid)
        with self._lock:
            if jid in self._reports:
                return False
            rec = dict(report)
            rec["job_id"] = jid
            rec["sent"] = bool(rec.get("sent", False))
            self._reports[jid] = rec
            try:
                self._append(rec)
            except OSError as e:
                log.error("uplink: failed to persist report %s: %s", jid, e)
        log.info("uplink: stored report %s (%d pending)", jid, self.pending_count())
        return True

    def mark_sent(self, job_ids: list[str]) -> None:
        with self._lock:
            changed = False
            for jid in job_ids:
                rec = self._reports.get(str(jid))
                if rec is not None and not rec.get("sent"):
                    rec["sent"] = True
                    changed = True
            if changed:
                try:
                    self._rewrite()
                except OSError as e:
                    log.error("uplink: failed to persist sent flags: %s", e)

    def all_reports(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._reports.values()]

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._reports.values() if not r.get("sent"))

    def _unsent(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._reports.values() if not r.get("sent")]

    # ---- flushing
    @staticmethod
    async def _reachable(url: str) -> bool:
        hp = _host_port(url)
        if hp is None:
            return False
        try:
            _, writer = await asyncio.wait_for(asyncio.open_connection(*hp), CONNECT_TIMEOUT_S)
        except (OSError, asyncio.TimeoutError):
            return False
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), CONNECT_TIMEOUT_S)
        except (OSError, asyncio.TimeoutError):
            pass
        return True

    async def flush_once(self) -> list[str]:
        """Try to deliver unsent reports to the first reachable sink. Returns delivered job_ids."""
        if not self.cfg.uplink_urls or not self.is_leader():
            return []
        pending = self._unsent()
        if not pending:
            return []
        sink = None
        for url in self.cfg.uplink_urls:
            if await self._reachable(url):
                sink = url
                break
        if sink is None:
            log.debug("uplink: no sink reachable (%d pending)", len(pending))
            return []
        endpoint = _sink_endpoint(sink)
        sent: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=POST_TIMEOUT_S) as client:
                for rec in pending:
                    if not self.is_leader():
                        break
                    body = {k: v for k, v in rec.items() if k != "sent"}
                    try:
                        r = await client.post(endpoint, json=body)
                    except httpx.HTTPError as e:
                        log.warning("uplink: POST %s to %s failed: %r", rec["job_id"], endpoint, e)
                        break
                    if 200 <= r.status_code < 300:
                        sent.append(rec["job_id"])
                    else:
                        log.warning("uplink: sink %s rejected %s: HTTP %d", endpoint, rec["job_id"],
                                    r.status_code)
        finally:
            if sent:
                self.mark_sent(sent)
        if sent:
            log.info("uplink: flushed %d report(s) to %s (%d pending): %s", len(sent), endpoint,
                     self.pending_count(), ", ".join(sent))
            await self.broadcast("report.sent", {"job_ids": sent})
        return sent

    async def run(self) -> None:
        interval = max(0.1, float(self.cfg.uplink_interval))
        while True:
            try:
                await self.flush_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("uplink: flush cycle error: %r", e)
            await asyncio.sleep(interval)
