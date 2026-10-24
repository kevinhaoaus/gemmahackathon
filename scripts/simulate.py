"""One-machine swarm simulator: 3 mock LLMs + 3 nodes, with an interactive chaos CLI.

    python scripts/simulate.py            (from the repo root)

Logs go to data/sim/logs/<name>.log (nodes: sim_a.., mocks: mock_a..); `tail a` shows them.
Importable for tests:  from scripts.simulate import SimCluster
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:                     # so `python scripts/simulate.py` finds swarmmind
    sys.path.insert(0, str(ROOT))

import atexit  # noqa: E402
import shlex  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import IO, Any  # noqa: E402

import httpx  # noqa: E402


@dataclass(frozen=True)
class SimNode:
    key: str              # a | b | c
    node_id: str
    http_port: int
    mock_port: int
    tps: float
    model: str

    @property
    def config(self) -> str:
        return f"configs/{self.node_id}.yaml"


NODES: dict[str, SimNode] = {
    "a": SimNode("a", "sim_a", 8701, 11501, 12, "gemma4:e4b"),
    "b": SimNode("b", "sim_b", 8702, 11502, 6, "gemma4:e2b"),
    "c": SimNode("c", "sim_c", 8703, 11503, 4, "gemma4:e2b"),
}


def _key(name: str) -> str:
    """Accept 'a', 'sim_a' or 'A'."""
    k = name.strip().lower().removeprefix("sim_")
    if k not in NODES:
        raise KeyError(f"unknown node {name!r} (use a, b or c)")
    return k


def _port_open(port: int, path: str, timeout: float = 0.5) -> bool:
    try:
        return httpx.get(f"http://127.0.0.1:{port}{path}", timeout=timeout).status_code < 500
    except httpx.HTTPError:
        return False


class SimCluster:
    """Owns the 6 child processes. Every child runs with cwd = repo root and its own process
    group / session, so Ctrl+C in the CLI does not reach them; stop() kills them all."""

    def __init__(self, log_dir: str | os.PathLike | None = None, tps_scale: float = 1.0) -> None:
        self.log_dir = Path(log_dir) if log_dir else ROOT / "data" / "sim" / "logs"
        self.tps_scale = tps_scale                # >1 speeds up every mock LLM (handy in tests)
        self.nodes: dict[str, subprocess.Popen[bytes]] = {}
        self.mocks: dict[str, subprocess.Popen[bytes]] = {}
        self._logs: list[IO[bytes]] = []
        atexit.register(self.stop)

    # ---- process plumbing
    def _spawn(self, name: str, args: list[str]) -> subprocess.Popen[bytes]:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        f = open(self.log_dir / f"{name}.log", "ab")
        f.write(f"\n===== {time.strftime('%H:%M:%S')} start {' '.join(args)}\n".encode())
        f.flush()
        self._logs.append(f)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH", "")) if p)
        kw: dict[str, Any] = {}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        return subprocess.Popen([sys.executable, *args], cwd=str(ROOT), env=env, stdout=f,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)

    @staticmethod
    def _kill(proc: subprocess.Popen[bytes] | None) -> None:
        if proc is not None and proc.poll() is None:
            proc.kill()                            # TerminateProcess on Windows, SIGKILL on Linux
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def _start_mock(self, k: str, wait: bool = True) -> None:
        n = NODES[k]
        if self.mock_alive(k):
            return
        self.mocks[k] = self._spawn(f"mock_{k}", [
            "-m", "swarmmind.mock_llm", "--port", str(n.mock_port),
            "--tps", str(n.tps * self.tps_scale), "--model", n.model])
        if wait:
            self._wait(lambda: _port_open(n.mock_port, "/api/tags"), 15)

    def _start_node(self, k: str) -> None:
        if not self.alive(k):
            self.nodes[k] = self._spawn(NODES[k].node_id,
                                        ["-m", "swarmmind.agent", "--config", NODES[k].config])

    @staticmethod
    def _wait(cond: Any, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if cond():
                return True
            time.sleep(0.2)
        return False

    # ---- public API (used by tests/test_failover.py)
    def start(self, wait: bool = True, timeout: float = 30.0) -> bool:
        """Start all mocks (waits for them), then all nodes. If wait, block until every
        node answers GET /peers; returns whether they all came up in time."""
        for k in NODES:
            self._start_mock(k, wait=False)
        self._wait(lambda: all(_port_open(NODES[k].mock_port, "/api/tags") for k in NODES), 15)
        for k in NODES:
            self._start_node(k)
        return self.wait_ready(timeout) if wait else True

    def wait_ready(self, timeout: float = 30.0, names: str | list[str] | None = None) -> bool:
        keys = [_key(n) for n in (names or list(NODES))]
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if all(_port_open(NODES[k].http_port, "/peers") for k in keys):
                return True
            if not any(self.alive(k) for k in keys):
                return False                      # every target process exited: fail fast
            time.sleep(0.2)
        return False

    def kill(self, name: str) -> None:
        """Hard-kill a node process (no graceful shutdown: simulates pulling the plug)."""
        self._kill(self.nodes.pop(_key(name), None))

    def revive(self, name: str, wait: bool = False, timeout: float = 30.0) -> bool:
        k = _key(name)
        self._start_node(k)
        return self.wait_ready(timeout, [k]) if wait else True

    def sick(self, name: str) -> None:
        """Kill this node's mock LLM: the node stays up but its health check fails."""
        self._kill(self.mocks.pop(_key(name), None))

    def heal(self, name: str) -> None:
        self._start_mock(_key(name), wait=True)

    def alive(self, name: str) -> bool:
        p = self.nodes.get(_key(name))
        return p is not None and p.poll() is None

    def mock_alive(self, name: str) -> bool:
        p = self.mocks.get(_key(name))
        return p is not None and p.poll() is None

    def url(self, name: str) -> str:
        return f"http://127.0.0.1:{NODES[_key(name)].http_port}"

    def alive_urls(self) -> list[str]:
        return [self.url(k) for k in NODES if self.alive(k)]

    def log_path(self, name: str) -> Path:
        k = name.strip().lower()
        if k.startswith("mock_"):
            return self.log_dir / f"mock_{_key(k[5:])}.log"
        return self.log_dir / f"{NODES[_key(k)].node_id}.log"

    def stop(self) -> None:
        for procs in (self.nodes, self.mocks):
            for k in list(procs):
                self._kill(procs.pop(k))
        for f in self._logs:
            try:
                f.close()
            except OSError:
                pass
        self._logs.clear()

    def __enter__(self) -> "SimCluster":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


# ---------------------------------------------------------------- interactive CLI
HELP = """commands:
  status                 GET /peers from any alive node
  ask <question>         POST /ask, then stream the answer (Ctrl+C stops watching)
  kill <a|b|c>           hard-kill a node process        revive <a|b|c>   restart it
  sick <a|b|c>           stop that node's mock LLM       heal <a|b|c>     restart it
  tail <a|b|c|mock_a..> [n]   last n log lines (default 20)
  open                   dashboard URLs
  help | quit"""


def _get_json(cluster: SimCluster, path: str) -> tuple[str, Any] | None:
    for base in cluster.alive_urls():
        try:
            r = httpx.get(base + path, timeout=1.5)
            if r.status_code == 200:
                return base, r.json()
        except (httpx.HTTPError, ValueError):
            continue
    return None


def cmd_status(cluster: SimCluster) -> None:
    procs = "  ".join(f"{n.node_id}:{'up' if cluster.alive(k) else 'DOWN'}"
                      f"/llm:{'up' if cluster.mock_alive(k) else 'DOWN'}" for k, n in NODES.items())
    print(f"processes  {procs}")
    got = _get_json(cluster, "/peers")
    if not got:
        print("no node answered /peers")
        return
    base, d = got
    print(f"view from {d.get('self')} ({base}): leader={d.get('leader_id')} term={d.get('term')}")
    print(f"  {'node':8} {'rank':>4} {'alive':5} {'healthy':7} {'lead':4} {'tps':>6}  {'model':12} {'vision':6} job")
    for p in sorted(d.get("peers", []), key=lambda p: -int(p.get("rank", 0))):
        print(f"  {p.get('node_id', '?'):8} {p.get('rank', 0):>4} {str(p.get('alive')):5} "
              f"{str(p.get('healthy')):7} {'*' if p.get('is_leader') else '':4} "
              f"{float(p.get('tps') or 0):6.1f}  {str(p.get('model')):12} "
              f"{str(p.get('vision_mode')):6} {p.get('active_job_id') or '-'}")


def cmd_ask(cluster: SimCluster, question: str, timeout: float = 240.0) -> None:
    job_id = None
    for base in cluster.alive_urls():
        try:
            r = httpx.post(base + "/ask", json={"question": question}, timeout=5)
            if r.status_code == 200:
                d = r.json()
                job_id = d.get("job_id")
                print(f"job {job_id} (leader {d.get('leader_id')}, asked via {base})")
                break
            print(f"{base}: HTTP {r.status_code} {r.text[:200]}")
        except httpx.HTTPError as e:
            print(f"{base}: {e!r}")
    if not job_id:
        print("no node accepted the question")
        return
    toks: dict[int, str] = {}         # seq -> token, after applying the handoff rule
    printed, seen_handoffs = 0, 0
    end = time.monotonic() + timeout
    try:
        while time.monotonic() < end:
            got = _get_json(cluster, "/state")
            job = next((j for j in (got[1].get("jobs") or []) if j.get("job_id") == job_id),
                       None) if got else None
            if job:
                hos = job.get("handoffs") or []
                for h in hos[seen_handoffs:]:
                    at = int(h.get("at_seq", 0))
                    print(f"\n  [handoff {h.get('from_node')} -> {h.get('to_node')} at seq {at}, "
                          f"gap {h.get('gap_ms')} ms]", flush=True)
                    # handoff rule: drop seq >= at_seq, then re-print what survived
                    toks = {s: t for s, t in toks.items() if s < at}
                    printed = 0
                    while printed in toks:
                        printed += 1
                    print("  " + "".join(toks[s] for s in range(printed)), end="", flush=True)
                seen_handoffs = len(hos)
                for t in job.get("tokens") or []:
                    toks.setdefault(int(t["seq"]), t.get("token", ""))
                if printed == 0 and 0 in toks and not seen_handoffs:
                    print("  ", end="")
                while printed in toks:                # print contiguous new tokens only
                    print(toks[printed], end="", flush=True)
                    printed += 1
                if job.get("state") == "done":
                    nodes = sorted({t.get("node_id") for t in job.get("tokens") or []} - {None})
                    print(f"\n  [done: {printed} tokens, nodes {', '.join(nodes)}]")
                    return
            time.sleep(0.3)
        print(f"\n  [stopped watching after {timeout:.0f}s; job {job_id} may still be running]")
    except KeyboardInterrupt:
        print(f"\n  [stopped watching job {job_id}]")


def cmd_tail(cluster: SimCluster, name: str, n: int = 20) -> None:
    path = cluster.log_path(name)
    if not path.exists():
        print(f"no log yet: {path}")
        return
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in deque(f, maxlen=n):
            print(line.rstrip("\n"))


def main() -> None:
    cluster = SimCluster()
    print(f"starting 3 mock LLMs + 3 nodes; logs in {cluster.log_dir}")
    ok = cluster.start(wait=True, timeout=30)
    print("all nodes up" if ok else "WARNING: not every node answered /peers within 30 s "
          "(see `tail a`)")
    cmd_status(cluster)
    print(HELP)
    try:
        while True:
            try:
                line = input("sim> ").strip()
            except EOFError:
                break
            if not line:
                continue
            cmd, _, rest = line.partition(" ")
            cmd, rest = cmd.lower(), rest.strip()
            try:
                if cmd in ("quit", "exit", "q"):
                    break
                elif cmd == "help":
                    print(HELP)
                elif cmd == "status":
                    cmd_status(cluster)
                elif cmd == "ask":
                    cmd_ask(cluster, rest) if rest else print("usage: ask <question>")
                elif cmd in ("kill", "revive", "sick", "heal"):
                    getattr(cluster, cmd)(rest)
                    print(f"{cmd} {NODES[_key(rest)].node_id}: ok")
                elif cmd == "tail":
                    parts = shlex.split(rest) or ["a"]
                    cmd_tail(cluster, parts[0], int(parts[1]) if len(parts) > 1 else 20)
                elif cmd == "open":
                    for k, n in NODES.items():
                        print(f"  {n.node_id}: {cluster.url(k)}/  {'' if cluster.alive(k) else '(down)'}")
                else:
                    print(f"unknown command {cmd!r}; type help")
            except (KeyError, ValueError) as e:
                print(f"error: {e}")
    except KeyboardInterrupt:
        print()
    finally:
        print("stopping all processes...")
        cluster.stop()


if __name__ == "__main__":
    main()
