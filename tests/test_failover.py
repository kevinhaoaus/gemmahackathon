"""End-to-end failover test on the 3-node simulator (mock LLMs, real processes, real sockets).

1. start sim_a (rank 100), sim_b (50), sim_c (10); sim_a must lead
2. ask a question, wait until tokens are streaming, hard-kill sim_a mid-answer
3. sim_b must resume the SAME answer: no duplicate / missing seq, handoff gap < 1.5 s
4. revive sim_a while sim_b is still streaming: sim_a must NOT retake until the job is done
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.simulate import SimCluster  # noqa: E402


def _get(url: str, path: str) -> dict | None:
    try:
        r = httpx.get(url + path, timeout=1.0)
        return r.json() if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


def _wait(cond, timeout: float, step: float = 0.1):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = cond()
        if v:
            return v
        time.sleep(step)
    return None


def _job(url: str, job_id: str) -> dict | None:
    st = _get(url, "/state")
    if not st:
        return None
    return next((j for j in st["jobs"] if j["job_id"] == job_id), None)


class LeaderWatcher(threading.Thread):
    """Polls /peers on an observer node and records (monotonic_time, leader_id) transitions."""

    def __init__(self, url: str):
        super().__init__(daemon=True)
        self.url = url
        self.changes: list[tuple[float, str | None]] = []
        self.stop_flag = False

    def run(self) -> None:
        last = object()
        while not self.stop_flag:
            p = _get(self.url, "/peers")
            if p is not None and p["leader_id"] != last:
                last = p["leader_id"]
                self.changes.append((time.monotonic(), last))
            time.sleep(0.05)


@pytest.fixture(scope="module")
def cluster():
    c = SimCluster(log_dir=ROOT / "data" / "sim" / "logs_test")
    for d in (ROOT / "data" / "sim").glob("sim_*"):       # fresh report queues
        for f in d.glob("*.jsonl"):
            f.unlink()
    assert c.start(wait=True, timeout=40), "sim nodes did not come up"
    yield c
    c.stop()


def test_failover_mid_sentence(cluster: SimCluster):
    a, b, cc = cluster.url("a"), cluster.url("b"), cluster.url("c")

    # 1. sim_a leads once everyone is up
    assert _wait(lambda: all((_get(u, "/peers") or {}).get("leader_id") == "sim_a" for u in (a, b, cc)), 20), \
        "sim_a never became leader on all nodes"

    watcher = LeaderWatcher(cc)
    watcher.start()

    # 2. ask via a follower (exercises forwarding), wait for streaming, then pull the plug
    r = httpx.post(b + "/ask", json={"question": "What do you see? Any hazards?"}, timeout=10)
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    assert _wait(lambda: len((_job(b, job_id) or {}).get("tokens", [])) >= 12, 15), "no tokens streamed"
    cluster.kill("a")
    killed_at = time.monotonic()

    # sim_b takes over and resumes
    assert _wait(lambda: (_get(b, "/peers") or {}).get("leader_id") == "sim_b", 3), "sim_b did not take over"
    took_over = time.monotonic() - killed_at

    # 4. bring the old leader back while sim_b is still streaming
    job = _wait(lambda: (j := _job(b, job_id)) and j["handoffs"] and j, 5)
    assert job, "no handoff recorded"
    assert job["state"] == "streaming", "job finished before we could test hysteresis; slow the mocks"
    cluster.revive("a")

    done = _wait(lambda: (j := _job(b, job_id)) and j["state"] == "done" and j, 90)
    assert done, "job never completed after failover"
    done_at = time.monotonic()

    # 3. integrity of the answer on every surviving replica
    for url in (b, cc):
        j = _wait(lambda: (x := _job(url, job_id)) and x["state"] == "done" and x, 5)
        assert j, f"{url} has no completed copy of the job"
        seqs = [t["seq"] for t in j["tokens"]]
        assert len(seqs) == len(set(seqs)), f"duplicate seq on {url}"
        assert seqs == list(range(len(seqs))), f"missing seq on {url}: {seqs}"
        assert j["final_seq"] == len(seqs)
        assert {t["node_id"] for t in j["tokens"]} == {"sim_a", "sim_b"}
    text_b = "".join(t["token"] for t in done["tokens"])
    text_c = "".join(t["token"] for t in _job(cc, job_id)["tokens"])
    assert text_b == text_c, "replicas disagree on the answer"

    h = done["handoffs"][0]
    assert h["from_node"] == "sim_a" and h["to_node"] == "sim_b"
    assert h["gap_ms"] < 1500, f"handoff gap {h['gap_ms']} ms"
    assert took_over < 1.5, f"took {took_over:.2f}s to elect sim_b"

    # the mock continues its canned answer from the prefix word count, so a seamless handoff
    # yields no repeated or skipped words
    words = text_b.split()
    assert not any(words[i:i + 4] == words[i + 4:i + 8] for i in range(len(words) - 8)), "repeated span"

    # sim_a retakes only after the job ended (and after its 3 s stability window)
    assert _wait(lambda: (_get(cc, "/peers") or {}).get("leader_id") == "sim_a", 15), "sim_a never retook leadership"
    watcher.stop_flag = True
    watcher.join(2)
    retakes = [t for t, lid in watcher.changes if lid == "sim_a" and t > killed_at]
    assert retakes, f"no sim_a retake recorded: {watcher.changes}"
    assert retakes[0] >= done_at - 0.3, "sim_a preempted a job that was still streaming"

    # the report exists on every node exactly once
    for url in (cluster.url("b"), cc):
        reps = [rp for rp in (_get(url, "/state") or {}).get("reports", []) if rp["job_id"] == job_id]
        assert len(reps) == 1
        assert reps[0]["answer"] == text_b


def test_task_bus_reassigns(cluster: SimCluster):
    """Generic task bus: an `echo` task submitted via any node completes on a capable node."""
    url = cluster.url("c")
    assert _wait(lambda: (_get(url, "/peers") or {}).get("leader_id"), 10)
    r = httpx.post(url + "/task", json={"capability": "echo", "args": {"hello": "arm"}}, timeout=5)
    assert r.status_code == 200, r.text
    tid = r.json()["task_id"]
    t = _wait(lambda: next((x for x in (_get(url, "/tasks") or {}).get("tasks", [])
                            if x["task_id"] == tid and x["state"] == "done"), None), 10)
    assert t and t["result"] == {"echo": {"hello": "arm"}}
