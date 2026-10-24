# SwarmMind protocol v1

The constants and dataclasses live in `swarmmind/protocol.py`. Node configuration lives in `swarmmind/config.py` (`NodeConfig`), and the behavior is implemented in `transport.py`, `election.py`, `jobs.py`, `tasks.py`, `topics.py` and `agent.py`. If this doc and the code disagree, treat it as a bug and fix one or the other.

All traffic stays on the LAN. No node ever contacts a cloud service, and no node ever calls another node's LLM: each node runs inference only on its own local backend (Ollama or LiteRT-LM).

Contents:
1. [Layers](#1-layers)
2. [Message types](#2-message-types)
3. [Leader election](#3-leader-election)
4. [Jobs, streaming and mid-sentence handoff](#4-jobs-streaming-and-mid-sentence-handoff)
5. [Store-and-forward reports](#5-store-and-forward-reports)
6. [Capabilities & task bus](#6-capabilities--task-bus)
7. [Topics (streams), choosing a primitive, phone teleop](#7-topics-streams)
8. [Timing budget](#8-timing-budget)
9. [Failure modes](#9-failure-modes)

---

## 1. Layers

```
            clients / dashboard / teleop
                   |  HTTP + WS /ws/ui, /ws/topics   (c) client API, any node
   +---------------+----------------+
   |  node A  <== WS /ws/peer ==>  node B  ...        (b) peer bus, full mesh, TCP
   |     \___ UDP :50000 heartbeats ___/              (a) discovery + liveness
   +--------------------------------+
        each node -> its own local LLM only (127.0.0.1)
```

### (a) UDP heartbeat / discovery (port 50000)

- Every node sends one heartbeat every `heartbeat_interval` = **200 ms**. It goes to the broadcast address (`broadcast_addr`, default `255.255.255.255`, or `""` to disable broadcast) and is also unicast to every `static_peers` entry (`"host:http_port[:udp_port]"`, where udp_port defaults to 50000).
- Each heartbeat is one JSON object per datagram (`Heartbeat.to_bytes()`) with `"type": "hb"`. Receivers ignore datagrams whose `node_id` is their own.
- A peer is **dead** after `peer_timeout` = **600 ms** with no heartbeat from it. That is 3 missed beats.
- A node counts itself as **healthy** when its own LLM answered a health check within the last `health_window` = **2 s**. It runs that check every `health_interval` = 0.5 s, with a 1.5 s timeout. For Ollama, the check is `GET /api/tags` succeeding with the model listed. For LiteRT, it is the engine being loaded. A `backend: none` node is never healthy.

| Field | Type | Meaning |
|---|---|---|
| `node_id` | str | unique node name |
| `rank` | int | election priority (higher wins) |
| `ip` | str | sender's LAN IP (`host_ip`); peers dial `ws://ip:http_port/ws/peer` |
| `http_port` | int | sender's HTTP/WS port |
| `model` | str | display name of the local model |
| `vision_mode` | str | effective vision mode after fallback: `native`, `coral`, `cpu` or `none` |
| `healthy` | bool | own LLM answered within `health_window` |
| `leader_id` | str or null | who the sender believes is leader (itself if it claims leadership) |
| `term` | int | the sender's current term |
| `active_job_id` | str or null | a job the sender is streaming right now as leader |
| `seq` | int | per-sender counter, incremented on every heartbeat |
| `capabilities` | list[str] | section 6.1 |
| `tps` | float | tokens/s of the sender's last or current stream |
| `boot_id` | str | random string, new on every process start |
| `v` | int | protocol version (1) |
| `type` | str | always `"hb"` |

### (b) Peer bus (WebSocket `/ws/peer`, full mesh)

- **Connections.** For each alive peer, a node opens exactly **one outbound** WebSocket to `ws://<ip>:<http_port>/ws/peer`. A node **sends only on its outbound connections** and **receives only on inbound connections**. That makes every link one-way, so each message travels exactly one hop from its source. There is no relaying, so loops are impossible.
- **Broadcast** means "send on every outbound connection". **Unicast** means sending on the outbound connection to one peer. Only `sync.req` and `sync.resp` are unicast. Everything else is broadcast, and messages for one node say so in the body (for example `task.assign.assignee`).
- **Reconnect.** When a peer dies, its outbound connection is closed. When the peer comes back (heartbeats resume), the node dials it again with backoff.
- **Envelope** (`protocol.Envelope`). Every bus message has this form:

```json
{"v":1, "type":"job.token", "src":"laptop", "term":7, "mid":"laptop:1729760000123456789:42", "ts":1729760000.12, "body":{}}
```

| Field | Meaning |
|---|---|
| `v` | protocol version. Receivers drop envelopes with an unknown `v`. |
| `type` | `<domain>.<verb>`. Receivers **ignore unknown types** (logged at debug), which lets new domains roll out node by node. |
| `src` | sending node_id |
| `term` | the sender's current term (used by the leader-only filter in section 3) |
| `mid` | globally unique message id (`src:time_ns:counter`) |
| `ts` | sender wall clock in seconds. Display only, never used for timing logic. |
| `body` | type-specific (section 2) |

- **Dedupe, two layers:**
  1. **Transport:** drop an envelope whose `mid` was already seen.
  2. **App:** each domain dedupes by its own keys:
     - `(job_id, seq)` for tokens
     - `job_id` for jobs and reports
     - `(job_id, at_seq, to_node)` for handoffs
     - `task_id` plus `attempt`/`updated_ts` for tasks
     - `(origin, topic, seq)` for topic samples

     This catches duplicates that arrive with different mids, for example a replay from `sync.resp`.
- **Ordering.** Each outbound connection is one TCP stream, so messages from one sender arrive in order. Messages from different senders have no ordering guarantee.
- **Catch-up (sync).** When a node's outbound link to a peer comes up, it unicasts `sync.req`. The link comes up on first contact, after a reconnect, and after the peer restarts with a new `boot_id`. The peer answers with `sync.resp` (full job, report and task state), and the requester **merges** it (rules in section 2). This is how a rebooted node, or one that missed messages while disconnected, catches up. Messages sent while a link was down are never re-sent individually; sync covers them.

### (c) HTTP API (clients, dashboard). Every node serves it on `http_port`.

| Method and path | Request | Response |
|---|---|---|
| `GET /` | | dashboard (`dashboard/index.html`) |
| `GET /teleop` | | phone teleop page (publishes topics) |
| `GET /peers` | | `{"self", "leader_id", "term", "peers": [Node]}`. Includes self. |
| `GET /state` | | same body as the `ui.snapshot` message |
| `GET /frame.jpg` | | latest camera frame this node has seen (404 if none) |
| `POST /ask` | `{"question": str}` | `{"job_id", "leader_id"}`. Leader only. Followers forward over HTTP (header-marked, one hop). 503 if there is no leader or it is unreachable. |
| `POST /task` | `{"capability": str, "args": {}, "task_id"?: str}` | `{"task_id", "assignee", "state"}`. Leader only, forwarded like `/ask`. Passing your own `task_id` makes resubmits idempotent. |
| `POST /task/{task_id}/cancel` | | `{"cancelled": bool}`. Leader only, forwarded. |
| `GET /tasks` | | `{"tasks": [Task]}` (latest 50) |
| `POST /pub/{topic}` | any JSON (`data`) | the published sample (section 7) |
| `GET /topics` | | `{"topics": {topic: latest sample}}` |
| `WS /ws/topics?sub=a.,b.` | send `{"topic", "data"}` to publish | receive samples whose topic starts with a subscribed prefix |
| `WS /ws/ui` | (keepalives only) | server push `{"type", "body"}`: `ui.snapshot` on connect, `ui.nodes` every 500 ms, plus `job.*`, `report.*`, `task.*`, `event` as they happen |
| `WS /ws/peer` | | peer bus (above) |

`Node` = `{node_id, rank, ip, http_port, model, vision_mode, alive, healthy, is_leader, tps, capabilities, active_job_id, last_seen_ms}`.

The UI may connect to **any** node: every node holds the full state. If that node dies, the dashboard reconnects to another node from the last `ui.nodes` within 1 s, and the snapshot it receives on connect restores its view.

---

## 2. Message types

L = in `LEADER_ONLY`: accepted only from the current leader at the current term (section 3).

| Type | L | Direction | Body | Receiver action |
|---|---|---|---|---|
| `hb` | | any to all (UDP) | Heartbeat fields (above) | update the peer table and liveness |
| `sync.req` | | any to one peer (unicast) | `{}` | reply with `sync.resp` |
| `sync.resp` | | peer to requester (unicast) | `{"jobs": [Job full], "reports": [Report], "tasks": [Task], "plans": [Plan]}` | merge (rules below) |
| `job.new` | L | leader to all | Job, full form (includes `image_b64`) | store if `job_id` is new |
| `job.token` | L | leader to all | `{"job_id", "seq", "token", "node_id", "term"}` | store if `(job_id, seq)` is new. `seq` starts at 0. |
| `job.handoff` | L | new leader to all, **before** its first token | `{"job_id", "from_node", "to_node", "at_seq", "gap_ms", "vision_note", "term"}` | drop the job's tokens with `seq >= at_seq`, record the handoff |
| `job.done` | L | leader to all | `{"job_id", "final_seq", "answer", "term", "nodes_used", "done_ts"}` | state `done` |
| `report.new` | L | leader to all | Report | store (dedupe by `job_id`) |
| `report.sent` | L | leader to all | `{"job_ids": [str]}` | mark those reports as sent |
| `task.submit` | | any to all (only the leader acts) | `{"capability", "args", "task_id"?}` | the leader creates and assigns the task (same as `POST /task`) |
| `task.assign` | L | leader to all | Task (`state: "assigned"`, `assignee`, `attempt`) | everyone merges. Only `assignee` runs it. |
| `task.progress` | | assignee to all | Task (`state: "running"`, `progress`) | merge |
| `task.result` | | assignee (or leader, on give-up) to all | Task (`state: "done"` or `"failed"`, `result`, `error`) | merge. Ignored if its `attempt` is older than the known one. |
| `task.cancel` | L | leader to all | Task (`state: "cancelled"`) | merge. The assignee cancels its running handler **now**. |
| `plan.new` | L | leader to all | Plan | store if `plan_id` is new |
| `plan.update` | L | leader to all | Plan (every state or step transition) | merge by `(term, rev)` |
| `topic.pub` | | any to all | sample `{"topic", "data", "origin", "seq", "ts"}` | keep if `seq` is newer for `(origin, topic)`, then deliver to subscribers |
| `event` | | any to all | `{"ts", "level", "node_id", "term", "msg"}` | append to the event log (dashboard) |

Data shapes:

- **Job (full)**: `{job_id, question, model, vision_mode, vision_text, has_image, image_b64, state: "streaming"|"done", node_id, term, tokens: [{seq, token, node_id}], handoffs: [handoff body], started_ts, done_ts, final_seq, nodes_used}`.
  - The UI form omits `image_b64`; the UI fetches `/frame.jpg`.
  - `job.new` on `/ws/ui` carries only the metadata (no tokens).
- **Job `kind`**: `"ask"` (default) | `"plan"` | `"verify"`, plus `system` (the system prompt it was started with, reused on resume) and `context` (extra user-turn text, e.g. the object list). Old nodes that don't send these get the defaults.
- **Plan** (robot cell, [ROBOT_CELL.md](ROBOT_CELL.md)): `{plan_id, goal, state: planning|validating|running|done|failed|cancelled, step, job_id, leader, term, rev, steps: [{op, object|target|check, state, task_id, job_id, attempt, result}], errors}`.
  - Task ids are deterministic: `<plan>-s<i>-a<attempt>`. Planner job ids: `<plan>-p<attempt>`, `<plan>-v<i>-a<attempt>`.
  - A new leader resumes running plans. It waits on a step's existing task (never resubmits it) and submits steps that have none. A planning job the old leader didn't finish is continued by the job layer (mid-sentence handoff).
- **Report**: `{job_id, question, answer, nodes_used, handoffs, vision, timestamp}`. Stored with `sent: bool`.
- **Task**: `{task_id, capability, args, state, assignee, attempt, result, error, progress, created_ts, updated_ts}`.
  - `state` is one of `queued | assigned | running | done | failed | cancelled`.
  - `done`, `failed` and `cancelled` are terminal.

Merge rules (`sync.resp`, and all `task.*` messages):

| Item | Rule |
|---|---|
| Job unknown locally | add it |
| Job known | apply remote handoffs not seen yet (each one truncates at its `at_seq`), then add remote tokens for seqs not held. `done` beats `streaming`. |
| Report | union by `job_id`. `sent: true` beats `false`. |
| Task | the incoming copy replaces the local one if its `attempt` is higher, or if `updated_ts` is the same or newer |
| Plan | the copy with the higher `(term, rev)` wins |

---

## 3. Leader election

Bully algorithm with terms (`election.py`).

**Definitions**

- `eligible(n)` = `n` is alive (heartbeat within 600 ms, or `n` is self) **and** `n.healthy`.
- `best` = the eligible node with the maximum `(rank, node_id)` tuple. Compare `rank` first, then `node_id` as a string.
- `max_term_seen` = the highest `term` seen in any heartbeat or accepted envelope.
- Every node tracks `(leader_id, term)`. Logs carry both on every line.
- `stable(n)` = `n` has been continuously alive and healthy for `takeover_stable_s` = 3 s.

**Rules**

1. **Startup grace.** For the first `startup_grace_s` = 1 s after boot, a node only listens. It learns the current leader and `max_term_seen` from heartbeats and never claims during this window.
2. **Only the winner self-promotes.** A node claims leadership only if it is `best`. It sets `term = max_term_seen + 1` and `leader_id = self`, and announces both in its next heartbeat. Non-winners never claim; they wait for the winner's heartbeat.
3. **Adoption.** A claimant is a healthy, alive node whose heartbeat has `leader_id == node_id`. A follower adopts the claimant with the **highest term**. If two claimants share a term, the one with the higher `(rank, node_id)` wins. A leader that sees a better claimant (by `(term, rank, node_id)`) **steps down** and follows it.
4. **Failover (no wait).** The leader is dropped when it is dead, unhealthy, or abdicated (its heartbeat no longer names itself). With no leader, the best eligible node claims **immediately**. Among the eligible nodes, the stable ones are preferred, so a node that is flapping or has just come back does not grab leadership. If none is stable, the best eligible node claims anyway.
5. **Preemption (hysteresis).** A higher-`(rank, node_id)` node takes over from an alive, healthy leader only when all of the following hold:
   - it is `best`,
   - it has been stable itself for **3 s**, and
   - the leader's latest heartbeat shows `active_job_id == null` and it has no unfinished job of its own.

   **A streaming job is never preempted.** The returning laptop waits for the Pi to finish the current answer.
6. **Unhealthy self.** A leader whose own LLM turns unhealthy abdicates: it clears `leader_id`, and failover (rule 4) follows.
7. **Leader-only filter.** For an envelope whose `type` is in `LEADER_ONLY`:
   - `env.term < term`: **drop** it. This covers a stale or partitioned old leader.
   - `env.term == term`: accept it only if `env.src == leader_id`, or if no leader is known.
   - `env.term > term`: a newer leader exists and its heartbeat has not arrived yet. Adopt `(src, env.term)` and accept the message.

   Non-leader-only types (`sync.*`, `task.submit`, `task.progress`, `task.result`, `topic.pub`, `event`) are accepted at any term.
8. **Stepping down.** When leadership moves away from a node, it cancels every LLM stream it is producing. Scheduling and uplink flushing happen only while `is_leader`, so they stop too. Its unfinished jobs stay in the store, and the new leader resumes them (section 4).

Split brain is prevented by rules 3 and 7. A partitioned old leader at term T may keep streaming on its side. Once the partition heals, every node is at term T+1 or higher, so its envelopes are dropped. Its own heartbeats then show it the higher term, and it steps down.

---

## 4. Jobs, streaming and mid-sentence handoff

**Every node stores the full job state**: prompt (`question`), vision context (`vision_text`), the frame (`image_b64`) and all tokens. That is why any node can finish any answer.

**Normal flow (leader)**

1. `POST /ask` reaches the leader. A follower forwards it over HTTP.
2. The leader takes the latest camera frame and builds the vision context:
   - `native`: the image goes to the model.
   - `coral` or `cpu`: run the detector and use the text `Detected: person 0.87 (left), ...`.
   - `none`: no vision context.
3. It broadcasts `job.new` (full form, with the frame) and calls `stream_chat(SYSTEM_PROMPT, user, image)`. The user turn is `vision_text + "\n\n" + question`. While the stream runs, its heartbeat carries `active_job_id`.
4. Each yielded token goes out as `job.token`, with `seq` counting up from 0.
5. At the end of the stream it broadcasts `job.done {final_seq, answer = "".join(tokens)}`, then builds the report (section 5).

**Resume after failover (new leader)**

Within **1 s** of becoming leader, the node resumes every job not in state `done`:

1. `at_seq` = the number of **contiguous** tokens it holds, starting at seq 0. If tokens 0..41 and 43 are present, `at_seq = 42`.
2. `prefix` = `"".join(tokens[0:at_seq])`.
3. Vision context. The continuation is text-only. The first branch that applies wins, and `vision_note` records it.

   | Condition | Vision context used | `vision_note` |
   |---|---|---|
   | The job has no image | the job's `vision_text`, if any | `no image` |
   | `vision_text` is set (detector labels) | reuse it | `reused labels` |
   | This node has a detector | run it on the job's frame (`image_b64`) | `re-ran <mode> detector` |
   | None of the above | `"[vision unavailable]"` | `vision unavailable` |
4. `gap_ms` = this node's **local monotonic** time from receiving the job's last token (or `job.new`, if no token had arrived) to the moment its first new token is ready. It is measured on a single clock, so clock skew between nodes is irrelevant.
5. Broadcast `job.handoff {job_id, from_node, to_node: self, at_seq, gap_ms, vision_note, term}` **before** the first new token.
6. Call `stream_continue(SYSTEM_PROMPT, user, prefix)` and emit tokens from `seq = at_seq` onward, then send `job.done`.

`stream_continue` is **raw prompt continuation**. It uses a hand-built Gemma chat template (`inference.build_gemma_prompt`) in which the model turn is pre-filled with `prefix` and has no closing turn marker, so the model carries on with the same sentence. With Ollama this is `/api/generate` with `raw: true`. With LiteRT-LM it is a session with `apply_prompt_template=False` and `run_prefill([prompt])`.

**Receiver rules**

- On `job.handoff`: **truncate**, dropping every token with `seq >= at_seq`. A dying leader may have reached some peers with tokens that others never got, and those tokens may differ from the new leader's continuation. Then record the handoff, deduped by `(at_seq, to_node)`.
- On `job.token`: dedupe by `(job_id, seq)`. Since the leader-only filter has already dropped any token from the dead leader's term, only the new leader's tokens arrive after the handoff.
- The dashboard renders the handoff inline in the answer, at `at_seq`: `⟶ handed to <to_node> in <gap_ms> ms`.

If the new leader also dies, the handoff fires again; `handoffs` is a list. Leadership goes back to the laptop only after the job finishes, because rule 5 never preempts a streaming job.

---

## 5. Store-and-forward reports

- When a job finishes, the leader builds a **Report** (`{job_id, question, answer, nodes_used, handoffs, vision, timestamp}`) and broadcasts `report.new`. **Every node stores every report**, deduped by `job_id` and persisted under `data/<node_id>/`, so any future leader can flush them.
- **Only the leader flushes.** Every `uplink_interval` (5 s), if `uplink_urls` is non-empty, the leader:
  1. finds the first sink in the list that is reachable on the LAN,
  2. `POST`s each unsent report as JSON to it,
  3. marks the accepted reports as sent, and broadcasts `report.sent {job_ids}`.

  Followers mark the same reports as sent.
- **Exactly once at the sink**: the uplink delivers at least once, and the sink dedupes by `job_id`. `scripts/uplink_sink.py` answers a duplicate with `{"duplicate": true}`. If a leader dies between the POST and `report.sent`, the next leader re-POSTs and the sink ignores the duplicate.
- Sinks are LAN HTTP endpoints only, for example a base-station laptop running `python scripts/uplink_sink.py --port 8800` with `uplink_urls: ["http://<base-ip>:8800/reports"]`. There is no cloud uplink.

---

## 6. Capabilities & task bus

The task bus gives the swarm generic, fault-tolerant work beyond LLM answers: robot arms, grippers, pan-tilt cameras. It reuses the same election, liveness, sync and dedupe as the rest of the protocol. For deciding between jobs, tasks and topics, see 7.1.

### 6.1 Capabilities

Each node advertises `capabilities` in every heartbeat.

| Capability | Added when |
|---|---|
| `llm.text` | backend is not `none` |
| `llm.vision` | the backend accepts images (`gemma4:e4b` on Ollama) |
| `vision.detect` | a Coral or CPU detector is loaded |
| `camera` | `camera_url` is set |
| each entry in config `capabilities` (for example `arm.kinova`) | always. A matching handler must be registered (6.3), or the task fails with `"<node> has no handler for <cap>"`. |

Naming: `<domain>.<thing>`, lowercase, for example `arm.kinova`, `arm.so101`, `gimbal.pantilt`. Built-in demo capabilities in `swarmmind/capabilities/builtin.py`: `echo` and `arm.mock`.

### 6.2 Flow

```
client --POST /task {capability, args}--> any node --(HTTP forward)--> leader
leader: create Task (queued), pick assignee, attempt += 1
leader --task.assign--> all                 (Task: assignee, attempt, state=assigned)
assignee runs handler --task.progress--> all ... --task.result--> all  (done | failed)
leader or operator: POST /task/{id}/cancel  --> leader --task.cancel--> all  (assignee stops now)
```

**Scheduling (leader).** Candidates are the leader itself if it has the capability, plus alive peers whose heartbeat lists it. They are ranked by highest `rank`, then the fewest unfinished tasks already assigned, then highest `node_id`.

Health rules for candidates:
- `healthy` (LLM health) is required only for `llm.*` capabilities.
- For device capabilities, a node removes the capability from its heartbeat while the device is not ready.
- Implementation note: `tasks.py` `_candidates` currently requires `hb.healthy` for every peer. A `backend: none` device node (never LLM-healthy) therefore only receives tasks when it is the leader itself. Until that is fixed, run arm controllers on a node that has an LLM (Legion, MacBook, Thor).

If there is no candidate, the task stays `queued`, and the leader retries every tick (50 ms).

**Reassignment.** On each tick, the leader re-sends `task.assign` with `attempt + 1` for any unfinished task whose assignee is dead. It prefers a different capable node, and falls back to the same node once it is back. After `MAX_ATTEMPTS` = 3, the task ends as `failed` ("gave up after 3 attempts"). A new leader runs the same watchdog over the synced task table. Tasks on alive assignees are left alone, because `task.progress` and `task.result` are not leader-only and survive a leader change.

**Delivery is at-least-once.** A `task_id` may run more than once, on different nodes or attempts. **Handlers must be idempotent by `task_id`**, which they receive as `args["_task_id"]`:
- Cache results by `task_id`, as `arm.mock` does.
- Express moves as absolute goals ("go to pose P"), not deltas ("move 5 cm").

A node never runs two copies of the same task at once, and results from superseded attempts are ignored.

### 6.3 Handler API

Handler modules live in `swarmmind/capabilities/*.py`. At startup, the agent imports **every** module in that package, and importing a module registers its handlers. A module that fails to import is logged and skipped, without taking the node down. A node *runs* a handler only if its config lists the capability. Config `capabilities` without a handler are logged at startup.

```python
from swarmmind.tasks import register_capability

@register_capability("arm.so101")
async def handler(args: dict, progress) -> dict:
    # args:     the "args" from POST /task, plus "_task_id"
    # progress: await progress({...}) -> broadcasts task.progress (shown in every dashboard)
    # return:   JSON-serializable -> task.result {state: "done", result}
    # raise:    any exception     -> task.result {state: "failed", error: "Type: msg"}
    ...

async def setup(node) -> None:
    # optional; called once at startup on EVERY node (whether or not it lists the capability),
    # so return early unless this node owns the device
    if "arm.so101" not in node.cfg.capabilities:
        return
    settings = node.cfg.capability_config.get("arm.so101", {})   # from the node YAML
    ...  # open the serial port, subscribe to topics, start a telemetry loop
```

The objects `setup` can use:

| Name | Use |
|---|---|
| `node.cfg` | `NodeConfig`, including `capability_config` |
| `node.election.term`, `node.election.leader_id`, `node.election.is_leader` | term and leader checks |
| `await node.publish(topic, data)` | publish a topic sample (section 7) |
| `node.event(msg, level=...)` | dashboard event log |

- Handlers run as asyncio tasks on the assignee. Blocking SDK and serial calls go through `await asyncio.to_thread(...)`.
- **Cancellation is the e-stop.** On `task.cancel`, or when the node shuts down, the handler's asyncio task is cancelled, and `asyncio.CancelledError` is raised at its next `await`. Arm handlers must:
  - `await` (a progress call or a short `asyncio.sleep`) between motion segments, so a cancel lands within one segment,
  - catch `CancelledError`, command the arm to stop or hold torque, and then **re-raise**,
  - never block the event loop for longer than one segment.

### 6.4 Worked example: the SO-101 robot cell ("robot building robot")

Hardware:
- A LeRobot **SO-101** arm (Feetech STS3215 servos on one USB serial bus), plugged into **any** node.
- USB webcams watching the workspace.
- A phone as the teleop controller.
- Gemma 4 on the leader (Thor when it is up) plans the assembly steps.

| Primitive | Used for |
|---|---|
| job | Gemma reads the camera frame plus the assembly goal and answers with a JSON plan |
| task `arm.so101` | one discrete step: `{action: "pick", part: "M3 screw", from: "tray A", to: "slot 2"}` |
| topic `arm.so101.state` | joint positions published by the arm node at about 20 Hz |
| topic `teleop.so101` | phone orientation at 30-50 Hz for manual control |

**1. Config.** Add these lines to the YAML of the machine the arm's USB cable is plugged into, for example a MacBook:

```yaml
capabilities: ["arm.so101"]
capability_config:
  arm.so101: {port: "/dev/tty.usbmodem5A460829821", mock: false}   # Linux: /dev/ttyACM0, Windows: COM5
```

**2. Handler module** (`swarmmind/capabilities/so101.py`, imported automatically):

```python
import asyncio, time
from swarmmind.tasks import register_capability
from swarmmind.topics import subscribe

BOUNDS = {"x": (0.10, 0.35), "y": (-0.20, 0.20), "z": (0.01, 0.25)}   # metres, this cell
STATIONS = {"tray A": (0.20, -0.12, 0.03), "tray B": (0.20, 0.12, 0.03),
            "slot 1": (0.28, -0.05, 0.05), "slot 2": (0.28, 0.05, 0.05)}
TELEOP_DEADMAN_S = 0.3
_arm = None                      # your driver around the Feetech bus (e.g. via LeRobot)
_node = None
_done: dict[str, dict] = {}      # idempotency cache: task_id -> result
_teleop = {"last": 0.0}

def _check(target):
    for axis, v in zip("xyz", target):
        lo, hi = BOUNDS[axis]
        if not lo <= v <= hi:
            raise ValueError(f"{axis}={v} outside [{lo}, {hi}]")      # fail, never clamp

async def _move(target, progress, term):
    _check(target)
    for pct, waypoint in _arm.plan_cartesian(target, steps=10):      # short segments
        if _node.election.term != term or _node.election.leader_id is None:
            raise RuntimeError("leader changed mid-move; stopping")   # term check
        await asyncio.to_thread(_arm.goto, waypoint)
        await progress({"step": "moving", "pct": pct})               # cancel lands here

@register_capability("arm.so101")
async def so101(args: dict, progress) -> dict:
    tid = args["_task_id"]
    if tid in _done:                                   # redelivered: never repeat a pick
        return _done[tid]
    term = _node.election.term
    try:
        if args["action"] == "pick":
            await _move(STATIONS[args["from"]], progress, term)
            await asyncio.to_thread(_arm.grip, True)
            await _move(STATIONS[args["to"]], progress, term)
            await asyncio.to_thread(_arm.grip, False)
        elif args["action"] == "home":
            await _move(_arm.HOME, progress, term)
        else:
            raise ValueError(f"unknown action {args['action']!r}")
    except BaseException:                              # cancel (e-stop), term change, error
        await asyncio.to_thread(_arm.hold)             # hold position, torque on
        raise
    _done[tid] = {"action": args["action"], "part": args.get("part"), "at": args.get("to")}
    return _done[tid]

async def _on_teleop(sample):                          # ~30-50 Hz from the phone
    if _node.election.leader_id is None:
        return
    target = _arm.from_orientation(sample["data"])     # map alpha/beta/gamma to a pose
    try:
        _check(target)
    except ValueError:
        return                                         # out of bounds: ignore the sample
    _teleop["last"] = time.monotonic()
    await asyncio.to_thread(_arm.goto, target)

async def _loops():
    while True:
        await asyncio.sleep(0.05)
        if _teleop["last"] and time.monotonic() - _teleop["last"] > TELEOP_DEADMAN_S:
            _teleop["last"] = 0.0
            await asyncio.to_thread(_arm.hold)          # deadman: no sample for 300 ms
        await _node.publish("arm.so101.state", {"joints": await asyncio.to_thread(_arm.read_joints)})

async def setup(node) -> None:
    global _arm, _node
    if "arm.so101" not in node.cfg.capabilities:
        return
    _node = node
    s = node.cfg.capability_config.get("arm.so101", {})
    _arm = await asyncio.to_thread(open_so101, s.get("port"), s.get("mock", False))
    subscribe("teleop.so101", _on_teleop)
    asyncio.create_task(_loops())
```

`open_so101` and the `_arm` methods (`plan_cartesian`, `goto`, `grip`, `hold`, `read_joints`, `from_orientation`, `HOME`) are your driver layer over the Feetech bus. `builtin.py`'s `arm.mock` shows the same shape with no hardware.

**3. Plan, confirm, execute** (client side; 6.6). Ask Gemma for a JSON plan, show it to the operator, then submit one task per step and wait for each `task.result` before sending the next.

A different arm follows the same pattern: a Kinova node is a config with `capabilities: ["arm.kinova"]` plus `swarmmind/capabilities/kinova.py` registering `@register_capability("arm.kinova")`.

### 6.5 Safety rules for actuators

- **Stop.** `POST /task/{id}/cancel` works on any node. It reaches the leader, which broadcasts `task.cancel`, and the assignee's handler is cancelled at its next `await`. A dashboard stop button calls it for every running arm task.
- **Term checks.** `task.assign` and `task.cancel` are leader-only, so a stale or partitioned leader can neither start nor redirect motion. A handler records `node.election.term` when it starts and stops moving if the term changes, or if `leader_id` becomes `None`, mid-task.
- **Deadman.** Teleop motion stops when no `teleop.so101` sample has arrived for **300 ms**. The handler never keeps executing the last command.
- **Bounds.** Validate every target against workspace limits before moving, and fail the task (or drop the teleop sample) rather than clamp silently.
- **Idempotency.** Cache results by `_task_id`, so a redelivered pick never runs twice.
- **Humans decide.** Gemma *proposes* steps, and a person confirms the plan before tasks are submitted.

### 6.6 Chaining: LLM plan to arm tasks

The swarm does not chain automatically. A client (script or dashboard button) asks for structured output, shows it to the operator, and submits the steps:

```python
import asyncio, json, httpx, websockets

NODE = "192.168.1.30:8700"   # any node

async def wait_for(ui, kind, key, value):
    async for raw in ui:                 # survives handoff/failover: any node relays it
        msg = json.loads(raw)
        if msg["type"] == kind and msg["body"].get(key) == value:
            return msg["body"]

async def main():
    async with httpx.AsyncClient(timeout=10) as http, websockets.connect(f"ws://{NODE}/ws/ui") as ui:
        q = ('Goal: put the M3 screw from tray A into slot 2. Look at the workspace. Reply with ONLY '
             'JSON {"steps": [{"action": "pick", "part": str, "from": str, "to": str}]}.')
        job_id = (await http.post(f"http://{NODE}/ask", json={"question": q})).json()["job_id"]
        text = (await wait_for(ui, "job.done", "job_id", job_id))["answer"]
        plan = json.loads(text[text.index("{"): text.rindex("}") + 1])["steps"]
        print(json.dumps(plan, indent=2))
        if input("Execute this plan? [y/N] ").lower() != "y":
            return
        for i, step in enumerate(plan):
            r = (await http.post(f"http://{NODE}/task", json={
                "capability": "arm.so101", "args": step,
                "task_id": f"{job_id}-{i}"})).json()        # stable id: resubmits are idempotent
            res = await wait_for(ui, "task.result", "task_id", r["task_id"])
            if res["state"] != "done":
                print("step failed:", res["error"]); break

asyncio.run(main())
```

If the node the client is connected to dies, reconnect to another node from `GET /peers`. Every node holds the jobs and tasks (`GET /state`, `GET /tasks`).

---

## 7. Topics (streams)

Topics are for high-rate, lossy, latest-value-wins streams: phone IMU to arm teleop, arm joint telemetry, per-frame detections. They are not leader-gated, not persisted and not synced. Each sample fans out over the full-mesh peer bus as `topic.pub`.

- **Names** are dotted, for example `imu.phone1`, `teleop.so101`, `arm.so101.state`.
- **Sample**: `{topic, data, origin, seq, ts}`. `seq` is microseconds since the epoch, strictly increasing per origin, so it keeps increasing across restarts.
- **Receivers** drop a sample whose `seq` is not newer than the last one seen for `(origin, topic)`. Reordering can never move an arm backwards.
- **Publish**:
  - `POST /pub/{topic}` (JSON body = `data`), or
  - send `{"topic", "data"}` on `WS /ws/topics`, or
  - call `await node.publish(topic, data)` in-process.
- **Subscribe**:
  - in-process (capability modules): `from swarmmind.topics import subscribe; subscribe("teleop.so101", async_cb)`. `async_cb(sample)` is called for every topic that starts with the prefix.
  - remote clients: `WS /ws/topics?sub=prefix1,prefix2` (bidirectional: publish on the same socket).
  - latest value per topic: `GET /topics`.

### 7.1 Choosing a primitive

| Primitive | Use when | Delivery | Example |
|---|---|---|---|
| **job** | an LLM answer, streamed token by token | leader-only, mid-sentence handoff on leader death, report on completion | "plan the assembly steps" |
| **task** | a discrete command with exactly one executor | leader assigns, reassigned on assignee death, at-least-once (idempotent by `task_id`), cancellable | "pick part 3 and place it in slot B" |
| **topic** | a continuous stream where only the newest value matters | best effort, latest wins, no persistence | IMU at 50 Hz, joint states, detections per frame |

### 7.2 Phone teleop and the IMU caveat

`GET /teleop` serves a phone page that publishes `teleop.so101` (device orientation) at about 30-50 Hz over `/ws/topics`. The arm node subscribes to it (6.4).

Browsers expose `DeviceOrientationEvent` and `DeviceMotionEvent` only in a **secure context** (HTTPS or `localhost`). A phone opening `http://192.168.1.30:8700/teleop` gets no sensor data. Options that need no internet:

| Option | How |
|---|---|
| (a) USB + adb (Android) | `adb reverse tcp:8700 tcp:8700`, then open `http://localhost:8700/teleop` on the phone. `localhost` is a secure context. |
| (b) HTTPS with a self-signed cert | Generate a cert offline (`openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 30 -subj /CN=swarmmind`). Serve the node with uvicorn's `ssl_keyfile` and `ssl_certfile`, which needs those two options passed to `uvicorn.Config` in `agent.py`. Then accept the certificate warning on the phone. |
| (c) Chrome flag (Android) | In `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, add `http://<node-ip>:8700` and relaunch Chrome. |

iOS Safari also requires `DeviceOrientationEvent.requestPermission()`, called from a user gesture (a button tap) on the page.

---

## 8. Timing budget

| Item | Value | Config key |
|---|---|---|
| Heartbeat interval | 200 ms | `heartbeat_interval` |
| Peer declared dead | 600 ms without a heartbeat | `peer_timeout` |
| Health check / window | every 500 ms (1.5 s timeout) / must have succeeded within 2 s | `health_interval` / `health_window` |
| Election after detection | ≤ ~100 ms (local decision on the next tick, announced by the next heartbeat or first leader-only envelope) | |
| Startup grace | 1 s listen-only | `startup_grace_s` |
| Preemption stability | 3 s stable, plus leader idle | `takeover_stable_s` |
| Resume unfinished jobs | ≤ 1 s after becoming leader | |
| Handoff gap (`gap_ms`) | detection (≤ 600 ms) + election (≤ ~100 ms) + continuation prefill and first token (backend-dependent; longest on the Pi) | |
| Task watchdog tick | 50 ms | |
| Teleop rate / deadman | 30-50 Hz / stop after 300 ms without a sample | |
| `ui.nodes` push | every 500 ms | |
| Dashboard reconnect to another node | ≤ 1 s | |
| Uplink flush attempt | every 5 s (leader) | `uplink_interval` |

---

## 9. Failure modes

| Failure | Detection | Behavior |
|---|---|---|
| **Leader dies mid-stream** (power, cable, crash) | No heartbeat for 600 ms | `best` claims term+1 immediately, then sends `job.handoff` (truncate at `at_seq`) and continues the same sentence via raw continuation. The UI shows `⟶ handed to pi in N ms`. |
| **Follower dies** | No heartbeat for 600 ms | Removed from candidates, and its outbound link closes. Tasks assigned to it are reassigned (`attempt + 1`). Jobs are unaffected. |
| **Partition / split brain** | Two claimants, or an envelope with a stale term | Each side may elect its own leader. On heal, the higher term wins (ties go to the higher `(rank, node_id)`). The old leader's envelopes are dropped by the leader-only filter, and it steps down. `sync` merges job and report state. Reports dedupe by `job_id`. |
| **LLM hangs** (Ollama stuck, LiteRT wedged) | Health check fails for > 2 s, so `healthy=false` | A leader abdicates (rule 6), and failover resumes its job elsewhere. An unhealthy node is never elected. |
| **Flapping node** (loose cable, WiFi drops) | Alive/dead toggles, stability timer resets | Failover away from it is immediate. Failback needs 3 s of stability **and** an idle leader, and failover elections prefer stable nodes. A flapper can't steal a stream or cause leadership ping-pong. |
| **Reboot catch-up** | Outbound link to the restarted peer comes up (new `boot_id`) | `sync.req` / `sync.resp` gives the rebooted node all jobs, reports and tasks. The 1 s startup grace teaches it the current term before it may claim. |
| **Duplicate or late messages** | | Dropped by `mid` (transport), then by app keys (section 1b). |
| **Base station absent** | POST fails | Reports stay queued on every node. They flush exactly once when a sink becomes reachable, even if the leader changes in between. |
| **Assignee dies mid-task** | No heartbeat for 600 ms | Reassigned with `attempt + 1` (at most 3 attempts). The handler is idempotent by `task_id`. |
| **No eligible node** | All dead or unhealthy | No leader. `POST /ask` and `POST /task` return 503. Stored state is kept, and the first node to become healthy claims and resumes. |
