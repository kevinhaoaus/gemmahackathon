# SwarmMind

Offline swarm of heterogeneous machines on one LAN that runs Gemma 4 and drives robots. Any node can die mid-answer and another finishes the same sentence. No cloud, no internet.

> **Now: LabMate, a research-assistant arm.** A Seeed reBot B601 (USB2CAN on COM5) and an overhead webcam (AC310, AprilTags) both plug into the Windows Legion. Gemma 4 finds items (springs, screwdrivers, files, superglue), clears hazards into a safe tray, works out which experiment the setup is for and which tools are missing, and fetches items. The Thor is the brain; the swarm keeps going if a computer or an arm dies. See [docs/LABMATE.md](docs/LABMATE.md). Earlier pivots: [docs/ROBOT_CELL.md](docs/ROBOT_CELL.md).

## Test plan (real hardware, Windows Legion)

Fastest route to a real pick-and-place is steps 1–7; everything else can wait. Details: [docs/BENCH.md](docs/BENCH.md).

**Before you touch the arm:** 24 V supply on, area clear, a hand on the power switch. Ctrl-C / `disable` turns torque off, so **the arm drops**: support it first. `stop` / SPACE holds position.

| # | Command (from the repo root) | Pass when |
|---|---|---|
| 0 | `.venv\Scripts\pip install motorbridge` | installs (needs internet once; it's the reBot SDK, not a model) |
| 1 | `.venv\Scripts\python scripts\bench.py cam` | ~26 fps, tags 0–3 green, **CALIBRATED** < 2 px. Click a spot and its x/y in cm matches a ruler from tag 0 (x toward tag 3, y toward tag 1). Clear the table, press `b`, then put an object down: it gets a number. `q` quits |
| 2 | `.venv\Scripts\python scripts\bench.py arm-read` | **read-only**: prints every motor's position, velocity and temperature with no errors. Nothing moves. If it fails: power, COM port, baud (see BENCH.md) |
| 3 | `.venv\Scripts\python scripts\bench.py jog-gui` (buttons) or `arm-jog` (typed) | type `ENABLE`; `j1 +5` … `j6 +5` each move the expected joint the expected way (note any reversed joint); `home` works; `stop` holds; `disable` (support the arm) |
| 4 | gripper: jog it fully open, then **Save as OPEN** (`grip set-open`); jog it closed on nothing, then **Save as CLOSED** (`grip set-closed`) | the gripper's open/closed angles are saved; `grip open` / `grip close` work |
| 5 | `.venv\Scripts\python scripts\bench.py calib-base` | jog the gripper tip onto ≥ 3 tag centres (`touch 0`, `touch 1`, `touch 3`), then `solve`: **RMS residual < 5 mm**. Paste the printed `base: [...]` into `configs/laptop.yaml` → `arm.cell.base` |
| 6 | `.venv\Scripts\python scripts\bench.py pick --dry-run` | click an object, then a destination: the plan is drawn on the image, arm targets are printed, and it reports reachable. Nothing moves |
| 7 | `.venv\Scripts\python scripts\bench.py pick --height 0.02 --approach 0.08` | press `y`: the arm goes above the object, descends slowly, grips, lifts, carries, places, retreats. SPACE stops at any time. Start with something light and grippable (a marker), then a screwdriver / spring |

**Gripper width:** pick closes to *(object width from the camera bounding box − 5 mm)* on a linear model (0 cm = closed, 10 cm = open). Tune with `pick --squeeze-mm`. In `jog-gui`, the **width** slider tests the model by hand.

**After pick-and-place works, test the full system:**

| # | Command | Pass when |
|---|---|---|
| 8 | `.venv\Scripts\python -m swarmmind.agent --config configs/legion_camera.yaml`, then open http://127.0.0.1:8700/ | camera-only node (never opens COM5): Cell panel shows the annotated live view and CALIBRATED; Ask "what's on the bench?" and Gemma 4 describes the live frame |
| 9 | same, with `configs/laptop.yaml` (camera **and** arm) | the node advertises `arm.cell` while the reBot is connected; unplug the USB and it disappears from the dashboard within ~2 s, replug and it's back |
| 10 | LabMate panel → **Inspect bench** (table with an open superglue, a screwdriver, a spring, a file) | items are labelled; the open superglue is announced, a 5 s cancel countdown runs, then it's moved to the safe tray (tag 10); the experiment and missing tools are listed; the question appears. Answer it and the report streams |
| 11 | LabMate → **Fetch** "screwdriver" | the screwdriver is brought to the handover spot (tag 11) |
| 12 | Thor joins (`docs/JETSON_AGX_THOR.md`), then pull the Thor's Ethernet mid-task | Thor leads while alive; on pull, the Legion takes over within ~1 s and the arm finishes the task (the arm and camera are on the Legion) |

**Not hardware-tested yet:** the reBot driver (joint signs/offsets assumed to match the URDF; verify in step 3), the SO-101 backup arm, LabMate end to end (one simulator e2e test still fails). Simulator-only tests: `.venv\Scripts\python -m pytest -q tests`.

## Nodes

| Node | Rank | LLM runtime → model | Vision |
|---|---|---|---|
| Jetson AGX Thor | 200 | Ollama → `gemma4:e4b` (copied over LAN) | native (image in prompt) |
| Lenovo Legion 5i, RTX 4070 | 100 | Ollama → `gemma4:e4b` | native |
| MacBook Airs | 60–80 | Ollama (Metal) → `gemma4:e4b` | native |
| Raspberry Pi 5 | 50 | LiteRT-LM → `gemma-4-E4B-it.litertlm` | Coral EdgeTPU → TFLite CPU → none |
| Arm host (any of the above) | — | — | serves `arm.so101` (Feetech serial) |
| Phone | — | not a node: browser client | IMU teleop via `/teleop` |

Highest alive, healthy rank leads. One YAML per node in `configs/`. Adding a machine means adding a file.

## Stack

Python 3.11 · asyncio · FastAPI + uvicorn (HTTP/WS server) · `websockets` (peer client) · httpx (Ollama, camera, uplink) · PyYAML · Pillow/numpy · `litert_lm` (Pi) · pycoral / ai-edge-litert (detector) · vanilla-JS dashboard (no build, no CDN).

## Wire protocol (`swarmmind/protocol.py`, full spec in `docs/PROTOCOL.md`)

```
           UDP :50000  heartbeat every 200 ms (broadcast + unicast to static_peers)
  node ◄──────────────────────────────────────────────────────────► node
       ◄══ WS /ws/peer: full mesh, each node SENDS only on its own outbound link ══►
                Envelope {v, type, src, term, mid, ts, body}
  HTTP :8700  /ask /task /pub/{topic} /peers /state /frame.jpg   WS /ws/ui /ws/topics
```

- **Heartbeat** carries `{node_id, rank, ip, http_port, model, vision_mode, healthy, leader_id, term, active_job_id, capabilities, tps, boot_id, seq}`. A peer with no heartbeat for 600 ms is dead. `healthy` means the node's own LLM answered within 2 s.
- **Election** (`election.py`): bully with terms. The winner claims `term = max_seen + 1`. Leader-only messages carrying an older term are dropped, which prevents split brain. Failover is immediate. A returning higher-rank node retakes leadership only after 3 s of stable health, and never during a streaming job.
- **Catch-up:** every new peer link sends `sync.req` and gets back all jobs, reports and tasks, so a rebooted node rejoins with full state.

## Three primitives on the bus

| Primitive | Use for | Semantics | Code |
|---|---|---|---|
| **Jobs** | Gemma answers | Leader streams `job.token {job_id, seq}` to every node. Every node holds full state. | `jobs.py` |
| **Tasks** | Discrete robot commands ("pick part 3 → slot B") | One executor chosen by capability. Reassigned if it dies (at-least-once, idempotent by `task_id`). `task.cancel` acts as e-stop. | `tasks.py`, `capabilities/*.py` |
| **Topics** | Continuous streams (phone IMU, joint states) | Lossy, latest-value-wins, seq-ordered, never leader-gated | `topics.py` |

## Mid-sentence handoff

1. Leader dies. Peers notice after 600 ms, and the next-ranked node claims the new term.
2. The new leader rebuilds the Gemma 4 chat template by hand, with the model turn pre-filled with the tokens so far:
   `<|turn>system\n…<turn|>\n<|turn>user\n…<turn|>\n<|turn>model\n{tokens so far}`
   It sends this to Ollama `/api/generate` with `raw: true`, or to a LiteRT-LM `Session` with `apply_prompt_template=False` (`run_prefill` then `run_decode_async`).
3. It broadcasts `job.handoff {at_seq, gap_ms}`. Receivers drop any tokens at or after `at_seq`, then dedupe by `(job_id, seq)`.
4. If the new leader can't take images, it re-runs its local detector on the job's frame ("Detected: person 0.87 (left)"). Otherwise the prompt gets `[vision unavailable]`.

## Robots (`capabilities/`)

A node lists `capabilities: ["arm.so101"]` in its YAML and advertises it in its heartbeat. The leader routes `POST /task {capability, args}` to the best alive node that has that capability. Handlers are `@register_capability` async functions; a module may also define `setup(node)` to open serial ports and subscribe to topics. The phone page `/teleop` publishes `teleop.so101` at 30 Hz, and the arm node subscribes. The deadman stops the arm after 300 ms of silence.

## Other modules

`inference.py` (Ollama / LiteRT / null backends, `/api/chat` with `think:false` and images) · `vision.py` (camera: phone `shot.jpg`, USB webcam `usb:0`, `file:`, `mock:`; detector fallback chain) · `uplink.py` (leader flushes the JSONL report queue to LAN sinks, `scripts/uplink_sink.py`, deduped by `job_id`) · `dashboard/index.html` (served by every node; reconnects to any alive peer within 1 s) · `mock_llm.py` (fake Ollama for the simulator).

## Run

```bash
.venv\Scripts\python scripts\simulate.py               # 3 local nodes + mock LLMs; CLI: ask / kill / revive / sick / heal
.venv\Scripts\python scripts\simulate.py --real        # same 3 nodes, but all on the real local gemma4:e4b (real answers)
.venv\Scripts\python -m pytest -q tests                 # failover + task bus tests
python -m swarmmind.agent --config configs/laptop.yaml  # a real node; dashboard at http://<node>:8700/
```

Setup, demo script and troubleshooting: `docs/SETUP.md`. Jetson Thor walkthrough: `docs/JETSON_AGX_THOR.md`. Wire spec: `docs/PROTOCOL.md`.

Every Ollama node loads and pins its model at startup (`keep_alive: -1`), so a standby never pays a model load during handoff. Phones need HTTPS for IMU access: set `ssl_certfile`/`ssl_keyfile` for an extra listener on `:8443`.

## Measured (this repo, Windows)

- Failover to a new leader 0–60 ms after the dead-peer timeout.
- Handoff gap 625 ms on real `gemma4:e4b`, 766–828 ms on the simulator. Both include the 600 ms timeout.
- Topic latency from one node to another 1.5 ms p50.
