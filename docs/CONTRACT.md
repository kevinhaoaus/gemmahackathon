# SwarmMind internal contract (for parallel builders)

Read `swarmmind/protocol.py` and `swarmmind/config.py` first — they are fixed. Python 3.11, asyncio, Windows + Linux.
Venv: `.venv/Scripts/python` (Windows). Installed: fastapi, uvicorn, starlette, httpx, websockets, pyyaml, numpy, pillow, pytest, pytest-asyncio, litert_lm, cv2.
**NO internet / cloud calls, and never download models.** Local assets:
- Ollama on this laptop at http://127.0.0.1:11434 with model `gemma4:e4b` (vision capable).
- LiteRT-LM models: `C:\Users\Default.DESKTOP-NUBLEQ1\Desktop\GemmaHackathonOld\GemmaHackathonBasic\models\gemma-4-E4B-it-litert-lm\gemma-4-E4B-it.litertlm` (also `-gpu.litertlm`).
- No Coral / detection .tflite models exist locally; vision code must degrade gracefully when model files are missing.

Logging: use `swarmmind.logutil.get_logger(name)` (core provides it). Every line is formatted with node_id and term automatically.

Run a node: `python -m swarmmind.agent --config configs/X.yaml`. Static peers: `"host:http_port"` or `"host:http_port:udp_port"` (udp defaults to 50000).
Simulator on one machine: each sim node has its own `udp_port` (50001/50002/50003), `broadcast_addr: ""` (disables broadcast), and lists the other two as static peers with their udp ports; `data_dir: data/sim`. Sim nodes: `sim_a` rank 100 http 8701 mock-LLM 11501 @ 12 tok/s model gemma4:e4b vision native camera `mock:`; `sim_b` rank 50 http 8702 mock 11502 @ 6 tok/s model gemma4:e2b vision cpu camera `mock:`; `sim_c` rank 10 http 8703 mock 11503 @ 4 tok/s model gemma4:e2b vision none.

## Module ownership
| File | Owner |
|---|---|
| protocol.py, config.py, logutil.py, transport.py, election.py, jobs.py, tasks.py, agent.py, tests/ | core |
| inference.py, mock_llm.py, scripts/simulate.py, configs/sim_*.yaml | inference builder |
| vision.py, uplink.py, scripts/uplink_sink.py | vision/uplink builder |
| dashboard/index.html | dashboard builder |
| scripts/setup_pi.sh, scripts/setup_windows.ps1, configs/laptop.yaml, configs/pi.yaml, README.md, docs/PROTOCOL.md, requirements.txt | docs/ops builder |

## inference.py
```python
class LLMBackend:                    # base class
    name: str                        # "ollama" | "litert" | "none"
    model: str
    supports_images: bool
    tps: float                       # rolling tokens/sec of the most recent stream
    async def check_health(self) -> bool            # must return within ~1.5 s; ollama = GET /api/tags ok AND model listed
    def stream_chat(self, system: str, user: str, image_b64: str | None) -> AsyncIterator[str]
    def stream_continue(self, system: str, user: str, prefix: str) -> AsyncIterator[str]
    async def aclose(self) -> None
def make_backend(cfg: NodeConfig) -> LLMBackend
def build_gemma_prompt(system: str, user: str, prefix: str, model: str) -> str   # gemma3 vs gemma4 template
```
- Yielded strings are "tokens"; `"".join(tokens)` is the answer text. `prefix` = join of tokens so far.
- stream_continue uses Ollama `/api/generate` with `raw: true` and the hand-built template, model turn pre-filled with prefix (no closing turn marker), so the model continues mid-sentence.
- Ollama chat: `/api/chat` with `think: false`, images = [base64] when given.
- LiteRT backend: `litert_lm.Engine(path, backend=Backend.CPU()/GPU())`; continuation via `engine.create_session(apply_prompt_template=False)`, `run_prefill([prompt])`, `run_decode_async()`; run blocking calls in a thread, pass tokens through an asyncio.Queue. Health = engine loaded.
- Cancellation: if the consumer stops iterating (aclose / CancelledError), stop the HTTP stream promptly.

## mock_llm.py
`python -m swarmmind.mock_llm --port 11501 --tps 12 --model gemma4:e4b [--host 127.0.0.1]` — Ollama-compatible:
GET /api/tags; POST /api/chat and /api/generate with NDJSON streaming (`{"model","created_at","response"|"message":{...},"done":false}` … final `done:true`).
Emits a deterministic canned answer (~60–80 words) word-by-word (tokens like `" word"`) at `--tps`.
For `/api/generate` raw continuation: extract the pre-filled model-turn text from the prompt, count its words, and continue the canned answer from that word index (so the handed-off answer is seamless). If the user turn contains "Detected:" or an image was given, mention it.

## Core HTTP / WS surface (served by every node on http_port)
- `GET /` → dashboard/index.html
- `GET /peers` → `{"self": id, "leader_id": id|null, "term": int, "peers": [Node...]}` (includes self)
  Node = `{node_id, rank, ip, http_port, model, vision_mode, alive, healthy, is_leader, tps, capabilities, active_job_id, last_seen_ms}`
- `GET /state` → same body as the `ui.snapshot` message
- `GET /frame.jpg` → latest camera frame seen by this node (404 if none)
- `POST /ask {"question": str}` → `{"job_id", "leader_id"}`; any node forwards to the leader; 503 if no leader
- `POST /task {"capability": str, "args": {}}` → `{"task_id"}` (generic task bus, e.g. robot arm)
- `WS /ws/ui` → server pushes `{"type": str, "body": {...}}`:
  - `ui.snapshot` `{self, leader_id, term, top_node_id, nodes:[Node], jobs:[Job], reports:[Report], events:[Event], tasks:[Task]}`
  - `ui.nodes` every 500 ms `{self, leader_id, term, top_node_id, nodes:[Node]}`
  - `job.new` Job (without tokens) ; `job.token` `{job_id, seq, token, node_id, term}` ;
    `job.handoff` `{job_id, from_node, to_node, at_seq, gap_ms, vision_note}` ; `job.done` `{job_id, final_seq, answer}`
  - `report.new` Report ; `report.sent` `{job_ids:[...]}` ; `event` `{ts, level, node_id, term, msg}` ; `task.*` Task
  - Job = `{job_id, question, model, vision_mode, vision_text, has_image, state:"streaming"|"done", node_id, term, tokens:[{seq, token, node_id}], handoffs:[...], started_ts, done_ts}`
  - Report = `{job_id, question, answer, nodes_used:[...], handoffs:[...], timestamp, sent: bool}`
  - **Handoff rule:** on `job.handoff`, discard any tokens with seq >= at_seq (they came from the dead leader and may differ), then continue. Dedupe by (job_id, seq).
- `WS /ws/peer` → peer bus (Envelope JSON, see protocol.py)

## vision.py
```python
class Vision:
    def __init__(self, cfg: NodeConfig): ...
    mode: str              # effective mode after fallback: native | coral | cpu | none  (log it)
    has_detector: bool     # True if coral or cpu detector loaded
    latest_jpeg: bytes | None
    async def capture_frame(self) -> bytes | None      # camera_url: http(s)://…/shot.jpg, file:<path>, mock: (synthetic jpeg via PIL), "" → None. timeout 1.5 s
    async def detect_text(self, jpeg: bytes) -> str | None   # "Detected: person 0.87 (left), smoke 0.64 (center)" or None if no detector; runs in thread
```
Fallback order when cfg.vision_mode == "coral": pycoral+EdgeTPU model → cpu (ai_edge_litert / tflite_runtime / tensorflow.lite Interpreter with detector_model_cpu) → none. "cpu" starts at cpu. "native" = backend handles images; additionally load a cpu detector if detector_model_cpu is set (optional). Missing model files ⇒ fall back, with a warning, never crash.

## uplink.py
```python
class Uplink:
    def __init__(self, cfg: NodeConfig, broadcast: Callable[[str, dict], Awaitable[None]], is_leader: Callable[[], bool]): ...
    def add_report(self, report: dict) -> bool    # dedupe by job_id; append to data/<node_id>/reports.jsonl; True if new
    def mark_sent(self, job_ids: list[str]) -> None   # persist (rewrite file / append sent markers)
    def all_reports(self) -> list[dict]           # each has "sent": bool
    async def run(self) -> None                   # every cfg.uplink_interval: if is_leader() and cfg.uplink_urls: find first reachable LAN sink, POST each unsent report as JSON, mark sent, await broadcast("report.sent", {"job_ids": [...]})
```
No Discord / cloud. Sinks are LAN HTTP endpoints (a base-station laptop running scripts/uplink_sink.py). Never double-post: dedupe by job_id, and sink is also idempotent (it dedupes by job_id).
