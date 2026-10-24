"""Ollama-compatible mock LLM for the simulator and tests.

    python -m swarmmind.mock_llm --port 11501 --tps 12 --model gemma4:e4b [--host 127.0.0.1]

Streams a deterministic canned answer word by word at --tps: a valid JSON cell plan for the
planner prompt, {"ok": true, ...} for the verify prompt, LabMate label / answer / report answers
(labels come only from the simulator-only sim_truth line, see label_words), else the canned
field answer. Raw
/api/generate continuation counts the words already in the pre-filled model turn and
resumes from that index, so an answer handed between nodes joins seamlessly.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from .protocol import (ANSWER_SYSTEM_PROMPT, LABEL_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT,
                       REPORT_SYSTEM_PROMPT, VERIFY_SYSTEM_PROMPT)

# The opener has the same word count with or without an image, so a native-vision answer
# and its text-only continuation stay word-aligned.
OPENER_IMAGE = "Reviewing the attached camera frame."
OPENER_TEXT = "Working from the text report."
CANNED = ("The scene shows a dry grass slope below a timber ridge, with a thin smoke column "
          "rising near the tree line and drifting east toward the access trail. Hazard: wind "
          "is pushing fire toward the only exit route, and two people may be on the trail. "
          "Next action: send the nearest team to clear the trail now and confirm the escape "
          "route to the west before anyone moves upslope.")

_MODEL_MARKERS = ("<|turn>model\n", "<start_of_turn>model\n")
_DETECTED = re.compile(r"Detected:[^\n<]*")
# object lines of the planner's user turn: "- servo_1: part, visible"
_OBJECT_LINE = re.compile(r"^- ([A-Za-z0-9_.\-]+): ([A-Za-z_]+)", re.M)
VERIFY_ANSWER = '{"ok": true, "reason": "servo appears seated"}'


def answer_words(user: str, has_image: bool) -> list[str]:
    """Deterministic canned answer for this request, as a list of words."""
    parts = [OPENER_IMAGE if has_image else OPENER_TEXT]
    m = _DETECTED.search(user or "")
    if m:
        det = m.group(0).strip().rstrip(".")
        parts.append(f"Detector reports {det[len('Detected:'):].strip()}.")
    parts.append(CANNED)
    return " ".join(parts).split()


def plan_words(user: str) -> list[str]:
    """Canned valid plan (pick part -> place slot -> verify) using names listed in the user turn."""
    objs = _OBJECT_LINE.findall(user or "")
    part = next((n for n, k in objs if k == "part"), "servo_1")
    slot = next((n for n, k in objs if k == "slot"), "so101_slot_1")
    plan = {"steps": [{"op": "pick", "object": part}, {"op": "place", "target": slot},
                      {"op": "verify", "check": f"{part} seated in {slot}"}]}
    return json.dumps(plan).split()       # json.dumps uses single spaces: " ".join() restores it


# LabMate label job user turn (swarmmind/cell/labmate.py label_context)
_MARK_LINE = re.compile(r"^- #(\d+):", re.M)
_HAZARD_TYPES = re.compile(r"^Hazard types[^:]*:\s*(.*)$", re.M)
_USE_CASE_LINE = re.compile(r"^- ([a-z0-9_]+): [^;\n]*; needs (.*)$", re.M)
# SIMULATOR ONLY: labmate adds this line only when cell.blobs carries sim_truth (synthetic scene)
_SIM_TRUTH = re.compile(r"sim_truth: (\{.*\})\s*$", re.M)
_REPLY = re.compile(r"Operator's reply: (.*)", re.S)
_PROPOSED = re.compile(r"^Proposed experiment: (.*?) \(([a-z0-9_]+)\)\s*$", re.M)


def label_words(user: str) -> list[str]:
    """Label JSON for LabMate. Answers correctly ONLY from the simulator's ground-truth line
    (`sim_truth: {"<mark>": "<item>[:open]"}`); without it every mark is "unknown"."""
    marks = [int(m) for m in _MARK_LINE.findall(user or "")]
    truth: dict = {}
    m = _SIM_TRUTH.search(user or "")
    if m:
        try:
            truth = {str(k): str(v) for k, v in json.loads(m.group(1)).items()}
        except (ValueError, AttributeError):
            truth = {}
    hz = _HAZARD_TYPES.search(user or "")
    hazard_types = {h.strip() for h in hz.group(1).split(",")} if hz else set()
    items, hazards, seen = [], [], set()
    for mk in marks:
        name, _, state = truth.get(str(mk), "unknown").partition(":")
        items.append({"mark": mk, "item": name, "state": state})
        if name != "unknown":
            seen.add(name)
        if state == "open" and "open_container" in hazard_types:
            hazards.append({"type": "open_container", "marks": [mk]})
    use_case, best = "unknown", 0.0
    for uid, needs in _USE_CASE_LINE.findall(user or ""):
        need = [n.strip() for n in needs.split(",") if n.strip()]
        share = sum(n in seen for n in need) / len(need) if need else 0.0
        if share > best and share >= 0.5:
            use_case, best = uid, share
    return json.dumps({"items": items, "hazards": hazards, "use_case": use_case}).split()


def answer_fields_words(user: str) -> list[str]:
    """Answer JSON for LabMate from the operator's reply (keyword rules, deterministic)."""
    m = _REPLY.search(user or "")
    reply = (m.group(1) if m else "").strip()
    low = reply.lower()
    p = _PROPOSED.search(user or "")
    proposed = p.group(1) if p else "unknown"
    confirmed = any(w in low for w in ("yes", "right", "correct", "yep", "exactly"))
    phase = ("first_test" if any(w in low for w in ("first", "trial", "try", "rough", "test run"))
             else "final_data" if any(w in low for w in ("final", "publish", "real data", "keep the data"))
             else "unknown")
    noise_ok = (True if "noise" in low and any(w in low for w in ("ok", "fine", "acceptable", "don't care"))
                else False if "clean" in low else None)
    exp = proposed if confirmed and proposed != "unknown" else (reply[:60] or proposed)
    return json.dumps({"experiment": exp, "confirmed": confirmed, "phase": phase,
                       "noise_ok": noise_ok}).split()


def report_words(user: str) -> list[str]:
    """Short spoken summary built from the report job's fact lines."""
    facts = [ln.strip() for ln in (user or "").splitlines()]
    hazards = [ln[2:] for ln in facts if ln.startswith("- ")]

    def get(key: str) -> str:
        return next((ln.split(":", 1)[1].strip() for ln in facts if ln.startswith(key)), "")

    parts = ["Bench check done."]
    parts += [f"{h}." for h in hazards if h != "none"] or ["I found no hazards."]
    exp = get("Operator says") or get("Experiment I proposed") or get("Experiment")
    if exp:
        parts.append(f"Experiment: {exp.split(';')[0]}.")
    parts.append(f"Missing tools: {get('Missing tools') or 'none'}.")
    note = next((ln for ln in facts if ln.startswith(("Design notes", "Keep every"))), "")
    if note:
        parts.append(note.rstrip(".") + ".")
    return " ".join(parts).split()


def words_for(system: str, user: str, has_image: bool) -> list[str]:
    """Pick the canned answer by prompt: plan JSON, verify JSON, LabMate label / answer / report,
    or the field answer."""
    text = f"{system or ''}\n{user or ''}"
    if PLAN_SYSTEM_PROMPT in text:
        return plan_words(user)
    if VERIFY_SYSTEM_PROMPT in text:
        return VERIFY_ANSWER.split()
    if LABEL_SYSTEM_PROMPT in text:
        return label_words(user)
    if ANSWER_SYSTEM_PROMPT in text:
        return answer_fields_words(user)
    if REPORT_SYSTEM_PROMPT in text:
        return report_words(user)
    return answer_words(user, has_image)


def split_raw_prompt(prompt: str) -> tuple[str, str]:
    """(text before the LAST model-turn marker, pre-filled model text after it)."""
    best = -1
    marker = ""
    for mk in _MODEL_MARKERS:
        i = prompt.rfind(mk)
        if i > best:
            best, marker = i, mk
    if best < 0:
        return prompt, ""
    return prompt[:best], prompt[best + len(marker):]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def create_app(model: str, tps: float) -> FastAPI:
    app = FastAPI(title="swarmmind mock LLM")
    delay = 1.0 / tps if tps > 0 else 0.0

    def model_ok(name: str) -> bool:
        strip = lambda s: s[:-7] if s.endswith(":latest") else s  # noqa: E731
        return strip(name or "") == strip(model)

    def tokens_for(words: list[str], start: int, prefix: str, limit: int) -> list[str]:
        """Word-level tokens (" word") from index `start`; the very first word has no space."""
        out = []
        for i, w in enumerate(words[start:start + limit]):
            lead = "" if (start + i == 0 or (i == 0 and prefix[-1:].isspace())) else " "
            out.append(lead + w)
        return out

    def final_fields(n: int, elapsed: float, truncated: bool) -> dict[str, Any]:
        ns = int(elapsed * 1e9)
        return {"done": True, "done_reason": "length" if truncated else "stop",
                "total_duration": ns, "load_duration": 0, "prompt_eval_count": 1,
                "prompt_eval_duration": 0, "eval_count": n, "eval_duration": ns}

    async def respond(stream: bool, toks: list[str], truncated: bool,
                      wrap: Any) -> Response:
        t0 = time.monotonic()
        if not stream:
            await asyncio.sleep(delay * len(toks))
            body = {"model": model, "created_at": _now(), **wrap("".join(toks)),
                    **final_fields(len(toks), time.monotonic() - t0, truncated)}
            return JSONResponse(body)

        async def gen() -> AsyncIterator[bytes]:
            for tok in toks:
                await asyncio.sleep(delay)
                yield (json.dumps({"model": model, "created_at": _now(), **wrap(tok),
                                   "done": False}) + "\n").encode()
            yield (json.dumps({"model": model, "created_at": _now(), **wrap(""),
                               **final_fields(len(toks), time.monotonic() - t0, truncated)})
                   + "\n").encode()
        return StreamingResponse(gen(), media_type="application/x-ndjson")

    def limit_of(body: dict[str, Any]) -> int:
        n = int((body.get("options") or {}).get("num_predict") or -1)
        return n if n > 0 else 10_000

    @app.get("/")
    async def root() -> PlainTextResponse:
        return PlainTextResponse("Ollama is running")

    @app.get("/api/version")
    async def version() -> dict[str, str]:
        return {"version": "0.0.0-swarmmind-mock"}

    @app.get("/api/tags")
    async def tags() -> dict[str, Any]:
        fam = "gemma4" if "gemma4" in model else "gemma3"
        return {"models": [{"name": model, "model": model, "modified_at": _now(), "size": 0,
                            "digest": "mock", "details": {"family": fam, "families": [fam],
                                                          "format": "mock"}}]}

    @app.post("/api/chat")
    async def chat(req: Request) -> Response:
        body = await req.json()
        if not model_ok(body.get("model", "")):
            return JSONResponse({"error": f"model '{body.get('model')}' not found"}, 404)
        msgs = body.get("messages") or []
        users = [m for m in msgs if m.get("role") == "user"]
        user = users[-1].get("content", "") if users else ""
        has_image = any(m.get("images") for m in users)
        system = "\n".join(str(m.get("content", "")) for m in msgs if m.get("role") == "system")
        words = words_for(system, user, has_image)
        lim = limit_of(body)
        toks = tokens_for(words, 0, "", lim)
        return await respond(body.get("stream", True) is not False, toks, len(words) > lim,
                             lambda t: {"message": {"role": "assistant", "content": t}})

    @app.post("/api/generate")
    async def generate(req: Request) -> Response:
        body = await req.json()
        if not model_ok(body.get("model", "")):
            return JSONResponse({"error": f"model '{body.get('model')}' not found"}, 404)
        prompt = body.get("prompt", "") or ""
        if not prompt:            # real Ollama: empty prompt = just load the model and return
            return JSONResponse({"model": body.get("model"), "response": "", "done": True,
                                 "done_reason": "load"})
        if body.get("raw"):
            head, prefix = split_raw_prompt(prompt)
        else:
            head, prefix = prompt, ""
        words = words_for(str(body.get("system") or ""), head, bool(body.get("images")))
        start = len(prefix.split())
        lim = limit_of(body)
        toks = tokens_for(words, start, prefix, lim) if start < len(words) else []
        return await respond(body.get("stream", True) is not False, toks,
                             len(words) - start > lim, lambda t: {"response": t})

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Ollama-compatible mock LLM")
    ap.add_argument("--port", type=int, default=11501)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--tps", type=float, default=12.0, help="tokens per second")
    ap.add_argument("--model", default="gemma4:e4b")
    a = ap.parse_args()
    print(f"mock LLM {a.model} on http://{a.host}:{a.port} @ {a.tps} tok/s", flush=True)
    uvicorn.run(create_app(a.model, a.tps), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
