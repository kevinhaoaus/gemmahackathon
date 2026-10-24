"""LLM backends: Ollama (HTTP), LiteRT-LM (in-process), and a null backend.

Every backend streams "tokens" (str pieces; "".join(tokens) is the answer) and supports
mid-sentence continuation: the model turn is pre-filled with the text produced so far,
so a new leader can finish an answer that a dead leader started.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import httpx

from .config import NodeConfig
from .logutil import get_logger

log = get_logger("inference")

# Turn markers that must never leak into answer text. Seeing an end marker means the
# model finished its turn, so the stream stops there.
_END_MARKERS = ("<turn|>", "<end_of_turn>", "<eos>", "<|turn>", "<start_of_turn>")
_STRIP_MARKERS = ("<bos>", "<|channel>", "<channel|>")


def _family(model: str) -> str:
    m = model.lower().replace("-", "").replace("_", "")
    return "gemma4" if "gemma4" in m else "gemma3"


def build_gemma_prompt(system: str, user: str, prefix: str, model: str) -> str:
    """Hand-built chat template with the model turn left open and pre-filled with `prefix`.

    gemma4 (verified against Ollama's own gemma4 renderer and LiteRT-LM's
    render_message_to_string): `<|turn>role\\n...<turn|>\\n`, real system role. No literal
    `<bos>`: Ollama prepends BOS itself in raw mode, and LiteRT's tokenizer does not parse it.
    gemma3: no system role, so the system prompt is folded into the user turn.
    """
    if _family(model) == "gemma4":
        sys_turn = f"<|turn>system\n{system}<turn|>\n" if system else ""
        return f"{sys_turn}<|turn>user\n{user}<turn|>\n<|turn>model\n{prefix}"
    body = f"{system}\n\n{user}" if system else user
    return f"<start_of_turn>user\n{body}<end_of_turn>\n<start_of_turn>model\n{prefix}"


def _infer_images(cfg: NodeConfig) -> bool:
    if cfg.model_supports_images is not None:
        return bool(cfg.model_supports_images)
    m = (cfg.model or cfg.litert_model_path).lower().replace("-", "").replace("_", "")
    if "gemma4" in m:
        return True                       # every gemma4 variant is multimodal
    if "gemma3" in m:
        return not any(s in m for s in ("gemma3:1b", "gemma31b", "270m"))
    return False


class _Cleaner:
    """Strips stray turn markers from a token stream; flags the end of the model turn."""

    def __init__(self) -> None:
        self.ended = False

    def __call__(self, piece: str) -> str:
        for mk in _END_MARKERS:
            i = piece.find(mk)
            if i >= 0:
                self.ended = True
                piece = piece[:i]
        for mk in _STRIP_MARKERS:
            piece = piece.replace(mk, "")
        return piece


class _TpsMeter:
    """tokens / seconds since the first token, updated live on the backend."""

    def __init__(self, backend: "LLMBackend") -> None:
        self.backend, self.n, self.t0 = backend, 0, 0.0

    def tick(self) -> None:
        now = time.monotonic()
        if self.n == 0:
            self.t0 = now
        self.n += 1
        if self.n > 1 and now > self.t0:
            self.backend.tps = round((self.n - 1) / (now - self.t0), 2)


class LLMBackend:
    name: str = "none"
    model: str = ""
    supports_images: bool = False
    tps: float = 0.0

    async def check_health(self) -> bool:
        return False

    def stream_chat(self, system: str, user: str, image_b64: str | None) -> AsyncIterator[str]:
        raise NotImplementedError

    def stream_continue(self, system: str, user: str, prefix: str, image_b64: str | None = None,
                        max_tokens: int | None = None) -> AsyncIterator[str]:
        """Continue `prefix` mid-sentence. image_b64: re-attach the job's frame (backends that can);
        max_tokens: remaining budget so a handed-off answer is no longer than an uninterrupted one."""
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class NullBackend(LLMBackend):
    """backend: none -- a node with no LLM (camera/vision/arm only). Never healthy."""

    def __init__(self, cfg: NodeConfig) -> None:
        self.model = ""
        self.supports_images = False
        self.tps = 0.0

    async def stream_chat(self, system: str, user: str, image_b64: str | None) -> AsyncIterator[str]:
        raise RuntimeError("this node has no LLM backend")
        yield ""  # pragma: no cover  (makes this an async generator)

    async def stream_continue(self, system: str, user: str, prefix: str, image_b64: str | None = None,
                        max_tokens: int | None = None) -> AsyncIterator[str]:
        raise RuntimeError("this node has no LLM backend")
        yield ""  # pragma: no cover


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, cfg: NodeConfig) -> None:
        self.cfg = cfg
        self.url = cfg.ollama_url.rstrip("/")
        self.model = cfg.model
        self.supports_images = _infer_images(cfg)
        self.tps = 0.0
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        # One shared client; streams have no read timeout (a slow CPU model is still alive).
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=2.0, read=None, write=10.0, pool=5.0))
        return self._client

    async def check_health(self) -> bool:
        try:
            r = await asyncio.wait_for(self.client.get(f"{self.url}/api/tags", timeout=1.5), 1.6)
            if r.status_code != 200:
                return False
            want = _strip_latest(self.model)
            for m in r.json().get("models", []):
                if want in (_strip_latest(m.get("name", "")), _strip_latest(m.get("model", ""))):
                    return True
            return False
        except Exception:
            return False

    async def _stream(self, path: str, payload: dict[str, Any],
                      pick: Callable[[dict[str, Any]], str]) -> AsyncIterator[str]:
        """POST an NDJSON stream; `async with` closes the HTTP stream on cancel/aclose."""
        meter, clean = _TpsMeter(self), _Cleaner()
        async with self.client.stream("POST", f"{self.url}{path}", json=payload) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode("utf-8", "replace")[:300]
                raise RuntimeError(f"ollama {path} HTTP {r.status_code}: {body}")
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("error"):
                    raise RuntimeError(f"ollama {path}: {obj['error']}")
                piece = clean(pick(obj) or "")
                if piece:
                    meter.tick()
                    yield piece
                if clean.ended or obj.get("done"):
                    return

    async def stream_chat(self, system: str, user: str, image_b64: str | None) -> AsyncIterator[str]:
        umsg: dict[str, Any] = {"role": "user", "content": user}
        if image_b64 and self.supports_images:
            umsg["images"] = [image_b64]
        msgs = ([{"role": "system", "content": system}] if system else []) + [umsg]
        payload = {"model": self.model, "messages": msgs, "stream": True, "think": False,
                   "keep_alive": -1, "options": {"num_predict": self.cfg.num_predict}}
        async for piece in self._stream("/api/chat", payload,
                                        lambda o: (o.get("message") or {}).get("content", "")):
            yield piece

    async def stream_continue(self, system: str, user: str, prefix: str, image_b64: str | None = None,
                        max_tokens: int | None = None) -> AsyncIterator[str]:
        payload: dict[str, Any] = {"raw": True, "stream": True, "keep_alive": -1, "model": self.model,
                                   "options": {"num_predict": max_tokens or self.cfg.num_predict}}
        if image_b64 and self.supports_images:
            # Raw mode takes images only via an [img-N] placeholder in the prompt (verified on
            # gemma4:e4b: without it Ollama fails to tokenize; with it the model sees the frame).
            user = f"[img-0]{user}"
            payload["images"] = [image_b64]
        payload["prompt"] = build_gemma_prompt(system, user, prefix, self.model)
        async for piece in self._stream("/api/generate", payload, lambda o: o.get("response", "")):
            yield piece

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _strip_latest(name: str) -> str:
    return name[:-7] if name.endswith(":latest") else name


class LiteRTBackend(LLMBackend):
    """In-process LiteRT-LM. Blocking native calls run in a worker thread; tokens cross to
    asyncio through a queue. One generation at a time per engine (serialised by a lock)."""
    name = "litert"

    def __init__(self, cfg: NodeConfig) -> None:
        self.cfg = cfg
        self.model = cfg.model
        self.supports_images = _infer_images(cfg)
        self.tps = 0.0
        self._engine: Any = None
        self._lm: Any = None
        self._load_error: str | None = None
        self._gen_lock = threading.Lock()
        # lazy-load at startup in a daemon thread so the node comes up immediately
        threading.Thread(target=self._load, name="litert-load", daemon=True).start()

    def _load(self) -> None:
        t0 = time.monotonic()
        try:
            import litert_lm
            self._lm = litert_lm
            path = self.cfg.litert_model_path
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(f"litert_model_path not found: {path!r}")
            hw = litert_lm.Backend.GPU() if self.cfg.litert_backend == "gpu" else litert_lm.Backend.CPU()
            ctx = max(1024, self.cfg.num_predict + 1024)
            try:
                kw = {"vision_backend": litert_lm.Backend.CPU(), "max_num_images": 1} if self.supports_images else {}
                self._engine = litert_lm.Engine(path, backend=hw, max_num_tokens=ctx, **kw)
            except Exception as e:
                if not self.supports_images:
                    raise
                log.warning("litert vision init failed (%s); loading text-only", e)
                self.supports_images = False
                self._engine = litert_lm.Engine(path, backend=hw, max_num_tokens=ctx)
            log.info("litert engine loaded in %.1fs (%s, images=%s)",
                     time.monotonic() - t0, self.cfg.litert_backend, self.supports_images)
        except Exception as e:
            self._load_error = str(e)
            log.error("litert engine load failed: %s", e)

    async def check_health(self) -> bool:
        return self._engine is not None

    async def _bridge(self, start: Callable[[], tuple[Iterator[Any], Any]],
                      text_of: Callable[[Any], str]) -> AsyncIterator[str]:
        """Run start() -> (blocking iterator, native obj) in a thread and relay its text.
        On consumer cancel/aclose: cancel_process() the native obj so the thread exits fast."""
        if self._engine is None:
            raise RuntimeError(f"litert engine not loaded: {self._load_error or 'still loading'}")
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        stop = threading.Event()
        holder: dict[str, Any] = {}

        def put(item: tuple[str, Any]) -> None:
            try:
                loop.call_soon_threadsafe(q.put_nowait, item)
            except RuntimeError:
                pass                                  # event loop already closed

        def worker() -> None:
            with self._gen_lock:
                obj = None
                try:
                    if stop.is_set():
                        return
                    it, obj = start()
                    holder["obj"] = obj
                    for chunk in it:
                        if stop.is_set():
                            break
                        put(("tok", text_of(chunk)))
                    put(("end", None))
                except BaseException as e:            # noqa: BLE001 - relay everything
                    put(("err", e))
                finally:
                    if obj is not None:
                        try:
                            obj.close()
                        except Exception:
                            pass

        threading.Thread(target=worker, name="litert-gen", daemon=True).start()
        meter, clean = _TpsMeter(self), _Cleaner()
        try:
            while True:
                kind, val = await q.get()
                if kind == "end":
                    return
                if kind == "err":
                    raise RuntimeError(f"litert: {val}") from val
                piece = clean(val or "")
                if piece:
                    meter.tick()
                    yield piece
                if clean.ended:
                    return
        finally:
            stop.set()
            obj = holder.get("obj")
            if obj is not None:
                try:
                    obj.cancel_process()
                except Exception:
                    pass

    async def stream_chat(self, system: str, user: str, image_b64: str | None) -> AsyncIterator[str]:
        lm = self._lm
        # message attempts, best first: inline image blob -> image temp file -> text only
        attempts: list[Any] = []
        tmp_path: str | None = None
        if image_b64 and self.supports_images:
            attempts.append({"role": "user", "content": [{"type": "image", "blob": image_b64},
                                                         {"type": "text", "text": user}]})
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="swarmmind_")
            with os.fdopen(fd, "wb") as f:
                f.write(base64.b64decode(image_b64))
            attempts.append({"role": "user", "content": [{"type": "image", "path": tmp_path},
                                                         {"type": "text", "text": user}]})
        attempts.append(user)

        def text_of(m: Any) -> str:
            c = getattr(m, "contents", None)
            if c is not None:
                return str(c)
            return "".join(x.get("text", "") for x in m.get("content", []) if isinstance(x, dict))

        try:
            for i, msg in enumerate(attempts):
                def start(msg: Any = msg) -> tuple[Iterator[Any], Any]:
                    conv = self._engine.create_conversation(system_message=system or None)
                    it = conv.send_message_async(
                        msg, max_output_tokens=self.cfg.num_predict,
                        thinking_config=lm.ThinkingConfig(enable_thinking=False))
                    return it, conv
                got = False
                try:
                    async for piece in self._bridge(start, text_of):
                        got = True
                        yield piece
                    return
                except RuntimeError as e:
                    if got or i == len(attempts) - 1:
                        raise
                    log.warning("litert chat attempt %d failed (%s); falling back", i, e)
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    async def stream_continue(self, system: str, user: str, prefix: str, image_b64: str | None = None,
                        max_tokens: int | None = None) -> AsyncIterator[str]:
        prompt = build_gemma_prompt(system, user, prefix, self.model or self.cfg.litert_model_path)

        def start() -> tuple[Iterator[Any], Any]:
            # raw LiteRT sessions are text-only: the caller passes detector labels instead of images
            s = self._engine.create_session(apply_prompt_template=False,
                                            max_output_tokens=max_tokens or self.cfg.num_predict)
            try:
                s.run_prefill([prompt])
            except Exception:
                s.close()
                raise
            return s.run_decode_async(), s

        async for piece in self._bridge(start, lambda r: r.texts[0] if r.texts else ""):
            yield piece

    async def aclose(self) -> None:
        eng, self._engine = self._engine, None
        if eng is not None:
            await asyncio.to_thread(self._close_engine, eng)

    def _close_engine(self, eng: Any) -> None:
        with self._gen_lock:                          # wait for any in-flight generation
            try:
                eng.close()
            except Exception:
                pass


def make_backend(cfg: NodeConfig) -> LLMBackend:
    kind = (cfg.backend or "none").lower()
    if kind == "ollama":
        return OllamaBackend(cfg)
    if kind == "litert":
        return LiteRTBackend(cfg)
    if kind != "none":
        log.warning("unknown backend %r; running without an LLM", cfg.backend)
    return NullBackend(cfg)
