"""Small, dependency-free Python interface to the local Ollama server.

    from swarmmind.gemma import chat, stream_chat
    answer = chat("Say hello", model="gemma4:e2b")
    for token in stream_chat("Describe the bench", model="gemma4:e4b", image="bench.jpg"):
        print(token, end="", flush=True)

    python3 -m swarmmind.gemma --list
    python3 -m swarmmind.gemma --model gemma4:e4b "Say hello"

Weights must already be imported from USB/LAN. This module never pulls models.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma4:e4b"
VLLM_BASE_URL = "http://127.0.0.1:8000"
VLLM_MODEL = "gemma4-31b"


def _open(path: str, payload: dict | None, base_url: str, timeout: float):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(base_url.rstrip("/") + path, data=data,
                      headers={"Content-Type": "application/json"})
    try:
        return urlopen(request, timeout=timeout)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(body).get("error", body)
        except ValueError:
            detail = body
        if exc.code == 404 and payload and "model" in payload:
            detail = (f"Model {payload['model']!r} is unavailable. Import the USB weights "
                      "first, then use --list to check the installed model names. " + str(detail))
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach Ollama at {base_url}: {exc.reason}. "
                           "Start it with: systemctl --user start ollama") from exc


def list_models(*, base_url: str = BASE_URL) -> list[str]:
    """Return names already installed in the local runtime."""
    with _open("/api/tags", None, base_url, 5) as response:
        return [model["name"] for model in json.load(response).get("models", [])]


def stream_chat(prompt: str, *, model: str = DEFAULT_MODEL,
                image: str | Path | None = None, system: str = "",
                base_url: str = BASE_URL, max_tokens: int = 768,
                timeout: float = 180) -> Iterator[str]:
    """Yield answer chunks. Images are read locally and sent as base64 to Ollama."""
    message: dict = {"role": "user", "content": prompt}
    if image is not None:
        message["images"] = [base64.b64encode(Path(image).read_bytes()).decode("ascii")]
    messages = ([{"role": "system", "content": system}] if system else []) + [message]
    payload = {"model": model, "messages": messages, "stream": True, "think": False,
               "keep_alive": -1, "options": {"num_predict": max_tokens}}
    with _open("/api/chat", payload, base_url, timeout) as response:
        for line in response:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("error"):
                raise RuntimeError(f"Ollama: {event['error']}")
            content = (event.get("message") or {}).get("content", "")
            if content:
                yield content
            if event.get("done"):
                return
    raise RuntimeError("Ollama closed the response before generation completed.")


def chat(prompt: str, **kwargs) -> str:
    """Return the complete answer; accepts the same options as stream_chat."""
    return "".join(stream_chat(prompt, **kwargs))


def stream_vllm_chat(prompt: str, *, model: str = VLLM_MODEL,
                     image: str | Path | None = None, system: str = "",
                     base_url: str = VLLM_BASE_URL, max_tokens: int = 768,
                     timeout: float = 300) -> Iterator[str]:
    """Stream from the local 31B vLLM server, including optional image bytes."""
    content: str | list = prompt
    if image is not None:
        raw = Path(image).read_bytes()
        mime = "image/png" if raw.startswith(b"\x89PNG") else "image/jpeg"
        content = [{"type": "image_url", "image_url": {
            "url": f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")}},
            {"type": "text", "text": prompt}]
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": content})
    payload = {"model": model, "messages": messages, "stream": True,
               "max_tokens": max_tokens, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": False}}
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    request = Request(url + "/chat/completions", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    return
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(f"vLLM: {event['error']}")
                for choice in event.get("choices", []):
                    token = (choice.get("delta") or {}).get("content")
                    if token:
                        yield token
    except HTTPError as exc:
        raise RuntimeError(f"vLLM HTTP {exc.code}: "
                           f"{exc.read().decode('utf-8', 'replace')}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach vLLM at {base_url}: {exc.reason}") from exc
    raise RuntimeError("vLLM closed the response before generation completed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", nargs="?", help="prompt to send to the local model")
    parser.add_argument("--backend", choices=["ollama", "vllm"], default="ollama")
    parser.add_argument("--base-url", help="server URL, including a remote Jetson's LAN address")
    parser.add_argument("--model", help="exact installed model name")
    parser.add_argument("--list", action="store_true", help="list locally installed models")
    parser.add_argument("--image", type=Path, help="optional image for a vision-capable model")
    parser.add_argument("--system", default="", help="optional system instruction")
    parser.add_argument("--max-tokens", type=int, default=768)
    args = parser.parse_args()
    is_vllm = args.backend == "vllm"
    model = args.model or (VLLM_MODEL if is_vllm else DEFAULT_MODEL)
    base_url = args.base_url or (VLLM_BASE_URL if is_vllm else BASE_URL)
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if not args.list and not args.prompt:
        parser.error("provide a prompt or --list")
    try:
        if args.list:
            if is_vllm:
                root = base_url.rstrip("/").removesuffix("/v1")
                with urlopen(root + "/v1/models", timeout=5) as response:
                    models = [entry["id"] for entry in json.load(response).get("data", [])]
            else:
                models = list_models(base_url=base_url)
            print("\n".join(models) if models else "Ollama is running; no USB models imported yet.")
        else:
            stream = stream_vllm_chat if is_vllm else stream_chat
            for content in stream(args.prompt, model=model, image=args.image, base_url=base_url,
                                       system=args.system, max_tokens=args.max_tokens):
                print(content, end="", flush=True)
            print()
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
