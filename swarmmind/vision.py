"""Camera capture + optional object detection (Coral EdgeTPU / CPU tflite) for vision context.

Effective mode after fallback: native | coral | cpu | none.
  * coral: pycoral + EdgeTPU model -> cpu -> none
  * cpu:   ai_edge_litert / tflite_runtime / tensorflow.lite with the plain .tflite model -> none
  * native: the LLM backend handles images itself; a cpu detector is loaded additionally if available.
Missing libraries or model files never crash the node; they only downgrade the mode.
"""
from __future__ import annotations

import asyncio
import io
import math
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .config import NodeConfig
from .logutil import get_logger

log = get_logger("vision")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORAL_MODEL = REPO_ROOT / "models" / "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
DEFAULT_CPU_MODEL = REPO_ROOT / "models" / "ssd_mobilenet_v2_coco_quant_postprocess.tflite"
DEFAULT_LABELS = REPO_ROOT / "models" / "coco_labels.txt"

SCORE_THRESHOLD = 0.4
TOP_K = 5
CAPTURE_TIMEOUT_S = 1.5
LOG_EVERY_S = 10.0


def describe_unavailable() -> str:
    return "[vision unavailable]"


def _resolve(path_str: str, default: Path) -> Path:
    if not path_str:
        return default
    p = Path(path_str).expanduser()
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p


def load_labels(path: Path) -> dict[int, str]:
    """Supports "id label" per line, or one label per line (index = line number)."""
    labels: dict[int, str] = {}
    if not path.is_file():
        return labels
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            labels[int(parts[0])] = parts[1].strip()
        else:
            labels[idx] = line
    return labels


def _position(x_center: float) -> str:
    if x_center < 1.0 / 3.0:
        return "left"
    if x_center < 2.0 / 3.0:
        return "center"
    return "right"


def format_detections(dets: list[tuple[str, float, float]]) -> str:
    """dets: (label, score, x_center normalized 0..1)."""
    dets = sorted((d for d in dets if d[1] >= SCORE_THRESHOLD), key=lambda d: d[1], reverse=True)[:TOP_K]
    if not dets:
        return "Detected: nothing above threshold"
    return "Detected: " + ", ".join(f"{lab} {score:.2f} ({_position(xc)})" for lab, score, xc in dets)


class _RateLimitedLog:
    def __init__(self) -> None:
        self._last: dict[str, float] = {}

    def warn(self, kind: str, msg: str) -> None:
        now = time.monotonic()
        if now - self._last.get(kind, -1e9) >= LOG_EVERY_S:
            self._last[kind] = now
            log.warning(msg)


# ---------------------------------------------------------------- detectors

class _BaseDetector:
    kind = "base"

    def __init__(self, model_path: Path, labels: dict[int, str]) -> None:
        self.model_path = model_path
        self.labels = labels
        self._lock = threading.Lock()

    def label(self, cls_id: int) -> str:
        return self.labels.get(cls_id, f"class{cls_id}")

    def detect(self, img: Image.Image) -> list[tuple[str, float, float]]:
        raise NotImplementedError


class _CpuDetector(_BaseDetector):
    kind = "cpu"

    def __init__(self, model_path: Path, labels: dict[int, str]) -> None:
        super().__init__(model_path, labels)
        interp_cls, self.lib = _import_cpu_interpreter()
        self.interp = interp_cls(model_path=str(model_path))
        self.interp.allocate_tensors()
        inp = self.interp.get_input_details()[0]
        self.input_index = inp["index"]
        self.input_dtype = inp["dtype"]
        shape = inp["shape"]
        self.in_h, self.in_w = int(shape[1]), int(shape[2])
        q = inp.get("quantization") or (0.0, 0)
        self.in_scale, self.in_zero = float(q[0] or 0.0), int(q[1] or 0)
        self.outputs = self.interp.get_output_details()
        if len(self.outputs) < 4:
            raise RuntimeError(f"expected 4 detection outputs, model has {len(self.outputs)}")

    def _prepare(self, img: Image.Image) -> np.ndarray:
        arr = np.asarray(img.convert("RGB").resize((self.in_w, self.in_h), Image.BILINEAR))
        if self.input_dtype == np.float32:
            data = (arr.astype(np.float32) - 127.5) / 127.5
        elif self.input_dtype == np.uint8:
            data = arr.astype(np.uint8)
        elif self.input_dtype == np.int8:
            if self.in_scale > 0:
                real = arr.astype(np.float32) / 255.0
                data = np.clip(np.round(real / self.in_scale + self.in_zero), -128, 127).astype(np.int8)
            else:
                data = (arr.astype(np.int16) - 128).astype(np.int8)
        else:
            data = arr.astype(self.input_dtype)
        return np.expand_dims(data, 0)

    def _read_outputs(self) -> dict[str, np.ndarray]:
        tensors: list[tuple[str, np.ndarray]] = []
        for od in self.outputs:
            t = self.interp.get_tensor(od["index"])
            q = od.get("quantization") or (0.0, 0)
            if t.dtype in (np.uint8, np.int8) and q[0]:
                t = (t.astype(np.float32) - q[1]) * q[0]
            tensors.append((str(od.get("name", "")).lower(), np.asarray(t, dtype=np.float32)))
        return _assign_outputs(tensors)

    def detect(self, img: Image.Image) -> list[tuple[str, float, float]]:
        data = self._prepare(img)
        with self._lock:
            self.interp.set_tensor(self.input_index, data)
            self.interp.invoke()
            out = self._read_outputs()
        boxes = out["boxes"].reshape(-1, 4)
        classes = out["classes"].reshape(-1)
        scores = out["scores"].reshape(-1)
        n = int(out["count"].reshape(-1)[0]) if out["count"].size else len(scores)
        n = max(0, min(n, len(scores), len(classes), len(boxes)))
        dets = []
        for i in range(n):
            s = float(scores[i])
            if s < SCORE_THRESHOLD:
                continue
            ymin, xmin, ymax, xmax = (float(v) for v in boxes[i])
            xc = min(max((xmin + xmax) / 2.0, 0.0), 1.0)
            dets.append((self.label(int(round(float(classes[i])))), s, xc))
        return dets


def _assign_outputs(tensors: list[tuple[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Map the 4 postprocess outputs to boxes/classes/scores/count using shapes and names."""
    result: dict[str, np.ndarray] = {}
    rest: list[tuple[str, np.ndarray]] = []
    for name, t in tensors:
        if "boxes" not in result and t.ndim >= 2 and t.shape[-1] == 4 and t.size > 4:
            result["boxes"] = t
        elif "count" not in result and t.size == 1:
            result["count"] = t
        else:
            rest.append((name, t))
    if "boxes" not in result:
        raise RuntimeError("could not identify boxes output")
    pair = rest[:2]
    if len(pair) < 2:
        raise RuntimeError("could not identify classes/scores outputs")
    by_name: dict[str, np.ndarray] = {}
    for name, t in pair:
        if "score" in name:
            by_name["scores"] = t
        elif "class" in name:
            by_name["classes"] = t
    if len(by_name) == 2:
        result.update(by_name)
    else:
        (n0, t0), (n1, t1) = pair
        # TFLite_Detection_PostProcess naming: base=boxes, :1 classes, :2 scores, :3 count
        if n0.endswith(":1") and n1.endswith(":2"):
            result["classes"], result["scores"] = t0, t1
        elif n0.endswith(":2") and n1.endswith(":1"):
            result["classes"], result["scores"] = t1, t0
        else:
            def classiness(t: np.ndarray) -> float:
                integral = float(np.mean(np.abs(t - np.round(t)) < 1e-4)) if t.size else 0.0
                above_one = 1.0 if t.size and float(t.max()) > 1.0 else 0.0
                return integral + above_one
            if classiness(t0) >= classiness(t1):
                result["classes"], result["scores"] = t0, t1
            else:
                result["classes"], result["scores"] = t1, t0
    if "count" not in result:
        result["count"] = np.array([result["scores"].size], dtype=np.float32)
    return result


class _CoralDetector(_BaseDetector):
    kind = "coral"

    def __init__(self, model_path: Path, labels: dict[int, str]) -> None:
        super().__init__(model_path, labels)
        from pycoral.adapters import common, detect  # type: ignore
        from pycoral.utils.edgetpu import make_interpreter  # type: ignore
        self._common, self._detect = common, detect
        self.interp = make_interpreter(str(model_path))
        self.interp.allocate_tensors()
        self.in_w, self.in_h = common.input_size(self.interp)

    def detect(self, img: Image.Image) -> list[tuple[str, float, float]]:
        resized = img.convert("RGB").resize((self.in_w, self.in_h), Image.BILINEAR)
        with self._lock:
            self._common.set_input(self.interp, resized)
            self.interp.invoke()
            objs = self._detect.get_objects(self.interp, SCORE_THRESHOLD)
        dets = []
        for o in objs:
            xc = ((o.bbox.xmin + o.bbox.xmax) / 2.0) / float(self.in_w)
            dets.append((self.label(int(o.id)), float(o.score), min(max(xc, 0.0), 1.0)))
        return dets


def _import_cpu_interpreter() -> tuple[Any, str]:
    errors = []
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
        return Interpreter, "ai_edge_litert"
    except Exception as e:  # noqa: BLE001
        errors.append(f"ai_edge_litert: {e}")
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        return Interpreter, "tflite_runtime"
    except Exception as e:  # noqa: BLE001
        errors.append(f"tflite_runtime: {e}")
    try:
        import tensorflow as tf  # type: ignore
        return tf.lite.Interpreter, "tensorflow.lite"
    except Exception as e:  # noqa: BLE001
        errors.append(f"tensorflow: {e}")
    raise ImportError("no tflite interpreter available (" + "; ".join(errors) + ")")


# ---------------------------------------------------------------- mock camera

def _mock_jpeg(t: float) -> bytes:
    w, h = 640, 480
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    horizon = 300
    for y in range(horizon):  # sky gradient
        f = y / horizon
        d.line([(0, y), (w, y)], fill=(int(90 + 100 * f), int(150 + 70 * f), int(235 - 20 * f)))
    for y in range(horizon, h):  # ground
        f = (y - horizon) / (h - horizon)
        d.line([(0, y), (w, y)], fill=(int(30 + 20 * f), int(80 - 30 * f), int(30 + 5 * f)))
    for i in range(18):  # tree line
        x = i * 38 + 5
        th = 70 + (i * 37) % 50
        d.polygon([(x, horizon + 10), (x + 20, horizon + 10 - th), (x + 40, horizon + 10)],
                  fill=(20 + (i * 7) % 20, 70 + (i * 11) % 30, 25))
    # drifting orange "smoke" blob
    blob = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bd = ImageDraw.Draw(blob)
    cx = int(w / 2 + 180 * math.sin(t / 7.0))
    cy = int(200 + 30 * math.sin(t / 3.0))
    for k in range(5):
        r = 60 - k * 8
        ox, oy = int(25 * math.cos(t + k)), -k * 22
        bd.ellipse([cx + ox - r, cy + oy - r, cx + ox + r, cy + oy + r],
                   fill=(240, 120 + k * 15, 40 + k * 20, 150))
    blob = blob.filter(ImageFilter.GaussianBlur(12))
    img = Image.alpha_composite(img.convert("RGBA"), blob).convert("RGB")
    d = ImageDraw.Draw(img)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"
    d.rectangle([8, h - 28, 260, h - 6], fill=(0, 0, 0))
    d.text((14, h - 24), f"MOCK CAM {stamp}", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


# ---------------------------------------------------------------- Vision

class Vision:
    def __init__(self, cfg: NodeConfig):
        self.cfg = cfg
        self.camera_url = (cfg.camera_url or "").strip()
        self.latest_jpeg: bytes | None = None
        self.mode: str = "none"
        self.has_detector: bool = False
        self._detector: _BaseDetector | None = None
        self._client: httpx.AsyncClient | None = None
        self._rl = _RateLimitedLog()
        self._init_detector()

    # ---- detector setup with fallback chain
    def _init_detector(self) -> None:
        requested = (self.cfg.vision_mode or "none").lower()
        labels_path = _resolve(self.cfg.detector_labels, DEFAULT_LABELS)
        labels = load_labels(labels_path)
        cpu_path = _resolve(self.cfg.detector_model_cpu, DEFAULT_CPU_MODEL)
        reasons: list[str] = []

        if requested == "coral":
            det, why = self._try_coral(_resolve(self.cfg.detector_model, DEFAULT_CORAL_MODEL), labels)
            if det:
                self._set(det, "coral", requested, reasons)
                return
            reasons.append(why)
            log.warning("vision: coral unavailable: %s", why)
        if requested in ("coral", "cpu"):
            det, why = self._try_cpu(cpu_path, labels)
            if det:
                self._set(det, "cpu", requested, reasons)
                return
            reasons.append(f"cpu: {why}" if reasons else why)
            log.warning("vision: cpu detector unavailable: %s", why)
            self._set(None, "none", requested, reasons)
            return
        if requested == "native":
            self.mode = "native"
            if self.cfg.detector_model_cpu or cpu_path.is_file():
                det, why = self._try_cpu(cpu_path, labels)
                if det:
                    self._detector, self.has_detector = det, True
                    log.info("vision mode: native (+cpu detector %s)", cpu_path.name)
                    return
                if self.cfg.detector_model_cpu:
                    log.warning("vision: optional cpu detector for native mode unavailable: %s", why)
            log.info("vision mode: native (no detector; backend handles images)")
            return
        if requested != "none":
            reasons.append(f"unknown vision_mode {requested!r}")
        self._set(None, "none", requested, reasons)

    def _set(self, det: _BaseDetector | None, mode: str, requested: str, reasons: list[str]) -> None:
        self._detector, self.mode, self.has_detector = det, mode, det is not None
        if det is not None and not det.labels:
            log.warning("vision: labels file missing or empty; class ids will be shown")
        detail = f" model={det.model_path.name}" if det else ""
        if mode == requested and not reasons:
            log.info("vision mode: %s%s", mode, detail)
        else:
            log.info("vision mode: %s (fallback from %s: %s)%s", mode, requested, "; ".join(reasons), detail)

    @staticmethod
    def _try_coral(path: Path, labels: dict[int, str]) -> tuple[_BaseDetector | None, str]:
        try:
            import pycoral.utils.edgetpu  # type: ignore  # noqa: F401
            import pycoral.adapters.detect  # type: ignore  # noqa: F401
        except Exception as e:  # noqa: BLE001
            return None, f"pycoral import failed: {e}"
        if not path.is_file():
            return None, f"model file missing: {path}"
        try:
            return _CoralDetector(path, labels), ""
        except Exception as e:  # noqa: BLE001
            return None, f"edgetpu load failed: {e}"

    @staticmethod
    def _try_cpu(path: Path, labels: dict[int, str]) -> tuple[_BaseDetector | None, str]:
        try:
            _import_cpu_interpreter()
        except Exception as e:  # noqa: BLE001
            return None, str(e)
        if not path.is_file():
            return None, f"model file missing: {path}"
        try:
            det = _CpuDetector(path, labels)
            log.info("vision: cpu interpreter %s loaded %s", det.lib, path.name)
            return det, ""
        except Exception as e:  # noqa: BLE001
            return None, f"tflite load failed: {e}"

    # ---- capture
    async def capture_frame(self) -> bytes | None:
        url = self.camera_url
        if not url:
            return None
        try:
            if url.startswith(("http://", "https://")):
                data = await self._capture_http(url)
            elif url.startswith("file:"):
                data = await asyncio.to_thread(self._read_file, url[len("file:"):])
            elif url.startswith("mock:"):
                data = await asyncio.to_thread(_mock_jpeg, time.time())
            else:
                self._rl.warn("bad_url", f"vision: unsupported camera_url {url!r}")
                return None
        except asyncio.CancelledError:
            raise
        except httpx.TimeoutException:
            self._rl.warn("http_timeout", f"vision: camera timeout ({CAPTURE_TIMEOUT_S}s) at {url}")
            return None
        except httpx.HTTPError as e:
            self._rl.warn("http_error", f"vision: camera http error at {url}: {e!r}")
            return None
        except OSError as e:
            self._rl.warn("io_error", f"vision: camera read failed for {url}: {e}")
            return None
        except Exception as e:  # noqa: BLE001
            self._rl.warn("capture_error", f"vision: capture failed for {url}: {e!r}")
            return None
        if not data or not data.startswith(b"\xff\xd8"):
            self._rl.warn("bad_image", f"vision: camera at {url} returned no JPEG data")
            return None
        self.latest_jpeg = data
        return data

    async def _capture_http(self, url: str) -> bytes | None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=CAPTURE_TIMEOUT_S)
        r = await self._client.get(url, timeout=CAPTURE_TIMEOUT_S)
        if r.status_code != 200:
            self._rl.warn("http_status", f"vision: camera at {url} returned HTTP {r.status_code}")
            return None
        return r.content

    @staticmethod
    def _read_file(path: str) -> bytes:
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            p = REPO_ROOT / p
        return p.read_bytes()

    # ---- detection
    async def detect_text(self, jpeg: bytes) -> str | None:
        det = self._detector
        if det is None or not jpeg:
            return None
        try:
            return await asyncio.to_thread(self._detect_sync, det, jpeg)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self._rl.warn("detect_error", f"vision: detection failed: {e!r}")
            return None

    @staticmethod
    def _detect_sync(det: _BaseDetector, jpeg: bytes) -> str:
        with Image.open(io.BytesIO(jpeg)) as im:
            img = im.convert("RGB")
        return format_detections(det.detect(img))

    async def aclose(self) -> None:
        if self._client is not None:
            client, self._client = self._client, None
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass
