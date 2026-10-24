"""Per-node YAML config. One file per node; adding a node = adding a file."""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def _guess_ip() -> str:
    """Best-effort LAN IP without sending any packet off the LAN (UDP connect sends nothing)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


@dataclass
class NodeConfig:
    node_id: str
    rank: int
    host_ip: str = ""
    http_port: int = 8700
    bind_host: str = "0.0.0.0"
    # optional extra HTTPS listener for phones (browsers only expose IMU/DeviceOrientation to secure
    # pages). Peers keep using plain http_port. Self-signed cert is fine.
    https_port: int = 8443
    ssl_certfile: str = ""
    ssl_keyfile: str = ""
    # inference
    backend: str = "ollama"              # ollama | litert | none
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "gemma4:e4b"            # display + Ollama model name
    litert_model_path: str = ""          # .litertlm file when backend == litert
    litert_backend: str = "cpu"          # cpu | gpu
    model_supports_images: bool | None = None  # None = infer from backend/model
    num_predict: int = 512
    # vision
    vision_mode: str = "none"            # native | coral | cpu | none
    detector_model: str = ""             # EdgeTPU .tflite (coral) model path
    detector_model_cpu: str = ""         # plain .tflite model path (cpu)
    detector_labels: str = ""            # COCO labels txt
    camera_url: str = ""                 # http://phone:8080/shot.jpg | file:path.jpg | "" (none)
    # network
    static_peers: list[str] = field(default_factory=list)   # "ip:http_port"
    udp_port: int = 50000
    broadcast_addr: str = "255.255.255.255"
    heartbeat_interval: float = 0.2
    peer_timeout: float = 0.6
    health_interval: float = 0.5
    health_window: float = 2.0
    takeover_stable_s: float = 3.0
    startup_grace_s: float = 1.0
    # extra capabilities this node advertises (e.g. "arm.kinova"); llm/vision/camera are auto-added
    capabilities: list[str] = field(default_factory=list)
    # per-capability settings, e.g. {"arm.so101": {"port": "/dev/ttyACM0", "mock": false}}
    capability_config: dict[str, dict] = field(default_factory=dict)
    # store-and-forward uplink: LAN-only HTTP sinks (base station, command post). No cloud.
    uplink_urls: list[str] = field(default_factory=list)
    uplink_interval: float = 5.0
    data_dir: str = "data"
    log_level: str = "INFO"

    @property
    def http_addr(self) -> str:
        return f"{self.host_ip}:{self.http_port}"

    @property
    def node_data_dir(self) -> Path:
        p = Path(self.data_dir) / self.node_id
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | os.PathLike) -> NodeConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    known = set(NodeConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
    if "node_id" not in raw or "rank" not in raw:
        raise ValueError(f"{path}: node_id and rank are required")
    for k in ("static_peers", "capabilities", "uplink_urls", "capability_config"):
        if raw.get(k) is None:
            raw.pop(k, None)
    cfg = NodeConfig(**raw)
    if not cfg.host_ip:
        cfg.host_ip = _guess_ip()
    cfg.vision_mode = cfg.vision_mode.lower()
    cfg.backend = cfg.backend.lower()
    return cfg
