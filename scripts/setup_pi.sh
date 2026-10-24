#!/usr/bin/env bash
# SwarmMind setup for Raspberry Pi 5 (Raspberry Pi OS Bookworm, 64-bit, Python 3.11).
# Also usable on other aarch64/x86_64 Linux nodes, e.g. a Jetson AGX Thor (Ubuntu/JetPack):
#   scripts/setup_pi.sh --config configs/thor.yaml --requirements requirements.txt --no-coral
# (Jetson: Ollama runs natively with CUDA; import the model over the LAN, see docs/SETUP.md.)
#
# Everything SwarmMind runs is on-device. This script NEVER downloads a model from the
# internet: models are copied from the laptop over the LAN (scp) or from a local/USB path.
#
# Steps (each idempotent, safe to re-run):
#   1. [network] apt packages + Coral EdgeTPU runtime (Google apt repo) + pip from PyPI.
#      Skip with --skip-packages. Offline alternative: --wheels DIR (pre-staged wheels).
#   2. [LAN]     copy models into models/ from --from sources, then verify them.
#   3. [local]   Coral USB check, plugdev group, systemd service swarmmind.service.
#
# Usage:
#   scripts/setup_pi.sh --from kevin@192.168.1.36:/C:/Users/Default.DESKTOP-NUBLEQ1/Desktop/GemmaHackathonOld/GemmaHackathonBasic/models/gemma-4-E4B-it-litert-lm
#   scripts/setup_pi.sh --from /media/pi/USB/models --skip-packages --wheels /media/pi/USB/wheels
#
# Options:
#   --from SRC       model source dir; repeatable. "user@host:/path" (scp) or a local path.
#                    Each needed file is looked up as SRC/<file> and SRC/gemma-4-E4B-it-litert-lm/<file>.
#   --skip-packages  skip apt, the Coral apt repo and online pip (no internet needed).
#   --wheels DIR     install Python packages offline: pip install --no-index --find-links DIR.
#                    Default: <repo>/wheels if it exists.
#   --config PATH    node config the service runs (default: configs/pi.yaml).
#   --requirements F pip requirements file (default: requirements-pi.txt; Jetson: requirements.txt).
#   --no-coral       skip the Coral EdgeTPU runtime and pycoral (Jetson, or no Coral attached).
#   --no-service     do not install the systemd service.
#   --no-start       install + enable the service but do not (re)start it now.
#   --user NAME      user the service runs as (default: the invoking user).
#   -h, --help       show this help.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$REPO_DIR/models"
VENV_DIR="$REPO_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"
SERVICE_NAME="swarmmind"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

LLM_MODEL="gemma-4-E4B-it.litertlm"
CORAL_MODEL="ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
CPU_MODEL="ssd_mobilenet_v2_coco_quant_postprocess.tflite"
LABELS="coco_labels.txt"

SOURCES=()
NODE_CONFIG="configs/pi.yaml"
REQUIREMENTS="requirements-pi.txt"
WITH_CORAL=1
SKIP_PACKAGES=0
WHEELS_DIR=""
INSTALL_SERVICE=1
START_SERVICE=1
SERVICE_USER="${SUDO_USER:-$(id -un)}"

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ ok  ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn ]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail ]\033[0m %s\n' "$*" >&2; exit 1; }

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)          [[ $# -ge 2 ]] || die "--from needs a value"; SOURCES+=("$2"); shift 2 ;;
    --skip-packages) SKIP_PACKAGES=1; shift ;;
    --wheels)        [[ $# -ge 2 ]] || die "--wheels needs a value"; WHEELS_DIR="$2"; shift 2 ;;
    --config)        [[ $# -ge 2 ]] || die "--config needs a value"; NODE_CONFIG="$2"; shift 2 ;;
    --requirements)  [[ $# -ge 2 ]] || die "--requirements needs a value"; REQUIREMENTS="$2"; shift 2 ;;
    --no-coral)      WITH_CORAL=0; shift ;;
    --no-service)    INSTALL_SERVICE=0; shift ;;
    --no-start)      START_SERVICE=0; shift ;;
    --user)          [[ $# -ge 2 ]] || die "--user needs a value"; SERVICE_USER="$2"; shift 2 ;;
    -h|--help)       usage; exit 0 ;;
    *)               die "unknown argument: $1 (see --help)" ;;
  esac
done

if [[ -z "$WHEELS_DIR" && -d "$REPO_DIR/wheels" ]]; then
  WHEELS_DIR="$REPO_DIR/wheels"
fi

if [[ $EUID -eq 0 ]]; then SUDO=""; else SUDO="sudo"; fi

[[ -f "$REPO_DIR/$NODE_CONFIG" ]] || die "config not found: $REPO_DIR/$NODE_CONFIG"
[[ -f "$REPO_DIR/$REQUIREMENTS" ]] || die "requirements file not found: $REPO_DIR/$REQUIREMENTS"
# The .litertlm is required only when the node runs the LiteRT backend.
LLM_REQUIRED=0
if grep -Eq '^[[:space:]]*backend:[[:space:]]*"?litert"?([[:space:]]|#|$)' "$REPO_DIR/$NODE_CONFIG"; then
  LLM_REQUIRED=1
fi

log "repo: $REPO_DIR"
ARCH="$(uname -m)"
[[ "$ARCH" == "aarch64" ]] || warn "expected aarch64 (64-bit Raspberry Pi OS), got $ARCH"

# ---------------------------------------------------------------------------
# 1. Packages (needs network unless --skip-packages / --wheels)
# ---------------------------------------------------------------------------
install_apt_packages() {
  log "apt: base packages"
  $SUDO apt-get update
  $SUDO apt-get install -y python3 python3-venv python3-dev python3-pip \
    libusb-1.0-0 usbutils curl gnupg ca-certificates openssh-client

  if [[ $WITH_CORAL -eq 0 ]]; then
    log "skipping Coral runtime (--no-coral)"
    return 0
  fi
  log "apt: Coral EdgeTPU runtime (Google apt repo)"
  local keyring="/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg"
  local list="/etc/apt/sources.list.d/coral-edgetpu.list"
  if [[ ! -s "$keyring" ]]; then
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | $SUDO gpg --dearmor --yes -o "$keyring"
  fi
  if [[ ! -f "$list" ]]; then
    echo "deb [signed-by=$keyring] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
      | $SUDO tee "$list" >/dev/null
    $SUDO apt-get update
  fi
  # libedgetpu1-std = standard clock (cooler). Installs udev rules for the USB accelerator.
  if ! $SUDO apt-get install -y libedgetpu1-std; then
    warn "libedgetpu1-std failed to install; Coral disabled, vision will use the cpu detector"
  fi
}

ensure_venv() {
  if [[ -x "$VENV_PY" ]]; then
    ok "venv exists: $VENV_DIR"
  else
    log "creating venv: $VENV_DIR"
    python3 -m venv "$VENV_DIR" || die "python3 -m venv failed (install python3-venv, or run without --skip-packages)"
  fi
  local pyver
  pyver="$("$VENV_PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
  log "venv python: $pyver"
  [[ "$pyver" == "3.11" ]] || warn "SwarmMind targets Python 3.11; found $pyver"
}

install_python_packages() {
  if [[ -n "$WHEELS_DIR" ]]; then
    [[ -d "$WHEELS_DIR" ]] || die "--wheels dir not found: $WHEELS_DIR"
    log "pip (offline): $WHEELS_DIR"
    "$VENV_PY" -m pip install --no-index --find-links "$WHEELS_DIR" -r "$REPO_DIR/$REQUIREMENTS"
  elif [[ $SKIP_PACKAGES -eq 0 ]]; then
    log "pip (online, PyPI)"
    "$VENV_PY" -m pip install --upgrade pip
    "$VENV_PY" -m pip install -r "$REPO_DIR/$REQUIREMENTS"
  else
    warn "--skip-packages and no wheels dir: Python packages not installed"
  fi
}

install_pycoral() {
  if [[ $WITH_CORAL -eq 0 ]]; then return 0; fi
  # pycoral's official wheels stop at Python 3.9; Bookworm ships 3.11, so this often fails.
  # Without pycoral, vision_mode coral falls back to cpu via ai-edge-litert (slower, same model family).
  if "$VENV_PY" -c 'import pycoral' 2>/dev/null; then
    ok "pycoral already importable"
    return 0
  fi
  local whl=""
  if [[ -n "$WHEELS_DIR" ]]; then
    whl="$(find "$WHEELS_DIR" -maxdepth 1 -name 'pycoral-*.whl' | head -n 1 || true)"
  fi
  if [[ -n "$whl" ]]; then
    log "pycoral from staged wheel: $whl"
    "$VENV_PY" -m pip install --no-index --find-links "$WHEELS_DIR" "$whl" \
      || warn "pycoral wheel install failed"
  elif [[ $SKIP_PACKAGES -eq 0 ]]; then
    log "pycoral from Coral's Python package index"
    "$VENV_PY" -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral \
      || warn "no pycoral wheel for this Python version"
  fi
  if "$VENV_PY" -c 'import pycoral' 2>/dev/null; then
    ok "pycoral installed"
  else
    warn "pycoral not available. Vision will fall back: coral -> cpu (ai-edge-litert) -> none."
    warn "To use the EdgeTPU on Python 3.11, stage a community-built pycoral cp311 aarch64 wheel"
    warn "(with a matching tflite-runtime wheel) in the wheels dir and re-run with --wheels DIR."
  fi
}

if [[ $SKIP_PACKAGES -eq 0 ]]; then
  install_apt_packages
else
  log "skipping apt / Coral repo (--skip-packages)"
fi
ensure_venv
install_python_packages
install_pycoral

# ---------------------------------------------------------------------------
# 2. Models: copy over the LAN / from local media, never from the internet
# ---------------------------------------------------------------------------
mkdir -p "$MODELS_DIR"

SSH_CTL_DIR="$(mktemp -d)"
cleanup() {
  for sock in "$SSH_CTL_DIR"/*; do
    [[ -S "$sock" ]] && ssh -o ControlPath="$sock" -O exit dummy 2>/dev/null || true
  done
  rm -rf "$SSH_CTL_DIR"
}
trap cleanup EXIT
# One SSH connection per host (password prompt at most once).
SCP_OPTS=(-q -o ConnectTimeout=8 -o ControlMaster=auto -o "ControlPath=$SSH_CTL_DIR/%r@%h:%p" -o ControlPersist=120)

is_remote() {
  local s="$1"
  [[ "$s" == *:* && "$s" != /* && "$s" != ./* && "$s" != ../* ]]
}

# copy_one NAME: try every source until NAME lands in models/
copy_one() {
  local name="$1" dest="$MODELS_DIR/$1" src rel
  if [[ -s "$dest" ]]; then
    ok "present: models/$name"
    return 0
  fi
  for src in "${SOURCES[@]+"${SOURCES[@]}"}"; do
    src="${src%/}"
    for rel in "$name" "gemma-4-E4B-it-litert-lm/$name"; do
      if is_remote "$src"; then
        if scp "${SCP_OPTS[@]}" "$src/$rel" "$dest.part" 2>/dev/null; then
          mv -f "$dest.part" "$dest"; ok "copied (LAN): $src/$rel"; return 0
        fi
      elif [[ -f "$src/$rel" ]]; then
        cp -f "$src/$rel" "$dest.part" && mv -f "$dest.part" "$dest"
        ok "copied (local): $src/$rel"; return 0
      fi
    done
  done
  rm -f "$dest.part"
  return 1
}

MISSING=()
WANTED=("$CORAL_MODEL" "$CPU_MODEL" "$LABELS")
if [[ $LLM_REQUIRED -eq 1 ]]; then WANTED=("$LLM_MODEL" "${WANTED[@]}"); fi
if [[ $WITH_CORAL -eq 0 ]]; then WANTED=("${WANTED[@]/$CORAL_MODEL}"); fi
for f in "${WANTED[@]}"; do
  [[ -n "$f" ]] || continue
  if ! copy_one "$f"; then MISSING+=("$f"); fi
done

# sanity check: the .litertlm is ~3.6 GB; a truncated copy fails at load time
if [[ -s "$MODELS_DIR/$LLM_MODEL" ]]; then
  size=$(stat -c %s "$MODELS_DIR/$LLM_MODEL")
  if (( size < 1000000000 )); then
    warn "models/$LLM_MODEL is only $size bytes (expected ~3.6 GB); deleting the truncated copy"
    rm -f "$MODELS_DIR/$LLM_MODEL"
    MISSING+=("$LLM_MODEL")
  fi
fi

PI_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
PI_IP="${PI_IP:-PI_IP}"

LLM_MISSING=0
if [[ ${#MISSING[@]} -gt 0 ]]; then
  warn "missing model files in $MODELS_DIR:"
  for f in "${MISSING[@]}"; do
    warn "  - $f"
    if [[ "$f" == "$LLM_MODEL" ]]; then LLM_MISSING=1; fi
  done
  echo >&2
  echo "How to get them onto this node (LAN or USB only, no internet download):" >&2
  if [[ $LLM_MISSING -eq 1 ]]; then
    cat >&2 <<EOF

  $LLM_MODEL  (required for backend: litert)
    From the laptop, push it (Windows has scp built in), in PowerShell:
      scp C:\\Users\\Default.DESKTOP-NUBLEQ1\\Desktop\\GemmaHackathonOld\\GemmaHackathonBasic\\models\\gemma-4-E4B-it-litert-lm\\$LLM_MODEL $SERVICE_USER@$PI_IP:$MODELS_DIR/
    Or pull it from the Pi (needs "OpenSSH Server" enabled on the laptop):
      scripts/setup_pi.sh --from USER@LAPTOP_IP:/C:/Users/Default.DESKTOP-NUBLEQ1/Desktop/GemmaHackathonOld/GemmaHackathonBasic/models/gemma-4-E4B-it-litert-lm
    Or from a USB stick: scripts/setup_pi.sh --from /media/$SERVICE_USER/<stick>/models
EOF
  fi
  cat >&2 <<EOF

  $CORAL_MODEL, $CPU_MODEL, $LABELS  (optional; vision)
    These are the standard Coral SSD MobileNet v2 COCO detection files. Stage them on a
    USB stick from a connected machine before the event, then: --from /media/.../models
    Without them vision degrades: coral -> cpu -> none ("[vision unavailable]"); the LLM still works.

EOF
fi

# ---------------------------------------------------------------------------
# 3. Coral check, systemd service
# ---------------------------------------------------------------------------
if [[ $WITH_CORAL -eq 1 ]] && command -v lsusb >/dev/null 2>&1; then
  # 1a6e:089a = Coral USB before firmware load, 18d1:9302 = after (Google Inc.)
  if lsusb | grep -qiE '1a6e:089a|18d1:9302'; then
    ok "Coral USB Accelerator detected"
  else
    warn "Coral USB Accelerator not detected (lsusb). Use a blue USB3 port; vision falls back to cpu."
  fi
fi
if getent group plugdev >/dev/null 2>&1 && ! id -nG "$SERVICE_USER" | tr ' ' '\n' | grep -qx plugdev; then
  log "adding $SERVICE_USER to plugdev (Coral USB access; re-login needed for shells)"
  $SUDO usermod -aG plugdev "$SERVICE_USER"
fi

if [[ $INSTALL_SERVICE -eq 1 ]]; then
  command -v systemctl >/dev/null 2>&1 || die "systemctl not found; use --no-service"
  log "installing $SERVICE_FILE (runs as $SERVICE_USER)"
  tmp_unit="$(mktemp)"
  GROUPS_LINE=""
  if getent group plugdev >/dev/null 2>&1; then GROUPS_LINE="SupplementaryGroups=plugdev"; fi
  cat > "$tmp_unit" <<EOF
[Unit]
Description=SwarmMind node (offline Gemma 4 swarm)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
$GROUPS_LINE
WorkingDirectory=$REPO_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$VENV_PY -m swarmmind.agent --config $NODE_CONFIG
Restart=always
RestartSec=1

[Install]
WantedBy=multi-user.target
EOF
  if [[ -f "$SERVICE_FILE" ]] && cmp -s "$tmp_unit" "$SERVICE_FILE"; then
    ok "service unit unchanged"
  else
    $SUDO install -m 0644 "$tmp_unit" "$SERVICE_FILE"
    $SUDO systemctl daemon-reload
    ok "service unit written"
  fi
  rm -f "$tmp_unit"
  $SUDO systemctl enable "$SERVICE_NAME" >/dev/null 2>&1
  ok "service enabled (starts on boot)"
  if [[ $START_SERVICE -eq 1 ]]; then
    if [[ $LLM_MISSING -eq 1 ]]; then
      warn "not starting the service: models/$LLM_MODEL is missing"
    else
      $SUDO systemctl restart "$SERVICE_NAME"
      ok "service (re)started: journalctl -u $SERVICE_NAME -f"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
log "LAN IP: $PI_IP  (add \"$PI_IP:8700\" to the other nodes' static_peers)"
log "Check $NODE_CONFIG: static_peers = other nodes' IP:8700, camera_url"
log "Dashboard: http://$PI_IP:8700/"
log "Manual run: $VENV_PY -m swarmmind.agent --config $NODE_CONFIG"
if command -v vcgencmd >/dev/null 2>&1; then
  log "temperature: $(vcgencmd measure_temp 2>/dev/null || echo n/a), throttled: $(vcgencmd get_throttled 2>/dev/null || echo n/a)"
fi

if [[ $LLM_MISSING -eq 1 ]]; then
  die "setup incomplete: models/$LLM_MODEL missing (see instructions above)"
fi
ok "setup complete"
