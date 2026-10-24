# Jetson AGX Thor setup

The Thor is the swarm's **rank-200 leader**. It answers every question while it's alive and healthy, sees the workspace through native Gemma 4 vision. If it dies, the Legion (rank 100) finishes the answer mid-sentence. When it comes back, it takes over again only after the current job finishes.

| | |
|---|---|
| OS | JetPack 7.x = Ubuntu 24.04, aarch64, CUDA 13, Python 3.12 |
| LLM | Ollama → `gemma4:e4b` (the only Gemma 4 we have; 6.2 GB, vision-capable) |
| Config | `configs/thor.yaml` |
| Ports | TCP 8700 (API, dashboard, peer bus), UDP 50000 (heartbeats), TCP 8443 (optional HTTPS for phones) |

Rule: **no model downloads.** The model is copied from the Legion over the LAN. Two small runtime pieces are needed: the Ollama binary and the Python wheels. Fetch them **once**, on whichever machine has a connection (step 1), then carry them over on a USB stick or by `scp`.

---

## 1. Stage the files on the Legion (before going offline)

Open PowerShell in the repo.

```powershell
# a) Python wheels for the Thor (aarch64, Python 3.12). Small (~30 MB).
#    opencv-python-headless is only needed for USB webcams (camera_url: usb:0).
py -3.11 -m pip download -r requirements.txt opencv-python-headless `
    --platform manylinux2014_aarch64 --platform manylinux_2_28_aarch64 `
    --python-version 3.12 --implementation cp --only-binary=:all: -d wheels-aarch64

# b) Ollama for Linux arm64. Match the Legion's version (0.40.10), which supports gemma4.
#    The release asset is .tgz or .tar.zst depending on the release; grab whichever exists.
#    https://github.com/ollama/ollama/releases/tag/v0.40.10  ->  ollama-linux-arm64.*
#    Skip this if `ollama --version` already works on the Thor.

# c) The model: just this folder (only gemma4:e4b is in it, 6.2 GB)
#    C:\Users\Default.DESKTOP-NUBLEQ1\.ollama\models   (blobs\ manifests\ manifests-v2\ metadata\)
```

## 2. Thor first boot

```bash
cat /etc/nv_tegra_release        # confirms the JetPack / L4T version
nvidia-smi                       # GPU visible?
sudo nvpmodel -q                 # current power mode
sudo nvpmodel -m 0               # MAXN (max performance). Mode numbers: cat /etc/nvpmodel.conf
sudo jetson_clocks               # pin clocks for a steady tokens/sec during the demo
tegrastats                       # live GPU/CPU/temp/power; Ctrl+C to stop
python3 --version                # expect 3.12
```

**Network:** plug the Thor into the same switch or router as the other nodes. Ethernet is better than venue Wi-Fi. Then give it a fixed IP so it matches the `static_peers` lists on the other nodes:

```bash
nmcli con show                                   # find the connection name, e.g. "Wired connection 1"
sudo nmcli con mod "Wired connection 1" ipv4.method manual \
     ipv4.addresses 192.168.1.30/24 ipv4.gateway 192.168.1.1
sudo nmcli con up "Wired connection 1"
sudo ufw status                                  # if active:
sudo ufw allow 8700/tcp && sudo ufw allow 50000/udp && sudo ufw allow 8443/tcp
```

## 3. Install Ollama offline

```bash
# from the USB stick / scp'd file (use the line that matches the file you have)
sudo tar -C /usr -xzf ollama-linux-arm64.tgz
sudo tar -C /usr --use-compress-program=unzstd -xf ollama-linux-arm64.tar.zst

sudo useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama
sudo usermod -aG video,render ollama                     # GPU device access
sudo tee /etc/systemd/system/ollama.service >/dev/null <<'EOF'
[Unit]
Description=Ollama
After=network-online.target
[Service]
User=ollama
Group=ollama
ExecStart=/usr/bin/ollama serve
Environment="OLLAMA_MODELS=/opt/ollama-models"
Environment="OLLAMA_KEEP_ALIVE=-1"
Restart=always
[Install]
WantedBy=multi-user.target
EOF
sudo mkdir -p /opt/ollama-models && sudo chown ollama:ollama /opt/ollama-models
sudo systemctl daemon-reload && sudo systemctl enable --now ollama
ollama --version
```

`OLLAMA_KEEP_ALIVE=-1` keeps the model in memory permanently. SwarmMind also pins it at startup. Either way, a handoff never waits on a model load.

## 4. Copy the model from the Legion (LAN only)

On the **Legion** (Windows has `scp` built in):

```powershell
scp -r C:\Users\Default.DESKTOP-NUBLEQ1\.ollama\models\* <thor-user>@192.168.1.30:/tmp/ollama-models/
```

On the **Thor**:

```bash
sudo rsync -a /tmp/ollama-models/ /opt/ollama-models/
sudo chown -R ollama:ollama /opt/ollama-models
sudo systemctl restart ollama
ollama list                                       # must show gemma4:e4b
# integrity check of the main weights blob (should print the same hash as the filename)
sha256sum /opt/ollama-models/blobs/sha256-370c2879f17648d337b0ab4e208fdce348e24d1060064e93d7645daafd56d39c
ollama run gemma4:e4b "Say ready in one word."
ollama ps                                         # PROCESSOR column must say 100% GPU
```

If `ollama ps` says CPU, the Ollama build isn't using Thor's GPU. Check `journalctl -u ollama -b | grep -iE "cuda|gpu|library"`. Then either try a newer Ollama release with Jetson/Thor support, or temporarily point `configs/thor.yaml` at `backend: litert` with the `.litertlm` file (see `configs/pi.yaml`; set `litert_backend: gpu`).

**Bigger model?** 128 GB of memory easily fits `gemma4:26b`/`31b`, but **we don't have one**. Only use it if someone brings the Ollama folder on a drive. Then copy it the same way and set `model:` in `configs/thor.yaml` to the exact name `ollama list` shows. Rank stays 200.

## 5. Install SwarmMind

```bash
# copy the repo over (git clone from the Legion's copy, scp -r, or USB), then:
cd ~/gemmahackathon
scp -r <legion-user>@192.168.1.36:C:/Users/Default.DESKTOP-NUBLEQ1/Desktop/gemmahackathon/wheels-aarch64 .
python3 -m venv .venv          # if this fails: python3-venv is missing (apt install python3-venv, needs apt access)
scripts/setup_pi.sh --config configs/thor.yaml --requirements requirements.txt \
    --no-coral --skip-packages --wheels wheels-aarch64 --no-start
```

That installs the wheels offline, skips Coral and model copying (Thor uses Ollama), and installs the `swarmmind` systemd service without starting it.

## 6. Edit `configs/thor.yaml`

| Key | Set to |
|---|---|
| `host_ip` | `192.168.1.30` (or `""` to auto-detect) |
| `static_peers` | every other node, e.g. `["192.168.1.36:8700", "192.168.1.41:8700"]`. Required if the venue Wi-Fi blocks broadcast. |
| `camera_url` | a USB webcam: `usb:0` (or `usb:/dev/video2`). Phone: `http://<phone-ip>:8080/shot.jpg`. List devices with `ls /dev/video*` or `v4l2-ctl --list-devices`. |
| `ssl_certfile` / `ssl_keyfile` | optional. HTTPS on :8443 so a phone browser exposes its IMU at `https://192.168.1.30:8443/teleop` |

Self-signed cert for the phone teleop page:

```bash
openssl req -x509 -newkey rsa:2048 -nodes -days 30 -subj "/CN=thor" \
  -keyout data/thor-key.pem -out data/thor-cert.pem
# then: ssl_certfile: data/thor-cert.pem   ssl_keyfile: data/thor-key.pem
```

**More than one webcam:** a node sends one camera frame per job. To use a second webcam on the Thor, run a second, camera-only node on the same machine. Copy `thor.yaml` to `thor_cam2.yaml` and change:

- `node_id: thor_cam2`, `rank: 1`, `backend: none`, `camera_url: usb:1`
- `http_port: 8710`, `udp_port: 50010`, `broadcast_addr: ""`

It can't share UDP 50000, so it only sees unicast heartbeats. Give it explicit `static_peers` (port 50000 is the default), and add `"192.168.1.30:8710:50010"` to every other node's `static_peers`. It never leads (it has no LLM), but it shows up in the dashboard with its own `/frame.jpg`.

## 7. Run and verify

```bash
.venv/bin/python -m swarmmind.agent --config configs/thor.yaml     # foreground first
```

Look for these log lines:

```
node thor rank 200 | backend ollama model gemma4:e4b images=True | vision native | caps [...]
udp heartbeat on :50000 -> ...
model gemma4:e4b loaded and pinned in N s
node legion up (rank 100, ...)
thor is now LEADER (term N)
```

```bash
curl -s localhost:8700/peers | python3 -m json.tool      # every node listed, leader_id = thor
curl -s -X POST localhost:8700/ask -H 'content-type: application/json' \
     -d '{"question":"What is on the workbench?"}'
```

Dashboard: `http://192.168.1.30:8700/` from any laptop. Once it all works, run it as a service:

```bash
sudo systemctl start swarmmind && journalctl -u swarmmind -f
```

## 8. Failover drill

1. Ask a question in the dashboard.
2. While it streams, pull the Thor's Ethernet cable (or `sudo systemctl stop swarmmind`). Within about 1 s the Legion continues the same sentence, and the dashboard shows "⟶ handed to legion in N ms" plus a DEGRADED banner.
3. Plug it back in. The Thor rejoins, syncs the job state, and becomes leader again only after that answer finishes and it has been healthy for 3 s.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Thor never sees peers | Is UDP broadcast blocked (venue Wi-Fi isolation)? Fill in `static_peers` on **every** node. Check `sudo ufw status`. Run `sudo tcpdump -ni any udp port 50000` to watch heartbeats. |
| Thor is never leader | Its `healthy` is false. Run `curl localhost:11434/api/tags` and check that `model:` exactly matches `ollama list`. |
| Slow tokens/sec | Confirm `ollama ps` shows GPU, `sudo nvpmodel -m 0`, `sudo jetson_clocks`. Watch `tegrastats` for thermal throttling. |
| First handoff to the Thor is slow | Its model wasn't loaded yet. Look for the "loaded and pinned" log line, and make sure `OLLAMA_KEEP_ALIVE=-1` is set. |
| `usb:0` gives no frame | Install `opencv-python-headless` (it's in the wheels). Check that the device isn't already in use (`fuser /dev/video0`) and try another index. |
| Arm "permission denied" | The `dialout` group (above), or check `ls -l /dev/ttyACM*`. |
