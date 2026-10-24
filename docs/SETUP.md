# SwarmMind setup and operations

This guide covers running SwarmMind: the simulator, every node type, the demo, troubleshooting, and adding nodes. The wire protocol and the extension points (tasks, topics, robot arms) are specified in [PROTOCOL.md](PROTOCOL.md).

**Hard rule: 100% on-device.**
- Nodes talk only over the LAN.
- Each node runs Gemma 4 on its own hardware.
- Models are copied between machines over the LAN or from USB, never downloaded.
- Store-and-forward reports go to a LAN base station, not a cloud service.

The only step that needs internet is installing packages (apt, pip, the Ollama installer). Do it before the event, or stage wheels and installers on USB.

## Contents

0. [Physical network: what plugs into what](#0-physical-network-what-plugs-into-what)
1. [Nodes and ports](#1-nodes-and-ports)
2. [Simulator (one machine, no hardware)](#2-simulator-one-machine-no-hardware)
3. [Legion laptop (Windows)](#3-legion-laptop-windows)
4. [Raspberry Pi 5](#4-raspberry-pi-5)
5. [Adding a node (Jetson Thor, MacBook, Linux)](#5-adding-a-node-jetson-thor-macbook-linux)
6. [Copying models over the LAN](#6-copying-models-over-the-lan)
7. [Base station (store-and-forward sink)](#7-base-station-store-and-forward-sink)
8. [Demo script](#8-demo-script)
9. [Troubleshooting](#9-troubleshooting)
10. [Prize tracks](#10-prize-tracks)

---

## 0. Physical network: what plugs into what

**Don't cable the computers to each other.** A direct cable only connects two machines, and the swarm needs every machine on **one shared network** (same subnet, e.g. `192.168.1.x`). Every machine plugs into, or joins, one central box. No internet is needed or used.

```
                    ┌──────────────────────────────┐
                    │  travel router (no WAN/uplink)│  DHCP + Wi-Fi "swarm"
                    └──┬───────┬───────┬───────────┘
              Ethernet │       │       │ Ethernet          Wi-Fi (same router)
                 ┌─────┴──┐ ┌──┴─────┐ ┌┴──────┐     MacBook Airs · phone (IP Webcam / teleop)
                 │  Thor  │ │ Legion │ │  Pi   │     · any laptop opening the dashboard
                 └──┬─────┘ └──┬─────┘ └───────┘
                USB │          │ USB
          SO-101 arm, webcams  webcams           (USB devices belong to ONE node; not networked)
```

### Options, best first

| Option | What you need | Notes |
|---|---|---|
| **A. Your own router** (recommended) | Any Wi-Fi router, e.g. a GL.iNet travel router or an old home router, plus 3 Ethernet cables | Leave its WAN/internet port empty. It hands out IPs (DHCP) and Wi-Fi. Wire the Thor, Legion and Pi into its LAN ports. MacBooks and the phone join its Wi-Fi. |
| B. Phone hotspot | A spare phone with **mobile data off** | Quickest setup, about 10 devices, Wi-Fi only (no Ethernet), more jitter. The IP Webcam phone can't also be the hotspot without extra hassle, so use a second phone. |
| C. Venue Wi-Fi | Nothing | Often has **client isolation**, which stops devices seeing each other. Test it first (below). If it passes, use `static_peers` everywhere, since broadcast is often blocked even when ping works. |
| D. Ethernet switch, no router | An unmanaged switch | No DHCP, so set a static IP on every machine by hand. MacBooks need USB-C Ethernet adapters, and the phone can't join, so there's no phone camera or teleop. |

### Rules that matter for this system

- **Wire the Thor and the Legion with Ethernet.** A peer counts as dead after 600 ms without a heartbeat, and Wi-Fi hiccups can cause false failovers.
- **Turn the Thor's Wi-Fi off** (`nmcli radio wifi off`) if pulling its Ethernet cable is your demo move. Otherwise it stays reachable over Wi-Fi and no handoff happens.
- **One address per node.** A machine on both Ethernet and Wi-Fi has two IPs. Set `host_ip` explicitly in its config (the Legion currently has `.11` on Ethernet and `.36` on Wi-Fi), or turn off the interface you don't use.
- **Fixed IPs.** In the router's admin page, set a DHCP reservation per machine (MAC address → IP) so the IPs in `static_peers` never change. Or set static IPs on each machine.
- **USB devices are not on the network.** The SO-101 and the webcams plug into one node each. That node advertises them (`capabilities: ["arm.so101"]`, `camera_url: usb:0`), and everything else reaches them through the swarm.
- **The phone** joins the same Wi-Fi. Use its IP for the IP Webcam URL, and open `http://<any-node>:8700/teleop` (HTTPS `:8443` is needed for the IMU, see the Thor guide).

### Two-minute connectivity check (do this before anything else)

```bash
ping 192.168.1.30                       # from every machine to every other (Windows: ping -n 3)
# with a node running on the target:
curl http://192.168.1.30:8700/peers     # Windows PowerShell: curl.exe ...
```

If `ping` fails, it's the network (isolation, wrong subnet, Windows "Public" profile). If ping works but `curl` fails, it's that machine's firewall (section 9).

---

## 1. Nodes and ports

| Node | Config | Rank | Backend / model | Vision |
|---|---|---|---|---|
| Jetson AGX Thor | `configs/thor.yaml` | 200 | Ollama `gemma4:e4b` (copied from the Legion; a bigger Gemma 4 only if someone brings one) | native |
| Lenovo Legion 5i (RTX 4070, Windows) | `configs/laptop.yaml` (`node_id: legion`) | 100 | Ollama `gemma4:e4b` | native |
| MacBook Air (Apple Silicon) | `configs/macbook.yaml`, `configs/macbook2.yaml`, ... | 80, 70, (60) | Ollama `gemma4:e4b` | native |
| Raspberry Pi 5 8GB | `configs/pi.yaml` | 50 | LiteRT-LM `gemma-4-E4B-it.litertlm` (text-only) | Coral, falling back to cpu, then none |

The highest healthy `(rank, node_id)` leads. Each config's `static_peers` uses this example IP plan. Your router decides the subnet (GL.iNet uses `192.168.8.x`, many home routers `192.168.1.x` or `192.168.0.x`), so keep the last numbers and edit the configs to match:

| Node | Example IP |
|---|---|
| thor | `192.168.1.30` |
| legion | `192.168.1.36` |
| pi | `192.168.1.40` |
| macbook | `192.168.1.41` |
| macbook2 | `192.168.1.42` |
| IP Webcam phone | `192.168.1.50` |
| base station | `192.168.1.36:8800` (on the Legion) |

| Port | Proto | Purpose | Open on |
|---|---|---|---|
| 50000 | UDP | heartbeat / discovery | every node |
| 8700 | TCP | node HTTP API, dashboard, `/ws/peer`, `/ws/ui`, `/ws/topics`, `/teleop` | every node |
| 8800 | TCP | uplink sink (`scripts/uplink_sink.py`) | base station only |
| 11434 | TCP | Ollama | **localhost only**: no peer ever calls another node's LLM |

## 2. Simulator (one machine, no hardware)

The simulator runs three mock Ollama servers and three nodes on one machine: `sim_a` (rank 100, 12 tok/s), `sim_b` (rank 50, 6 tok/s) and `sim_c` (rank 10, 4 tok/s). They use loopback with distinct UDP ports, so no firewall changes are needed.

```powershell
.venv\Scripts\python scripts\simulate.py
```

| Command | Effect |
|---|---|
| `status` | node table: leader, term, health |
| `ask <question>` | submit a question (to any node; it is forwarded to the leader) |
| `kill a` / `revive a` | hard-kill / restart node `a` (`b`, `c` also work) |
| `sick a` / `heal a` | make node `a`'s LLM unresponsive / responsive again (tests `healthy=false`) |
| `tail a [n]` | last n log lines of node `a` |
| `open` | print the dashboard URLs |
| `help`, `quit` | |

Dashboards: http://127.0.0.1:8701/ (sim_a), http://127.0.0.1:8702/ (sim_b), http://127.0.0.1:8703/ (sim_c).

Try it:
1. `ask what do you see?`
2. While it streams, `kill a`. sim_b continues the same sentence, and the answer shows `⟶ handed to sim_b in N ms`.
3. `revive a`. sim_a retakes leadership only after the answer finishes and it has been stable for 3 s.

Tests:

```powershell
.venv\Scripts\python -m pytest -q tests
```

## 3. Legion laptop (Windows)

Prerequisites:
- Python 3.11 with the `py` launcher.
- Ollama installed with `gemma4:e4b` already present (`ollama list`). SwarmMind never pulls models.

Run from the repo root. Use an **elevated** PowerShell, which the firewall rules need:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
# offline: ... -File scripts\setup_windows.ps1 -Wheels D:\wheels
```

The script does the following:
- creates `.venv` (`py -3.11`) and installs `requirements.txt`,
- checks that Ollama at `127.0.0.1:11434` lists `gemma4:e4b`,
- adds inbound firewall rules for UDP 50000, TCP 8700 and TCP 8800, skipping any that already exist,
- warns about `python.exe` block rules,
- prints the LAN IP and network profile.

Edit `configs/laptop.yaml`:
- `static_peers`: the other nodes' IPs.
- `camera_url`: the phone's IP. Install the Android "IP Webcam" app and start the server; the snapshot URL is `http://<phone-ip>:8080/shot.jpg`.
- `uplink_urls`: the base station.

Then run the node:

```powershell
.venv\Scripts\python -m swarmmind.agent --config configs/laptop.yaml
```

The dashboard is at http://127.0.0.1:8700/, or http://\<legion-ip\>:8700/ from any device on the LAN.

## 4. Raspberry Pi 5

Requirements:
- Raspberry Pi OS Bookworm, 64-bit (Python 3.11).
- Active cooler and the official 27 W PSU.
- Coral USB Accelerator in a blue USB 3 port.

**1. Copy the repo to the Pi** over the LAN, for example from the Legion:

```powershell
scp -r C:\Users\Default.DESKTOP-NUBLEQ1\Desktop\gemmahackathon pi@192.168.1.40:~/gemmahackathon
```

**2. Get the models onto the Pi.** Use one of these:

- **Push** the `.litertlm` from the Legion (~3.6 GB):

  ```powershell
  scp C:\Users\Default.DESKTOP-NUBLEQ1\Desktop\GemmaHackathonOld\GemmaHackathonBasic\models\gemma-4-E4B-it-litert-lm\gemma-4-E4B-it.litertlm pi@192.168.1.40:~/gemmahackathon/models/
  ```

- **Pull** from the Pi. This needs the Windows "OpenSSH Server" optional feature on the Legion:

  ```bash
  scripts/setup_pi.sh --from kevin@192.168.1.36:/C:/Users/Default.DESKTOP-NUBLEQ1/Desktop/GemmaHackathonOld/GemmaHackathonBasic/models/gemma-4-E4B-it-litert-lm
  ```

- **Copy from USB:** `--from /media/pi/<stick>/models`.

The Coral detection files are optional. Stage them on USB from a connected machine before the event, then pass the same `--from`:
- `ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite`
- `ssd_mobilenet_v2_coco_quant_postprocess.tflite`
- `coco_labels.txt`

**3. Run the setup script:**

```bash
cd ~/gemmahackathon
chmod +x scripts/setup_pi.sh
scripts/setup_pi.sh --from <source>                                         # online packages
scripts/setup_pi.sh --from <source> --skip-packages --wheels /media/pi/USB/wheels   # fully offline
```

| Step | Network? | What it does |
|---|---|---|
| packages (`--skip-packages` skips it) | yes | apt base packages, the Coral `libedgetpu1-std` runtime from Google's apt repo, and pip `requirements-pi.txt` (`ai-edge-litert`, `litert-lm-api`) |
| pip offline (`--wheels DIR`, or `./wheels`) | no | `pip install --no-index --find-links DIR` |
| pycoral | yes, or a staged wheel | no official wheel for Python 3.11. Uses a staged `pycoral-*.whl` from the wheels dir if present, otherwise tries Coral's index. If it fails, **vision falls back to cpu** via `ai-edge-litert`, using the same SSD model without the EdgeTPU. |
| models | LAN or USB | copies into `models/`, verifies the files (and the ~3.6 GB size), and prints instructions for anything missing |
| service | no | `/etc/systemd/system/swarmmind.service`: `WorkingDirectory` = repo, `ExecStart` = `.venv/bin/python -m swarmmind.agent --config configs/pi.yaml`, `Restart=always`. Enabled and started. |

To pre-stage wheels on a connected machine:

```bash
pip download -r requirements-pi.txt -d wheels/ --platform manylinux2014_aarch64 --python-version 3.11 --only-binary=:all:
```

Check `configs/pi.yaml` (`static_peers`, `camera_url`), then:

```bash
sudo systemctl restart swarmmind
journalctl -u swarmmind -f                      # logs: node=pi term=N ...
.venv/bin/python -m swarmmind.agent --config configs/pi.yaml   # or run it in the foreground
```

**Ollama on the Pi (alternative).** Install Ollama while you still have network, copy the model store from another machine (section 6), then set these in `configs/pi.yaml`:

```yaml
backend: ollama
model: gemma4:e4b
litert_model_path: ""
```

## 5. Adding a node (Jetson Thor, MacBook, Linux)

To add a node:
1. Create a config file with a **unique `node_id`** and a rank.
2. Add the new node's `ip:8700` to the other nodes' `static_peers`. This is only required where broadcast is blocked, but recommended.
3. Open UDP 50000 and TCP 8700 on the new node.
4. Run `python -m swarmmind.agent --config configs/<file>.yaml`.

Unknown YAML keys are an error. Every key must be a `NodeConfig` field.

### Jetson AGX Thor (Ubuntu / JetPack, aarch64)

`scripts/setup_pi.sh` works on the Jetson with these flags:

```bash
scripts/setup_pi.sh --config configs/thor.yaml --requirements requirements.txt --no-coral
```

How the Jetson differs from the Pi:

| Item | Jetson |
|---|---|
| JetPack | provides CUDA. Ollama has native Jetson support: install it with the Ollama installer while you have network. |
| Coral | not needed (`--no-coral`). The Thor GPU runs Gemma 4 with native vision. |
| `.litertlm` | not needed (`backend: ollama`). The script requires it only for `backend: litert` configs. |
| Python | Ubuntu's `python3` may not be 3.11. The script warns but continues (`python3 -m venv`). |
| Model | copy the largest Gemma 4 you have over the LAN (section 6), then make `model:` in `configs/thor.yaml` match `ollama list` exactly. |
| systemd unit | runs `--config configs/thor.yaml` |

Use `backend: ollama` (Ollama's API). SwarmMind has no llama.cpp backend; the backends are `ollama`, `litert` and `none`.

Firewall (ufw):

```bash
sudo ufw allow 50000/udp && sudo ufw allow 8700/tcp
```

### MacBook Air (Apple Silicon, macOS)

```bash
# Python 3.11 (python.org installer or Homebrew python@3.11), Ollama.app installed beforehand
cd ~/gemmahackathon                                    # repo copied over the LAN (scp -r / rsync / USB)
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt              # offline: --no-index --find-links wheels/
ollama list                                            # gemma4:e4b must be listed (section 6 to copy it)
.venv/bin/python -m swarmmind.agent --config configs/macbook.yaml
```

To add another MacBook, copy `configs/macbook.yaml` to `configs/macbook3.yaml` and change `node_id`, `rank` (60-80) and `static_peers`.

macOS firewall and privacy:
- **Application Firewall** (System Settings > Network > Firewall): if it is on, allow incoming connections for the venv's real Python binary:

  ```bash
  PY="$(.venv/bin/python -c 'import os,sys; print(os.path.realpath(sys.executable))')"
  sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add "$PY"
  sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp "$PY"
  ```

- **Local Network privacy** (macOS 15+): the terminal app that runs the node (Terminal, iTerm, VS Code) needs System Settings > Privacy & Security > **Local Network** enabled. Without it, UDP broadcast and LAN connections fail silently.
- Keep the Mac awake during the demo: `caffeinate -dims` in another terminal.

### Any other Linux box

Run `scripts/setup_pi.sh --config configs/<file>.yaml --requirements requirements.txt --no-coral`, or set it up by hand (venv, `pip install -r requirements.txt`, `python -m swarmmind.agent --config ...`). Open the same ports with `ufw`.

### Device nodes (robot arm, sensors)

To make a node an arm controller, add a capability and a handler module:

```yaml
capabilities: ["arm.so101"]
capability_config:
  arm.so101: {port: "/dev/ttyACM0", mock: false}
```

The handler goes in `swarmmind/capabilities/so101.py` (spec and full example in [PROTOCOL.md](PROTOCOL.md#6-capabilities--task-bus)). For a demo without hardware, add `"arm.mock"` to any node's `capabilities`, then:

```bash
curl -X POST http://<any-node>:8700/task -H "Content-Type: application/json" -d '{"capability":"arm.mock","args":{"action":"home"}}'
```

Phone teleop is served at `http://<node>:8700/teleop`. Phone sensors need a secure context; see [PROTOCOL.md section 7.2](PROTOCOL.md#72-phone-teleop-and-the-imu-caveat). With Android over USB: `adb reverse tcp:8700 tcp:8700`, then open `http://localhost:8700/teleop`.

Cameras: `camera_url` accepts `http(s)://.../shot.jpg`, `file:<path>`, `mock:` or `""`. USB webcams are not a `camera_url` scheme. Expose them as a snapshot URL on the LAN, or use the IP Webcam phone.

## 6. Copying models over the LAN

Never `ollama pull` at the venue. Copy the model store from a machine that already has the model.

**Ollama models (any OS to any OS; the store format is platform-independent).** Copy `manifests/` and `blobs/` from the source's model dir to the target's, then restart Ollama.

| OS | Model dir |
|---|---|
| Windows | `%USERPROFILE%\.ollama\models` |
| macOS | `~/.ollama/models` |
| Linux (systemd install) | `/usr/share/ollama/.ollama/models` (owned by `ollama`) |

```bash
# from the Legion to a Mac (PowerShell; scp is built into Windows)
scp -r C:\Users\<you>\.ollama\models\manifests C:\Users\<you>\.ollama\models\blobs me@192.168.1.41:~/.ollama/models/
# to a Linux/Jetson Ollama service (from any Linux/macOS shell holding a copy in <src>/models)
rsync -a <src>/models/ user@192.168.1.30:/tmp/ollama-models/ && \
  ssh user@192.168.1.30 'sudo rsync -a /tmp/ollama-models/ /usr/share/ollama/.ollama/models/ && sudo chown -R ollama:ollama /usr/share/ollama/.ollama && sudo systemctl restart ollama'
```

To copy only one model, take its manifest (`manifests/registry.ollama.ai/library/gemma4/<tag>`) plus the `blobs/sha256-*` files that manifest lists. Alternatively, copy a GGUF file and import it locally:

```bash
printf 'FROM ./gemma-4-31b-it.gguf\n' > Modelfile && ollama create gemma4:31b -f Modelfile
```

**LiteRT-LM (Pi):** copy `gemma-4-E4B-it.litertlm` into `models/` (section 4).

## 7. Base station (store-and-forward sink)

On any LAN machine (in the example plan, the Legion):

```bash
python scripts/uplink_sink.py --port 8800      # stores to data/basestation/reports.jsonl
```

To see what has arrived, open `http://<base-ip>:8800/` or `GET /reports`.

Every node keeps every finished report. The leader POSTs unsent reports to the first reachable `uplink_urls` entry every 5 s. The sink dedupes by `job_id`, so each report lands exactly once, even across leader changes.

## 8. Demo script

1. Start the nodes. Open http://\<any-node\>:8700/ on a big screen; every node serves the same live view. Confirm the leader badge (Thor if present, otherwise the Legion), the term, and that all nodes are green.
2. **Ask** a question in the dashboard, for example "What do you see? Any hazards?" The answer streams from the leader, with the camera frame shown.
3. **Kill the leader mid-answer.** Unplug its network cable or power, or press Ctrl+C on its agent. Within about a second, the next node continues **the same sentence**. The answer shows `⟶ handed to <node> in N ms`. If that node can't see images, the handoff note shows that it re-ran its detector or used `[vision unavailable]`.
4. **Bring the leader back.** It rejoins as a follower, catches up via sync, and retakes leadership only **after the current answer finishes** and it has been stable for 3 s. It never interrupts a stream.
5. **Store-and-forward.** Leave the base-station sink off while answering a few questions; the reports queue up on every node. Then start `python scripts/uplink_sink.py --port 8800`, or plug in the base-station laptop. Within 5 s, the queue flushes, the dashboard marks the reports as sent, and each one lands **exactly once** at the sink.
6. Optional robot arm (LabMate, see [LABMATE.md](LABMATE.md)):
   1. Press **Inspect bench** in the LabMate panel.
   2. Submit steps as `arm.so101` tasks (or `arm.mock`) if you want to drive the arm directly.
   3. Kill the arm's node, and watch the task reassign.
   4. Cancel a task with `POST /task/{id}/cancel` (the e-stop).

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Nodes don't see each other | **Venue WiFi blocks broadcast** (AP/client isolation). Fill in `static_peers` on every node (unicast heartbeats and the peer bus still work if unicast is allowed). If the WiFi blocks client-to-client traffic entirely, bring your own network: a travel router, an Ethernet switch, or a phone hotspot with mobile data off. None of these need internet. |
| Windows node invisible to others | Firewall: run `setup_windows.ps1` as admin (rules for UDP 50000 / TCP 8700 / TCP 8800, profile Any). Check `Get-NetConnectionProfile`: a **Public** network blocks most inbound traffic; switch with `Set-NetConnectionProfile -InterfaceAlias Wi-Fi -NetworkCategory Private`. An old **Block** rule for `python.exe` (from clicking Cancel on the Windows prompt) overrides Allow rules; the script lists them. |
| Mac node invisible | Application Firewall allow rule, plus **Local Network** permission for the terminal app (section 5). |
| Linux / Jetson invisible | `sudo ufw status`; `sudo ufw allow 50000/udp`, `sudo ufw allow 8700/tcp`. |
| Pi slow, or the answer stalls | **Thermal throttling.** Check `vcgencmd measure_temp` (keep it below about 80 °C) and `vcgencmd get_throttled` (`0x0` is good; bit 0 = under-voltage, bit 2 = throttled now). Use the active cooler and the **official 27 W PSU**, and lower `num_predict` in `configs/pi.yaml`. |
| "Ollama listening only on localhost" | Expected, and correct. Each node calls only its **own** LLM at `127.0.0.1:11434`; peers never call another node's Ollama. `OLLAMA_HOST` only matters if something remote calls Ollama, and SwarmMind doesn't. |
| `model not listed` / node never healthy | `ollama list` must contain the exact `model:` string from the config. Copy the model (section 6). For LiteRT, check `models/gemma-4-E4B-it.litertlm` exists and is ~3.6 GB. |
| Coral not detected | `lsusb` should show `1a6e:089a` (before first use) or `18d1:9302` (after). Use a blue USB 3 port and a short cable. The udev rules come from `libedgetpu1-std`; re-plug after installing, and make sure the user is in `plugdev` (setup_pi.sh adds it). Without pycoral or the EdgeTPU, the log shows `vision: coral unavailable ... -> cpu`, which is the designed fallback. |
| Handoff `gap_ms` looks odd across machines | Clock skew is irrelevant: `gap_ms` is measured on the new leader's own monotonic clock. |
| `unknown config keys` on start | A YAML key isn't a `NodeConfig` field (`swarmmind/config.py`). Check for typos. |
| `POST /ask` returns 503 | No healthy leader (an election is in progress, or every LLM is down). Check `GET /peers`. |
| Phone teleop sends nothing | No secure context. Use `adb reverse` with `localhost`, or HTTPS, or the Chrome flag (PROTOCOL.md 7.2). On iOS, tap the button to grant motion permission. |

## 10. Prize tracks

- **Gemma 4.** Every node runs Gemma 4 locally:
  - the largest model on the Jetson Thor,
  - `gemma4:e4b` on the Legion and the MacBooks (multimodal, native vision),
  - Gemma 4 E4B on the Pi.

  The mid-sentence handoff uses Gemma's chat template with the model turn pre-filled.
- **LiteRT.** The Pi runs Gemma 4 E4B through **LiteRT-LM** (`.litertlm`, raw-prompt continuation for handoff). Its CPU vision fallback is the SSD detector on the **LiteRT interpreter** (`ai-edge-litert`).
- **Raspberry Pi.** A Pi 5 is a full peer: it can lead, finish an answer mid-sentence when bigger nodes die, and hold and flush the report queue.
- **Coral.** On the Pi, the EdgeTPU runs object detection and supplies the vision context for the text-only LLM. It degrades gracefully to cpu, then to none.
- **Jetson (NVIDIA).** The Thor is the top-rank leader for the robot cell's planning.
