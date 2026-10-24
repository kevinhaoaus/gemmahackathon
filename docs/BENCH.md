# Bench quick start: webcam → objects → reBot picks them up

`scripts/bench.py` is a standalone tool: no swarm, no Gemma, no network. Run everything from the repo root.
Its outputs go to `data/bench/` (snapshots, plus `bench_calib.json` with the arm base and gripper angles).

## 0. One-time setup
- `.venv\Scripts\pip install motorbridge` (Seeed's reBot SDK; this needs internet). Without it, only `cam` works.
- The bench cell is built into bench.py: tag36h11, 5 cm tags. Tags 0 (0,0), 3 (45.8,0), 1 (0,58.5) and 2 (45.8,58.5) cm
  lie with their printed top edge pointing along table −y (yaw −90).

## 1. Order of commands
| step | command | what you should see |
|---|---|---|
| 1 | `.venv\Scripts\python scripts\bench.py cam` | ~25 fps; green outlines with ids; `CALIBRATED err ~1 px`; a 5 cm grid; a red X / green Y arrow at tag 0. Clear the table, press `b`, then put objects down: they get `#N` and x,y in cm. Click = table coords; `s` = snapshot; `q` = quit. |
| 2 | `... bench.py arm-read` | READ-ONLY. 3 printouts of 7 motors (j1–j6 + gripper): position, velocity, torque, temperatures, status; then the FK tool pose. It ends with `OK: all 7 motors answered`. |
| 3 | `... bench.py arm-jog` | Type `ENABLE`. `j1 +5` turns joint 1 by +5°: **check that each joint moves the way the printed pose says** (signs). With torque off, open the gripper by hand and run `grip set-open`; close it and run `grip set-closed`. Then `grip open` / `grip close` work. |
| 4 | `... bench.py calib-base` | Move the tool tip (vertical) onto ≥ 3 known points, either by jogging (`z -5`, `x +5`, … in mm) or by `disable` and moving it by hand. Then `touch 0`, `touch 1`, `touch 3` (or `touch <x_cm> <y_cm>` for any point you read off `cam`), then `solve`. It prints the residual (aim for < 5 mm), saves `base` to bench_calib.json and prints the `base: [...]` line for configs/laptop.yaml. |
| 5 | `... bench.py pick --dry-run` | Click an object (or press its number), then a destination. PICK/PLACE markers appear, and the terminal prints the arm-frame targets. It never connects. |
| 6 | `... bench.py pick` | Same, then press `y`: the arm enables (first time only) and runs pick then place. Speeds: travel 3 cm/s, descend 1.5 cm/s, joints ≤ 15°/s. **SPACE = stop.** Options: `--height 0.02` (grip height, m), `--approach 0.08`. |

`cam` also draws the arm base once `bench_calib.json` exists. Targets within 12 cm of the base axis, below
table + 5 mm, or out of IK reach are refused before anything moves.

## 2. Safety checklist (every session)
- 24 V supply connected, and **your hand on its switch** (or the e-stop) the whole time.
- A clear area: nothing inside the arm's reach (~55 cm), and no cables in the sweep.
- Always start with `arm-read`: all 7 motors answer, no fault status, temperatures sane.
- First motions: small `j<N> +5` steps. The default cap is 10° per command (`--max-step`).
- **Ctrl-C, `disable`, and quitting all turn the torque OFF, and the arm drops.** Support it first, or park it low.
- `stop` (jog) / SPACE (pick) holds the arm where it is with the torque on.
- Do the first `pick` with `--dry-run`, then for real with a light object.

## 3. Troubleshooting
- **Camera black or wrong:** the index shifts when cameras change. Probe with
  `.venv\Scripts\python -c "import cv2; [print(i, cv2.VideoCapture(i, cv2.CAP_DSHOW).read()[0]) for i in range(4)]"`,
  then pass `--camera N` (0 = integrated with the shutter closed, 2 = OBS virtual camera).
- **NOT CALIBRATED:** all 4 table tags must be in view and uncovered. If the error is > 6 px, a tag moved or is misdetected.
- **No blobs:** press `b` on the EMPTY table first. Shadows and hands cause false blobs, so re-take the background if the lighting changes.
- **`COM5 is not present`:** the tool lists the COM ports it sees. Device Manager > Ports > "USB Serial Device (COMx)"
  (VID 2E88 PID 4603). Pass `--port COMx`.
- **`could not open COMx`:** another program holds the port (another bench.py, a Seeed tool, a serial monitor).
- **NO MOTOR ANSWERED:** 24 V off or the e-stop pressed, the CAN cable from the bridge not seated, or the wrong baud (`--baud`, default 921600).
  **Only some motors answer:** look at the CAN daisy chain at the first silent motor.
- **Status UNDERVOLTAGE / OVERTEMP / COMM LOST:** check the supply, let the arm cool, or reseat the CAN cable. Then re-enable.
- **calib residual > 10 mm:** a bad touch, the tool not vertical, or joint signs/zeros that don't match the URDF (step 3).
- **`not reachable tool-down`:** the point is too far from the base; tag 2 is likely out of reach.
