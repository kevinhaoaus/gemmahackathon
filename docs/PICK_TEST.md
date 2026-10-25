# Pick-and-place test (Windows laptop, reBot, bench camera)

Run everything from the repo root in PowerShell. Nothing else may be using the camera or COM5 (close any swarm node, `jog-gui`, or other `bench.py` window first).

## State checked before the test
- Arm adapter present on **COM5**; motor SDK installed.
- Bench camera = **index 1** (the `--camera` default). It currently sees tags 0, 1, 2 (and 11): get **tag 3 into view too** so all four table corners are seen.
- Calibration saved in `data/bench/bench_calib.json`: arm base on the table at **x −11.7 cm, y 29.7 cm** (left of tag 0), gripper OPEN/CLOSED angles saved. So real picks are allowed.
- Gripper rule: it closes to **object width − 9 mm** (your setting). Change with `--squeeze-mm` (smaller = looser, larger = tighter).
- The Jetson link works (192.168.50.1, 1 ms) but is **not needed** for this test.

## Before the arm moves
24 V supply on · area around the arm clear · a hand on the power switch.
`SPACE` = stop and hold position. **Ctrl-C, `q` or closing = torque OFF, the arm goes limp**: support it first.

## Steps
1. **Camera + calibration**
   ```powershell
   .venv\Scripts\python scripts\bench.py cam
   ```
   - Badge says **CALIBRATED**. Click a spot: its x/y in cm should match a ruler measured from tag 0 (x toward tag 3, y toward tag 1).
   - **Empty the table, hands out, press `b`** (captures the background). Then put your object down: it gets a numbered box. Press `q`.
2. **Optional: motor check (reads only, nothing moves)**
   ```powershell
   .venv\Scripts\python scripts\bench.py arm-read
   ```
   All 7 motors should answer.
3. **Dry run (plans, never moves)**
   ```powershell
   .venv\Scripts\python scripts\bench.py pick --dry-run
   ```
   Press `b` on the empty table → place the object → click the **object** (or press its number) → click the **destination**. Terminal prints the object width, the gripper gap it will use, and the arm targets, and says nothing is refused.
4. **Real pick and place**
   ```powershell
   .venv\Scripts\python scripts\bench.py pick
   ```
   Same clicks, then press **`y`**. The arm opens the gripper, goes above the object, descends slowly, closes to width − 9 mm, lifts, carries level, places, and retreats. First try something light and 3 to 5 cm wide (a marker, a small box).

## Rules that avoid refusals
- **Click an object the camera has boxed**, not an empty spot. A raw click has no width, so the tool refuses to run (`NO WIDTH`) instead of closing the gripper fully. Manual override: `--width-cm 4`.
- **Stay in reach:** keep object and destination inside the four tags and within about **40 cm of the arm base**. Anything farther is refused before moving (`not reachable tool-down`). Your earlier click at table (−36.5, 70.6) cm was 48 cm out.
- Tag 3 hidden or the camera moved → `NOT CALIBRATED`; fix the view, press `b` again after moving anything.

## Tuning flags (all optional)
`--squeeze-mm 5` looser · `--squeeze-mm 12` tighter · `--height 0.015` grip lower (thin objects) · `--approach 0.10` higher approach · `--camera N` if the camera number changed.

## If something is off
| Symptom | Fix |
|---|---|
| `could not open COM5` | close other programs using it, or replug the adapter |
| a motor shows NO FEEDBACK | 24 V supply on? e-stop released? CAN cable seated? |
| camera shows the wrong picture | `--camera 0/1/2`; index 1 is the bench camera now, 2 is the OBS fake |
| gripper won't close/open | re-save the angles in `jog-gui` (Save as OPEN / CLOSED), pick refuses without them |
| arm pushes too hard / not enough | adjust `--squeeze-mm` |
