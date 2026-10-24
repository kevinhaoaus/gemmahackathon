# Robot cell: "robot building robot"

**Pivot:** SwarmMind's demo moves from wildfire Q&A to an assembly cell. A large arm picks a servo and places it into a slot in an SO-101 arm. Webcams and AprilTags locate everything. Gemma 4 runs on every machine and recognises parts, plans the steps and checks the result. If any computer dies, another Gemma takes over mid-plan and the cell keeps building.

**The swarm layer doesn't change** (election, handoff, tasks, topics: see [PROTOCOL.md](PROTOCOL.md)). New pieces:

| Piece | Path | Status |
|---|---|---|
| Robot-cell system prompt (replaces wildfire) | `swarmmind/protocol.py` | built, tested in sim |
| AprilTag perception: pixels → table → arm coordinates | `swarmmind/cell/perception.py`, `capabilities/cell_vision.py` | built, tested in sim |
| Planner: Gemma JSON plan → validated arm tasks, replicated, resumes on failover | `swarmmind/cell/planner.py`, core hooks | built, tested in sim |
| Arm driver for the large arm (model unknown: pluggable) | `swarmmind/cell/arm.py`, `swarmmind/cell/drivers/*`, `capabilities/cell_arm.py` | built, tested in sim |
| Dashboard cell panel | `dashboard/index.html` | built, tested in sim |

**Known hardware:** Logitech Brio webcam (UVC, `camera_url: usb:0`), SO-101 (the robot being built), AprilTags (tag36h11). The large arm's model isn't known yet, so drivers are selected in config: `mock` (default), `kinova`, `xarm`, `ur`, `so101`.

**Verified in the simulator with real Gemma 4** (`gemma4:e4b`, `simulate.py --real` setup): goal → Gemma writes a valid JSON plan (`pick servo_1 → place so101_slot_1 → verify`) → the mock large arm runs pick and place → Gemma checks the annotated frame and answers `{"ok": true}`. About 15 s end to end, no retries. With mock LLMs the test suite also kills the leader mid-plan-JSON and mid-place and checks the plan completes without re-running finished steps. Real-arm drivers (Kinova/xArm/UR) are untested on hardware.

## Layout rules learned the hard way

- **Put each slot's tag beside the slot, not under it.** The slot point must sit at least one tag width plus half a part's width from its tag (`offset` ≈ 7 cm for 4 cm tags). Otherwise a seated part covers the slot's tag, both become unreadable, and verify (correctly) says the part isn't there.
- **Verify waits for a post-move frame.** The planner only accepts a camera frame captured ≥ 0.3 s after the arm finished (up to 3 s), so Gemma never judges the scene from before the place.
- **Camera high and roughly overhead.** Positions come from a table-plane homography, so tags on tall parts read a few mm off near the edge of the image.

## Design rule

**Gemma decides *what*; geometry decides *where* and *how*.**

- Gemma picks object **names** from a whitelist, the step order, and gives yes/no verification from the camera image.
- Gemma never outputs coordinates.
- Positions come from AprilTags plus calibration. Motion is deterministic: approach above, descend slowly, grip, retreat.
- Every target is bounds-checked, and `stop` cancels immediately.

## Contract between the pieces

### Cell config (shared YAML block, identical on every node)

Lives in `capability_config["cell"]`, so every node, and therefore any leader, knows the cell layout.

```yaml
capability_config:
  cell:
    tag_size_m: 0.04                   # printed black-square edge length
    table_tags: {0: [0.0, 0.0], 1: [0.60, 0.0], 2: [0.60, 0.40], 3: [0.0, 0.40]}   # id: [x, y] metres, table frame
    objects:                           # everything Gemma may name; each anchored to a tag
      servo_1:      {tag: 10, kind: part,    offset: [0.0, 0.0, 0.015], grasp_yaw: 0}
      so101_slot_1: {tag: 20, kind: slot,    offset: [0.07, 0.0, 0.03], insert_depth: 0.02}   # slot point 7 cm from its tag: a seated part must not cover the tag
      tray_a:       {tag: 11, kind: fixture, offset: [0.0, 0.0, 0.0]}
    arms:                              # table frame -> arm base frame (x, y, yaw_deg, z_table)
      arm.big: {base: [0.30, -0.25, 90.0, 0.0], approach_h: 0.08, speed: 0.05}
```

### Perception → topic `cell.state` (published by the camera node about 2×/s)

```json
{"frame_ts": 0.0, "camera": "thor", "calibrated": true,
 "tags": [{"id": 10, "px": [u, v], "corners": [[u, v], ...], "table_xy": [x, y], "yaw_deg": 0.0}],
 "objects": [{"name": "servo_1", "kind": "part", "tag": 10, "visible": true, "table": [x, y, z], "yaw_deg": 0.0}],
 "reproj_err_px": 0.8}
```

`swarmmind.cell.perception` also exposes `table_to_arm(xyz_table, arm_cfg) -> (x, y, z)` and `annotate(jpeg, state) -> jpeg` (tag outlines plus object names, for Gemma and the dashboard).

### Arm capability `arm.<id>` (e.g. `arm.big`), actions in arm-base metres and degrees

```text
home | move_to {x,y,z,yaw} | gripper {value 0..1} | pick {x,y,z,yaw,approach_h} | place {x,y,z,yaw,approach_h,insert_depth} | stop
```

The driver interface is `Arm`: `connect`, `home`, `move_to`, `gripper`, `stop`, `pose`, plus `workspace` bounds. The arm publishes `arm.<id>.state`.

### Planner

1. `POST /cell/goal {"goal": "insert the servo into the SO-101"}` on any node is forwarded to the leader.
2. Gemma returns `{"steps": [{"op": "pick", "object": "servo_1"}, {"op": "place", "target": "so101_slot_1"}, {"op": "verify", "check": "servo seated in slot"}]}`.
3. The plan is validated (ops whitelist, names from `objects`, pick→place pairing), then compiled to arm tasks using the `cell.state` coordinates and executed step by step on the task bus.
4. Plan state is replicated (`plan.new`, `plan.update`, leader-only, included in sync). A new leader resumes at the current step.
5. Messages pushed to the UI: `plan.new` / `plan.update` with the body `{plan_id, goal, steps:[{op, object|target, state, task_id, result}], state, step, leader}`.

### Shared seams (who sets what)

- The perception capability sets `node.cell_state` (the latest `cell.state` dict) and `node.cell_annotated_jpeg` on its own node, and publishes `cell.state`. Every other node caches the latest `cell.state` from the topic in `node.topics.latest["cell.state"]`.
- The planner reads `node.topics.latest["cell.state"]["data"]` (or `node.cell_state` locally). Arm coordinates are always computed with `swarmmind.cell.perception.table_to_arm`.
- HTTP endpoints added by the planner work: `POST /cell/goal`, `GET /cell/plans`, `POST /cell/plans/{id}/cancel`, `GET /cell/frame.jpg` (the newest annotated frame: local, else fetched from the camera node).
- Simulator: `sim_a` runs `vision.cell` with a synthetic tag scene (`source: mock`), and `sim_b` runs `arm.big` with `driver: mock`.
