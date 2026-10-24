# Robot cell: "robot building robot" (in progress)

**Pivot:** SwarmMind's demo moves from wildfire Q&A to an assembly cell. A large arm picks a servo and places it into a slot in an SO-101 arm. Webcams and AprilTags locate everything. Gemma 4 runs on every machine and recognises parts, plans the steps and checks the result. If any computer dies, another Gemma takes over mid-plan and the cell keeps building.

**The swarm layer doesn't change** (election, handoff, tasks, topics: see [PROTOCOL.md](PROTOCOL.md)). New pieces:

| Piece | Path | Status |
|---|---|---|
| Robot-cell system prompt (replaces wildfire) | `swarmmind/protocol.py` | building |
| AprilTag perception: pixels → table → arm coordinates | `swarmmind/cell/perception.py`, `capabilities/cell_vision.py` | building |
| Planner: Gemma JSON plan → validated arm tasks, replicated, resumes on failover | `swarmmind/cell/planner.py`, core hooks | building |
| Arm driver for the large arm (model unknown: pluggable) | `swarmmind/cell/arm.py`, `swarmmind/cell/drivers/*`, `capabilities/cell_arm.py` | building |
| Dashboard cell panel | `dashboard/index.html` | building |

**Known hardware:** Logitech Brio webcam (UVC, `camera_url: usb:0`), SO-101 (the robot being built), AprilTags (tag36h11). The large arm's model isn't known yet, so drivers are selected in config: `mock` (default), `kinova`, `xarm`, `ur`, `so101`.

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
      so101_slot_1: {tag: 20, kind: slot,    offset: [0.02, 0.0, 0.03], insert_depth: 0.02}
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
