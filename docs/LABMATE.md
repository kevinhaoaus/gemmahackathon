# LabMate: research-assistant arm (contract, in progress)

An arm on a lab bench that:
- **sees** what's on the table (springs, screwdrivers, files, superglue, ...);
- **works out which experiment** the setup is for and which tools are missing;
- **clears hazards** (e.g. an open superglue bottle) into a safe tray;
- **fetches items** on request ("hand me the screwdriver").

Gemma 4 runs on every computer. The swarm keeps working if a computer or an arm dies. Builds on [ROBOT_CELL.md](ROBOT_CELL.md) (perception, arm interface, planner) and [PROTOCOL.md](PROTOCOL.md).

## Hardware layout (for now)

```
Brio webcam (overhead) ──USB──┐
reBot B601 arm ─────────USB───┤ Legion (Windows, RTX 4070)  "hands + eyes": vision.cell + arm.cell, rank 100
                              │
Jetson AGX Thor ──────────────┘ LAN                         "brain": leader, Gemma, rank 200
```

- **Leadership:** the Thor leads, being the fastest Gemma. A human at the dashboard can always stop or cancel (override B).
- **Pulling the Thor's cable:** the Legion becomes leader. The arm and camera are on the Legion, so they keep going.
- **One arm:** there is no backup arm. If the reBot or its host dies, LabMate keeps seeing the bench and reports each hazard it could not move as "NOT fixed; a human must <fix>".

## Rules

1. **Gemma names things; code places things.** Gemma labels numbered marks and picks the experiment from the fixed list. It uses a catalog name when one fits, and otherwise names the object itself (`scissors`, `tape_measure`, `beaker`: open vocabulary). A name outside the catalog gets a default grip (kind `other`, 3 cm high, grasped across if the blob is long and thin), and the two labelling passes still agree if they word it a little differently (`tape_measure` / `measuring_tape` ignore word order and plurals). `robot_arm`, `robot` and `gripper` are never picked up. Positions come from the AprilTag-calibrated table. Gemma never outputs coordinates.
2. **Arm tasks carry TABLE coordinates plus an item reference, never arm coordinates.** Each arm converts to its own frame with its own `base` calibration, and **re-finds the item in the latest `cell.blobs` before moving**. If the item isn't within 4 cm of where it was expected, the task fails with "object not where expected". It never grabs blindly after a handover.
3. **Plug-in detection.** The `arm.cell` capability is advertised only while the arm is connected and answering (`tasks.set_availability`).
4. **Hazards** are a fixed list. Each type is **movable** (the arm may relocate it) or **report-only** (never touched). Ranking is deterministic, safety first. Before moving a hazard, the robot announces it and waits `confirm_s` (default 5 s) for a cancel.
5. **Two sightings:** an item or hazard counts only if Gemma labels it the same way in two labelling passes on frames at least 1 s apart. There's no wrist camera, so this replaces "two angles".

## Catalog: `capability_config.cell.catalog` (identical on every node)

```yaml
capability_config:
  cell:
    # ... tag_size_m, table_tags, objects (tagged fixtures), arms ... as in ROBOT_CELL.md
    objects:
      safe_tray: {tag: 30, kind: fixture, offset: [0.07, 0.0, 0.0]}     # hazards go here
      handover:  {tag: 31, kind: fixture, offset: [0.07, 0.0, 0.0]}     # fetched items go here
    catalog:
      items:                      # everything Gemma may name. height = grip height above the table (m)
        screwdriver:    {kind: tool,       height: 0.012, grasp: across}   # across = grip across the long axis
        file:           {kind: tool,       height: 0.008, grasp: across}
        spring:         {kind: part,       height: 0.010, grasp: center}
        superglue:      {kind: consumable, height: 0.060, grasp: center}
        stepper_motor:  {kind: part,       height: 0.030, grasp: center}
        motor_driver:   {kind: part,       height: 0.015, grasp: center}
        power_supply:   {kind: equipment,  height: 0.050, grasp: none}     # none = never picked
        multimeter:     {kind: tool,       height: 0.030, grasp: center}
        ruler:          {kind: tool,       height: 0.003, grasp: across}
        stand:          {kind: equipment,  height: 0.200, grasp: none}
      hazards:                    # the 7 fixed types
        open_container:    {movable: true,  severity: 3, fix: "cap it and move it away from electronics"}
        item_near_edge:    {movable: true,  severity: 2, fix: "move it back from the bench edge"}
        sharp_tool_loose:  {movable: true,  severity: 2, fix: "store the tool, tip away from people"}
        clutter_in_path:   {movable: true,  severity: 1, fix: "clear the arm's working area"}
        cable_near_motion: {movable: false, severity: 3, fix: "route the cable away from moving parts"}
        hot_tool:          {movable: false, severity: 4, fix: "let it cool on its stand; do not touch"}
        spill:             {movable: false, severity: 4, fix: "stop and clean the spill before power-on"}
      use_cases:                  # experiments LabMate knows; anything else gets general notes only
        spring_pendulum: {title: "Spring pendulum",    needs: [spring, stand, ruler],                  optional: [multimeter]}
        stepper_test:    {title: "Stepper motor test", needs: [stepper_motor, motor_driver, power_supply, screwdriver], optional: [multimeter]}
        deburring:       {title: "Filing / deburring", needs: [file],                                  optional: [screwdriver]}
```

## Perception additions → topic `cell.blobs` (camera node, about 2×/s)

A "background" snapshot of the empty table is taken on request (`vision.cell` task action `background`). Afterwards every new region on the table is a blob, measured on the **rectified top-down table image** (metric directly through the homography):

```json
{"frame_ts": 0.0, "calibrated": true, "background": true,
 "blobs": [{"id": 3, "table_xy": [x, y], "yaw_deg": 0.0, "length_m": 0.18, "width_m": 0.02,
            "area_m2": 0.0036, "stable_frames": 12}]}
```

- `id` is a **stable track id** (nearest-neighbour tracking across frames), so Gemma's labels stay attached.
- `yaw_deg` is the long-axis direction, so "across" grasps are perpendicular to it.
- `node.cell_marked_jpeg` holds the frame with big numbered marks (`#3`) drawn on the blobs, for Gemma and the dashboard. It's served at `GET /cell/marked.jpg`.

## Arm capability `arm.cell` (shared by every arm; one executor per task)

Config `capability_config["arm.cell"]`: `{driver: rebot | so101 | mock, arm_id: "rebot", base: [x, y, yaw_deg, z_table], workspace, speeds..., port: "COM5"}`. Actions, all in the **table frame**:

```text
move_object {item, blob_id, table_xy:[x,y], yaw_deg, height, grasp, to:{table_xy:[x,y], yaw_deg, z}}
   -> relocate (nearest blob ≤ 4 cm in the latest cell.blobs), table→arm with this arm's base,
      pick at height (across: gripper yaw = blob yaw + 90°), carry upright and slowly, place, retreat
home | stop
```

The arm publishes `arm.cell.<arm_id>.state`. When several nodes offer `arm.cell`, the leader assigns by node rank, then by load. On the Legion, the reBot is the only `arm.cell` for now.

## LabMate flows (leader; replicated plans that resume on failover)

**Inspect: `POST /lab/inspect`**
1. Label with Gemma: the marked frame goes to a `kind:"label"` job. Gemma answers JSON `{"items":[{"mark":3,"item":"superglue","state":"open"}], "hazards":[{"type":"open_container","marks":[3]}], "use_case":"stepper_test"|"unknown"}`, validated against the catalog (hazards and the use case must match; item names may be free-form). This runs twice on frames ≥ 1 s apart and keeps only what agrees.
2. Rank hazards in code by severity, then by distance to the bench edge.
3. For each **movable** hazard: announce, wait `confirm_s`, then `move_object` to `safe_tray`, then re-label to verify.
4. Use case: candidates from the catalog plus Gemma's guess, computed in code. `missing = needs − seen`.
5. Ask the question (the plan state becomes `awaiting_answer`): "I see ...; I think this is <title>. Is that right, and what is this run for?"

**Answer: `POST /lab/answer {text}`.** Gemma turns the answer into `{"experiment": str, "confirmed": bool, "phase": "first_test"|"final_data"|"unknown", "noise_ok": bool|null}`. Then a final report job (text): hazards found and fixed, the experiment, missing tools, and notes adjusted for the phase.

**Fetch: `POST /lab/fetch {item}`.** Use the latest agreed labels to find the blob, then `move_object` to `handover`. If the item isn't found: "I can't see a <item>".

The UI receives `plan.new`/`plan.update` with a `lab` field: `{items:[{mark, item, state, table_xy}], hazards:[{type, marks, severity, movable, fix, status}], use_case:{id, title, confidence, missing:[...]}, question, answer, report}`.
