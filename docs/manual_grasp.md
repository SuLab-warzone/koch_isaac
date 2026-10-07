# Manual open–align–close–lift test

This mode runs one user-defined sequence with no ACT, diffusion, PPO, or physical
robot connection. It answers whether the configured geometry and actuators can
retain a grasp. It does not find a grasp pose automatically or change rewards.

## Define the poses

Copy `config/manual_grasp.template.json` to `config/manual_grasp.json`, then fill
all four six-value arrays. The template intentionally contains null placeholders:
it will fail before launching Isaac until you supply measured poses. No picking
trajectory has been validated for this asset.

Joint order is shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll,
gripper. `units` can be `radians` (absolute URDF angles), `encoder` (reported motor
counts), or `lerobot` (postprocessed motor values). LeRobot arms use [-100,100]
and gripper [0,100]; `--lerobot-use-degrees` changes arm units only. The existing
follower calibration file is still required for conversion and joint reports.
Use `--calibration PATH` if it is not in the default cache location.

| Pose | What to measure |
|---|---|
| open | Arm clear of the box and table; jaw open enough to surround the box |
| align | Box between finger pads; same gripper value as open |
| close | Same five arm values as align; only change the gripper |
| lift | Raise the box gently; keep the same gripper value as close |

Find an initial arm pose from `Joint state` → `sim_radians` in an existing policy
run (`--print-joints-every 1`). Tune it with hold mode, without a policy:

```bash
# Replace the six values with your measured pose; these zeros are not a grasp pose.
./run.sh --mode hold --radians-target 0 0 0 0 0 1 \
  --steps 150 --print-joints-every 30
```

Adjust one joint in small increments and view the jaws from above and from the
side. Confirm that the pads overlap the 8 mm-tall box and do not hit the table.
Use collision visualization in your simulator to inspect the actual contact
shapes, rather than relying only on the rendered meshes. Repeated hold commands
reset the scene; use the sequence below to test grasp retention in one run.

## Run the sequence

From the repository root on the Ubuntu Isaac machine:

```bash
./run.sh --mode grasp-test --grasp-sequence config/manual_grasp.json \
  --grasp-hold 1.0 --print-joints-every 30 --rerun
```

Omit `--rerun` if its optional SDK is unavailable. Use `--headless` for Isaac
without its GUI (Rerun still opens its own viewer). All four waypoints are checked
against simulation joint limits before launch; this mode rejects clipping.

The robot starts from its measured reset joints and interpolates to each pose
for `move_seconds`, then holds for `hold_seconds`. Defaults are 2 s and 1 s:
four phases, 12 s total. The run stops after one sequence. Its episode timeout is
extended beyond that duration. If a task termination occurs, the run stops and
reports its pre-reset metrics instead of replaying commands into a reset scene.
`--steps N` may shorten a run for inspection; it cannot extend it. A shortened
or closed-window run is reported INCOMPLETE. Increase JSON hold time to observe
the final pose longer; keep it at least as long as `--grasp-hold`.

Existing box-filtered contact sensors and Rerun force/jaw/height plots are enabled
in this mode. Console diagnostics report cumulative contact statistics and peak
lift at the joint-print interval. The final JSON includes contact statistics,
grasp outcome, sequence completion, and continuous grasp duration at the end.
No policy files or LeRobot worker interpreter are needed.

PASS requires a completed sequence and a grasp still meeting the existing
contact, proximity, height, and duration thresholds at the end. With the command
above that means both fingers contact the box, its bottom is at least 25 mm above
the table, it is near the TCP, and these conditions persist for 1 s. An earlier
grasp followed by a drop does not pass. These metrics remain provisional: inspect
the motion to confirm an opposing pinch rather than a trapped or bounced box.

If an automatic reset occurred, `state_after_auto_reset` is true: top-level joint
and box coordinates are reset state; the nested `grasp` record is pre-reset.
No automatic retry is performed. Snapshots (`--snapshot PATH`) are supported,
but this mode does not write rollout videos or JSONL files.

## Interpret a failed trial

- No finger contact: inspect TCP, jaw opening, grasp height, and collision shapes.
- Only one finger contacts: inspect lateral position and jaw orientation.
- Both fingers contact but the box slips: inspect closure, friction, and lift speed.
- Measured gripper angle does not follow its target: inspect table interference,
  actuator effort, and joint limits.
- Scripted grasp succeeds but PPO fails: inspect policy timing, observations,
  residual correction limits, and reward shaping next.
