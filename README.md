# Koch v1.1: pink box → yellow container

An editable Isaac Lab **3.0** manager-based scene and RL environment, created
for your Stage 5 simulation work. The robot is a public Koch follower model;
prop dimensions and control parameters are provisional. The ACT rollout mode uses your locally cached checkpoint; successful simulation
transfer is not assumed.

## Launch on this computer

```bash
cd /home/niel/Documents/1stProject/koch_isaac
./run.sh
```

`run.sh` uses `/home/niel/miniconda3/envs/isaaclab/bin/python`, even if your
terminal is in `lerobot061`. It does not change either environment or require
an editable installation. Override `KOCH_PYTHON` for another installation.
If running the Python file directly, use the full environment path because
this computer has both Miniconda and Miniforge:

```bash
conda activate /home/niel/miniconda3/envs/isaaclab
python scripts/run_scene.py
```

LeRobot policy training can stay in `lerobot061`; do not merge the environments.

The first launch converts the bundled URDF and meshes to USD. After that,
all scene assets are local; no Nucleus asset downloads are required.

```bash
# Gently exercise each of the six joints around home; this is not a pick policy.
./run.sh --mode joints

# Front RGB sensor and preview file, finite headless run.
./run.sh --headless --steps 90 --camera --snapshot outputs/front_camera.png

# Physics integration checks across four independent copies.
./run.sh --headless --num_envs 4 --steps 90 --check
```

Without a step limit, GUI mode runs until the window closes and headless mode
runs 300 steps. Use one camera environment initially; image observations cost
much more GPU memory than state-only observations.

## What is implemented

- Fixed-base Koch follower, five arm joints and one rotating gripper joint.
- Table, 25 mm pink cube (25 g), yellow open container with a 100 × 90 mm
  interior and 45 mm walls, plus lighting.
- Separate floor and wall colliders: there is no solid collider filling the bin.
- Vectorized environment instances, deterministic seeded resets, modest box
  distance/lateral/yaw variation, 20 s episodes and lost-object termination.
- Small reach/lift/transport/placement rewards; empty curriculum hook.
- Placement success requires full oriented-box containment, resting on the
  container floor, low linear/angular speed, and the gripper TCP moved away,
  continuously for 0.5 s. It is a starter metric, not a grasp/contact classifier.
- Optional 640 × 480 RGB front camera at 30 Hz.

World coordinates are **X forward, Y left, Z up**, in metres. Table top is
`z=0`. The robot's source model is rotated -90° around Z so it faces +X.
This installed Isaac Lab revision uses **xyzw quaternions** and explicit
`.torch` views of Warp-backed data. Do not copy older 2.x quaternion examples.

## Files to edit

| File | Purpose |
|---|---|
| `koch_isaac/settings.py` | Dimensions, initial poses, home joints, approximate TCP, camera viewpoint |
| `koch_isaac/scene_cfg.py` | Table, box, bin collision geometry, materials, camera intrinsics |
| `koch_isaac/robot_cfg.py` | URDF importer, PD gains, actuator torque/speed limits |
| `koch_isaac/env_cfg.py` | Actions, observations, reward weights, termination wiring, timing |
| `koch_isaac/mdp/rewards.py` | Reward functions |
| `koch_isaac/mdp/events.py` | Reset distribution and future physical/visual randomization |
| `koch_isaac/mdp/curriculum.py` | Curriculum terms (empty by default) |
| `koch_isaac/mdp/terminations.py` | Placement metric and lost-object condition |
| `config/baseline.json` | Your chosen ACT baseline and verified input/output contract |
| `scripts/smoke_checks.py` | Physics, control, reset-isolation and placement checks |
| `assets/koch/SOURCE.md` | Asset source, pinned revision, license, modifications and limitations |

Dimensions are read when configs are constructed: restart after editing
`settings.py`. Bin position is static; to randomize it later, move **all five
colliders together** and make the goal/containment calculations use the same
per-environment pose. Changing only a goal observation would be incorrect.

## Environment contract

Gym ID: `Koch-PinkBox-Place-v0` (import `koch_isaac` to register).
Physics: 240 Hz; decimation: 8; policy commands: **30 Hz**.
Actions: `[N, 6]`, **absolute URDF radians**, in this explicit order:

| Action | URDF joint | LeRobot feature name (semantic correspondence only) |
|---|---|---|
| 0 | follower_joint1 | shoulder_pan.pos |
| 1 | follower_joint2 | shoulder_lift.pos |
| 2 | follower_joint3 | elbow_flex.pos |
| 3 | follower_joint4 | wrist_flex.pos |
| 4 | follower_joint5 | wrist_roll.pos |
| 5 | follower_joint_gripper | gripper.pos |

Arm actions clip to ±2.7 rad; gripper clips to [0, 1.5] rad. A zero action
commands zero joint angles, not “do nothing”; use `settings.HOME` for holding.
The limits, gains, approximate TCP, masses, friction and camera placement
must be validated against your physical setup before transfer.

`obs["policy"]` is `[N, 31]`: joint positions (6), velocities (6), local box
position (3), xyzw orientation (4), local TCP (3), local goal (3), previous
action (6). This privileged state is useful for an RL baseline/critic.
Camera mode additionally returns `obs["images"]["front"]`: float RGB
`[N, 480, 640, 3]` in [0, 1].

## Your ACT baseline

Selected model: **JiachenSu/koch11_pick_place_act_home_v2**.
Its locally cached config confirms ACT, `observation.state` shape `[6]`,
`observation.images.front` shape `[3,480,640]`, `action` shape `[6]`,
100-step action chunks, and saved mean/std normalization. The exact cached
revision is recorded in `config/baseline.json`; weights were not copied.

Frozen evaluation is now available with `--mode act`; see the commands below.
The runner converts measured joints using your calibration and uses saved policy
pre/postprocessing. The 31-value privileged RL observation is not passed to ACT.
Training/PPO is not implemented by this rollout feature.

## Suggested progression

First match geometry and replay a simple reachable motion/grasp. Then evaluate
your ACT baseline over a fixed grid of forward distance, lateral displacement,
box yaw and lighting. Your current ~90% frontal success is a useful narrow
baseline; simulation robustness still needs validation on the real robot.

Once the baseline behaves sensibly in simulation, expand pose variation,
then add friction/mass/actuator/latency and camera randomization. Keep a held-out
set of task variations and count success/failure modes. Tune the small starter
reward and curriculum after the geometry and control mapping are trustworthy.

## Compatibility and startup

Built against local Isaac Lab source commit
`bffdce9d7467f349bfc8ab111fe633a0bb234851` (repository VERSION 3.0.0),
Isaac Sim 6.0.1.0, Python 3.12 and PhysX, on an RTX 5090. Package metadata for
`isaaclab` reports 6.1.17; the source commit identifies the tested API precisely.

The launcher starts Kit **before** importing the task config. On this local
installation, importing USD-dependent task modules first caused a native
protobuf/OpenBLAS startup crash. `run.sh` also limits BLAS threads. Simulation
exceptions are printed and preserved as nonzero exit codes through Kit shutdown.
No simulator installation or existing projects were modified.

Public asset reference: https://github.com/Maverobot/koch_ros
Isaac Lab reference: https://github.com/isaac-sim/IsaacLab

## Rendering synchronization

Fabric is enabled by default for both GUI and headless runs. On this installation,
turning it off left the rendered robot at its initial pose even though physical
joint readings changed. This also produced stale robot poses in camera images.
An earlier CUDA shim error motivated that workaround, but camera tests with Fabric
on subsequently rendered the commanded shoulder and gripper correctly without
that error. `--fabric off` remains a diagnostic option for preview only;
ACT evaluation rejects it because stale images would invalidate the policy input.

## Measured joint conversion and preview

`scripts/utils.py` uses provisional encoder zeros
`[2105, 1615, 1768, 2630, 1405, 1447]` in the order
shoulder pan/lift, elbow flex, wrist flex/roll, gripper. Positive counts mean
positive URDF angles. Shoulder lift, elbow and wrist flex still need reference
checks (second-point discrepancies about 5.4, 5.5 and 4.0 degrees).
The gripper assumes a directly driven rotating finger, not jaw width.

`encoder_to_sim` / `sim_to_encoder` convert reported motor counts and
absolute URDF radians. Do not add homing offsets again. These functions do not
clip or wrap. `sim_to_lerobot` / `lerobot_to_sim` additionally use the
JSON ranges to match LeRobot motor values (arms [-100,100], gripper [0,100]).
The latter follows LeRobot action clipping and integer truncation. Pass
`use_degrees=True` only for datasets recorded with that option: LeRobot's
degrees convention uses 4095 in its software normalization, whereas the physical
encoder conversion uses 4096 counts/revolution.

The runner reads the original follower calibration JSON, and rejects a changed
motor ID, homing offset or drive mode so new calibration is not silently mixed
with the old measured zeros. Override its path with `--calibration PATH`.
No physical robot connection is made.

```bash
cd /home/niel/Documents/1stProject/koch_isaac
# Hold an encoder-specified pose, with an open gripper; print measured joints.
./run.sh --encoder-target 2105 1615 1768 2630 1405 2100 --print-joints-every 30

# Alternatively supply postprocessed LeRobot motor values.
./run.sh --lerobot-target 0 0 0 0 0 75 --print-joints-every 30

# Unit checks without launching Kit.
/home/niel/miniconda3/envs/isaaclab/bin/python -m unittest discover -s tests -v
```

Both target options require hold mode. Existing environment action limits still
apply. Normal preview and joint exercise remain in radians. Final output includes
`joint_conversion` for measured environment-zero joints, with equivalent
counts, LeRobot state and flags for readings outside the calibration ranges.
The normal RL observation vector is unchanged; do not feed its 31 values to ACT.

## Run your ACT baseline

```bash
cd /home/niel/Documents/1stProject/koch_isaac

# Watch five full episodes and save the front-camera video.
./run.sh --mode act --episodes 5 --video

# Evaluate without a GUI. Each episode uses the current 20-second task limit.
./run.sh --mode act --headless --episodes 20 --video

# Short plumbing check. --steps stops early; partial episodes are not successes.
./run.sh --mode act --headless --steps 120
```

The default policy is the cached home_v2 revision in `config/baseline.json`.
No download, training, checkpoint modification or physical robot connection is
performed. Supply `--policy-path /absolute/path/to/pretrained_model` for another
compatible local ACT checkpoint. Its saved preprocessor and postprocessor must
be present alongside the weights. The loader checks six state/action values and
a single RGB front image [3,480,640], and strictly loads checkpoint weights.

Isaac stays in its existing environment. The launcher automatically starts a
private local worker using
`/home/niel/miniforge3/envs/lerobot061/bin/python`. Override with
`--policy-python PATH`. Use `--policy-device cpu` if desired; it affects
only inference, while `--device` controls the simulator. Only one simulated
environment is supported by this first rollout implementation.

The closed loop is:

1. Read actual joint positions in the named six-joint order and convert them with
   `sim_to_lerobot`. Range units are the default; use
   `--lerobot-use-degrees` only if the training robot used that convention.
2. Send the measured state and current uint8 RGB frame to the worker. Convert RGB
   to CHW floats in [0,1], then apply the saved preprocessor exactly once.
3. Call `ACTPolicy.select_action` with frozen weights. Preserve the checkpoint's
   100-action chunk queue, then apply the saved action postprocessor exactly once.
4. Convert those six motor values through `lerobot_to_sim` and the configured
   simulation joint limits. Advance 1/30 simulated second. Wall-clock speed may
   be slower; this synchronous evaluation does not drop policy actions.
5. At every automatic episode reset, clear the ACT queue and reset both processors
   before the next observation is used.

Results go to `outputs/act_TIMESTAMP/`, or `--output-dir PATH`:

- `summary.json`: completed episodes, success rate, timeout/lost-box outcomes,
  clipped-action counts, state-range violations and any unfinished episode.
- `rollout.jsonl`: checkpoint/calibration metadata, periodic states/actions,
  inference latency and per-episode results. `--log-every N` changes the interval.
- `first_frame.png` / `last_frame.png`: the policy camera view.
- `front.mp4`: optional video at 30 simulated fps, enabled with `--video`.

`--seed` defaults to 42. `--episode-seconds` can override the task horizon
for diagnostics; keep the default horizon for comparable baseline evaluations.
The reported success uses the existing stable-placement termination, not reward
alone. An interrupted or step-limited partial episode is excluded from the success
rate. Zero completed episodes reports a null success rate.

Camera viewpoint/intrinsics, appearance, dimensions, actuator tracking and joint
references are still provisional. Check the first-frame image against the real
training camera and inspect actions before interpreting simulation success as
policy quality. The previous pose test showed roughly 6.5 degrees of shoulder-lift
tracking error under load. A low simulation score does not establish low real-world
performance, and this ACT checkpoint is not directly a PPO actor checkpoint.

Implementation: `scripts/act_bridge.py` provides a private socket transport,
`scripts/act_worker.py` performs LeRobot inference, and
`scripts/act_rollout.py` owns the simulation evaluation loop.
