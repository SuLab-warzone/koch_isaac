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
Residual PPO is available on the residual-ppo branch; see the commands below.

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

## Evaluate a frozen policy

Run commands from /home/niel/Documents/1stProject/koch_isaac. run.sh uses the
Isaac Python interpreter and starts a separate LeRobot inference worker. No
physical robot connection is made. No checkpoint is downloaded or changed.

```bash
# Watch five episodes and save logs, camera images and video.
./run.sh --mode policy --episodes 5 --video

# Evaluate without writing rollout files.
./run.sh --mode policy --headless --episodes 20 --no-save-output

# Two environments, five completed episodes EACH. Files are automatically disabled.
./run.sh --mode policy --headless --num_envs 2 --episodes 5

# Select another local checkpoint explicitly.
./run.sh --mode policy --policy-path /absolute/path/to/pretrained_model --no-save-output
```

--mode act remains an alias for frozen-policy evaluation. --policy-path is the
only base-checkpoint selector; its config determines ACT or diffusion architecture.
There is no automatic checkpoint switching. The default remains the cached
home_v2 revision in config/baseline.json. Both architectures must provide six
state/action features, one front RGB image [3,480,640], and saved pre/postprocessors.
Temporal-ensemble ACT and stateful/custom processors are currently rejected.
Actual checkpoint validation has been performed with ACT; a diffusion checkpoint
has not yet been supplied for end-to-end verification.

--policy-python defaults to /home/niel/miniforge3/envs/lerobot061/bin/python;
--policy-device (default cuda) controls BOTH frozen inference and PPO; --device controls Isaac, and
--policy-batch-size (default 4, maximum 8) bounds inference microbatches.

Measured joints are converted with the follower calibration. RGB is converted
to CHW [0,1], then the checkpoint preprocessor is applied once. Chunk prediction
runs only when an environment needs new actions. Each environment owns its own
chunk cursor and observation history, cleared on its own reset. The saved action
postprocessor is applied once before conversion back to absolute URDF radians.
Diffusion history advances every control step, including steps using cached actions.
Control advances at 30 simulated Hz; wall-clock evaluation can be slower.

For one environment, --save-output (default) writes summary.json, rollout.jsonl,
first_frame.png, last_frame.png, and optional front.mp4 beneath outputs/, or
--output-dir PATH. --no-save-output creates none of those files and disables
--snapshot. With --num_envs greater than one these files, videos and snapshots
are always disabled, even if explicitly requested. Console summaries remain.
Isaac/Kit may still write its own runtime diagnostic logs outside the project.
Training checkpoints are controlled separately and remain enabled.

--episodes is a per-environment quota; faster environments do not bias evaluation
by contributing extra completed episodes. --steps caps vector steps, and partial
episodes do not enter success rates. --seed defaults to 42. --episode-seconds
changes the task horizon for diagnostics; leave the default 20 seconds when
comparing policy performance.

## Grasp and placement metrics

- grab_success: both fingers contact the box above --grasp-force (default 0.01 N
  per finger), its lowest point is at least --grasp-height (0.025 m) above the
  table, and the box remains near the TCP, continuously for --grasp-hold (0.2 s).
  This is remembered for the remainder of the episode.
- placement_success: at episode end the whole box is inside the container,
  resting near its floor, below the rim, moving slowly, and the TCP is clear.
  This must hold continuously for 0.5 s; earlier placement that is subsequently
  disturbed is not counted. Stable placement ends the episode early.
- pick_place_success: both conditions occurred in the same episode.

The console/summary report separate grasp and placement rates, timeouts, lost
boxes and clipping counts. They capture terminal state before Isaac automatically
resets. Contact reporting is enabled explicitly on both nested Koch finger bodies.
Thresholds and TCP geometry remain provisional; verify against representative
successful and failed grasps before using these as real-world ground truth.

## Residual PPO

The base ACT/diffusion policy stays frozen. PPO learns six corrections:

```text
command_radians = clip(base_command_radians + residual_limit * residual, joint_limits)
residual is clipped to [-1,1]; default residual_limit = 0.25 rad per joint.
```

The residual actor receives joint angles, velocities, the current base command,
previous correction, and chunk phase (25 values). The separate critic sees the
31-value privileged simulator state. The actor cannot read box coordinates from
the critic input. Deterministic initial corrections are zero; PPO training samples
small stochastic corrections. The starter task reward is retained, with a small
squared-correction penalty controlled by --residual-penalty (default 0.05).
This is a starting implementation for testing; it does not guarantee improved
success or real-robot robustness. Critic inputs omit the full base-policy chunk
history, and the actor has no direct visual features beyond the base command.

```bash
# Short plumbing/gradient check, not a meaningful performance experiment.
./run.sh --mode residual-ppo --train --headless --num_envs 2 \
  --total-timesteps 256 --ppo-steps 32 --ppo-batch-size 64 \
  --episode-seconds 2 --checkpoint-out checkpoints/my_smoke.zip

# Longer training with normal 20-second episodes; checkpoint only is saved.
./run.sh --mode residual-ppo --train --headless --num_envs 2 \
  --total-timesteps 100000 --checkpoint-out checkpoints/residual_100k.zip

# Evaluate the learned correction with the same base checkpoint and limits.
./run.sh --mode residual-ppo --headless --num_envs 2 --episodes 20 \
  --checkpoint checkpoints/residual_100k.zip --seed 123

# Resume training into a NEW checkpoint file.
./run.sh --mode residual-ppo --train --headless --num_envs 2 \
  --checkpoint checkpoints/residual_100k.zip --total-timesteps 100000 \
  --checkpoint-out checkpoints/residual_200k.zip
```

--total-timesteps counts transitions across all environments and is rounded up to
complete PPO rollout batches. --ppo-steps is per environment; --ppo-batch-size
must divide num_envs * ppo_steps. Both models use --policy-device (cuda by default).
The separate --ppo-device option has been removed. Use --policy-device cpu to run
both models on CPU; the simulator device remains independently controlled by --device.
Saved checkpoints include PPO optimizer state and the base-policy/calibration
contract; they do not bundle the frozen base model. A resume starts fresh simulator
episodes, not an exact continuation of an interrupted trajectory. Existing output
checkpoint paths are rejected to prevent overwriting experiments.

When changing --policy-path, train a new residual. Loading a residual against a
different base path, calibration, or correction limit is rejected. Keep the same
--residual-limit and --lerobot-use-degrees when evaluating/resuming.

Compare frozen and residual runs at identical seeds, environment counts, episode
lengths and scene settings, then repeat over held-out seeds. Low simulation success
can reflect camera, geometry, calibration or actuator mismatch; learning corrections
to those mismatches alone does not establish better physical-robot performance.

## Git experiments

This folder is an independent local Git repository. main and tag act-baseline
preserve the pre-residual version (12b356f). residual-ppo contains this approach.
No remote repository or push was created. Generated USD, rollout outputs and
checkpoints are ignored. The enclosing project repository is unchanged.

```bash
git status
# Create a future experiment from the original baseline:
git switch -c another-approach act-baseline
# Return to the residual implementation:
git switch residual-ppo
```

Implementation: scripts/policy_session.py shares vector stepping and conversion;
scripts/act_worker.py owns frozen inference; scripts/residual_ppo.py implements
training; koch_isaac/evaluation_env.py captures terminal metrics and critic inputs.
See VALIDATION.md for actual checks and their limits.

## Understanding the run settings

The frozen ACT/diffusion and residual PPO are two different models. --policy-path
loads the base model and its saved processors; --checkpoint loads the learned
correction, critic, and PPO optimizer state. New residual training needs only the
base model. Evaluation or resumed training of a learned residual needs both.
The default cached ACT path means it can be omitted when using that same baseline.
A combined deployment folder could package both models, but the PPO checkpoint
does not replace the ACT/diffusion checkpoint.

| Option | Purpose |
|---|---|
| --episodes | Completed evaluation attempts per environment; not used as a training stop condition |
| --episode-seconds | Maximum simulated duration of each attempt; success/failure may end it earlier |
| --total-timesteps | Total training transitions across all environments |
| --ppo-steps | New transitions collected per environment before each PPO optimization round |
| --ppo-epochs | Optimization passes over each collected rollout batch; no extra simulator steps |

For example, 2 environments with --ppo-steps 64 collect 128 transitions per PPO
round. With --ppo-epochs 4, PPO makes four passes over those 128 transitions. At
30 Hz, --episode-seconds 20 permits at most 600 control steps per attempt; PPO
batches can cross episode boundaries. Evaluation with --episodes 10 and two
environments counts 20 completed attempts.
