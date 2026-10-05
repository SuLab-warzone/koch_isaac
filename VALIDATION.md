# Validation — 2026-10-01 (local time)

Tested in the existing `isaaclab` conda environment on the RTX 5090.

| Check | Result |
|---|---|
| Public URDF conversion and six-joint articulation import | PASS |
| Four parallel environments, 90 initial control steps with finite observations/rewards | PASS |
| Box settles at 12.5 mm center height on table | PASS |
| Shoulder-pan position command produces joint motion | PASS |
| Indexed reset leaves other environments unchanged and clears box velocity | PASS |
| Success rejects hovering, rim overlap and moving boxes | PASS |
| Bin floor supports box; sustained placement triggers termination/reset | PASS |
| Front RGB camera, 90 steps, 640 × 480 saved image | PASS |
| Visual inspection: arm, box and open yellow container visible | PASS |
| Python compilation of project modules and scripts | PASS |

Logs: `outputs/smoke.log`, `outputs/camera.log`.
Preview: `outputs/front_camera.png`.

These checks do not establish learned pick-and-place success, real-world
calibration, gripper contact fidelity, or ACT policy compatibility. The policy
configuration was inspected and recorded; its weights were not loaded or
trained. Tests use a known box placement to validate the success metric.

The simulator emits nonfatal warnings about its extension metadata, rendering
integration and TGS solver settings. Native startup conflicts were resolved by
starting Kit before task imports; no global simulator files were edited.

## Follow-up: GUI Fabric workaround

- User log: 1,044 PhysX Fabric CUDA shim errors in a GUI session.
- Fresh GUI reproduction with original Fabric enabled: 180 joint-motion steps
  completed, no CUDA shim errors; the original trigger remains unidentified.
- GUI using USD synchronization: 240 initial steps plus joint/contact/reset/
  placement checks passed on `cuda:0`, with no CUDA shim errors.
- Default launcher now uses USD sync for GUI and Fabric for headless;
  `--fabric on|off|auto` allows an explicit choice.
- Logs: `outputs/gui_repro.log`, `outputs/gui_usd_check.log`.
- GUI launched from an explicitly activated Miniforge `lerobot061` shell:
  120 joint-motion steps passed with USD sync on `cuda:0`. The launcher used
  the Miniconda Isaac Lab interpreter as intended. Log:
  `outputs/gui_from_lerobot.log`.

## Residual PPO and vector rollout — 2026-10-03 local time

Implemented on independent Git branch residual-ppo; main and act-baseline preserve
the previous single-environment ACT implementation. Tested with the cached home_v2
ACT checkpoint, Isaac Lab 3.0.0, Sim 6.0.1, stable-baselines3 2.9.0, and RTX 5090.

| Check | Result |
|---|---|
| 17 unit tests: conversion/protocol, independent queues/history, metric hold/reset, output suppression, actor/critic isolation, contract rejection | PASS |
| Two-environment frozen ACT evaluation, one 60-step episode each | PASS; both timeouts, no successes |
| No multi-environment output directory even when --output-dir supplied | PASS |
| Filtered contact sensors initialize on both nested finger bodies | PASS |
| Forced timeout in env 0 leaves env 1 running; critic receives pre-reset terminal state | PASS |
| Deliberately placing box in bin produces terminal placement success without a grasp success | PASS |
| Terminal metrics clear for the next episode | PASS |
| Residual PPO: 256 transitions, two environments, four rollout batches, 16 optimization epochs | PASS |
| Learned checkpoint parameters finite; initially zero action-output weights changed (max absolute 0.00446169) | PASS |
| Reload checkpoint and evaluate two environments | PASS; both timeouts, no successes |
| Single-environment output: summary, JSONL, 640x480 images, 60-frame video at 30 fps | PASS |
| Saved first camera frame visually inspected | PASS; robot, box, container visible |

Saved smoke checkpoint: checkpoints/validation_256.zip (ignored by Git). It is
only a pipeline validation checkpoint, not a successfully trained pick-and-place
policy. Training used --episode-seconds 2; all four completed training episodes
timed out with no grasp or placement. No claim of improved robustness is made.
Use normal 20-second episodes for longer training and comparable evaluations.

The grasp metric's hold/reset logic is unit-tested; a naturally successful
simulated grasp was not observed in these short runs. Its force, height and TCP
thresholds still need validation on representative grasp trials. Placement was
verified by explicitly positioning the box in the bin, not by a learned pick.
No actual diffusion checkpoint was provided; its config-driven loading and
per-environment history support remain unverified end-to-end.

Temporary single-environment artifacts: /tmp/koch-residual-5BBycU/saved_rollout.
No rollout/log/video files are produced by the multi-environment application;
Isaac Kit still writes its own external runtime diagnostic logs.

To rerun unit checks:

```bash
/home/niel/miniconda3/envs/isaaclab/bin/python -B -m unittest discover -s tests -v
```

To rerun the simulator terminal-state and placement check:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/niel/miniconda3/envs/isaaclab/bin/python -B scripts/check_vector_env.py --headless
```


## Colour-plane residual observation path

- Added optional `--vision color-plane`, retaining `--vision none` for legacy checkpoints.
- 30 unit/regression tests pass, including projection geometry, missing/ambiguous RGB detections,
  per-environment cue gating/reset, actor/critic isolation, and old/new checkpoint serialization.
- Rendered Isaac check: three positions/yaws in two environment origins; all six detections
  succeeded. Measured XY errors were approximately 0.40–1.34 mm after setting silhouette
  projection to the box mid-height. This checks the current simulated scene only.
- Actual frozen ACT + residual PPO smoke test: 2 environments, 256 transitions, 4 completed
  two-second timeout episodes. Checkpoint saved successfully; detection rate 100%.
- Reloaded that checkpoint for 2 four-second evaluation episodes (240 transitions), crossing
  a 100-action ACT queue boundary. Evaluation completed; detection rate 100%.
- No grasp or placement improvement is claimed: these short runs validate plumbing only.
- Table-plane features are intentionally retired after a nearby gripper-close attempt;
  a validity input signals this to the actor. See `docs/vision.md` for limitations and commands.
- Model weights, real motor calibration, camera calibration under `config/local/`, recordings,
  logs, and test artifacts are not included in the public source tree.
