# Vision-assisted residual PPO

This implements **Path 1**: frozen ACT/diffusion + a PPO residual conditioned on an
RGB estimate of the pink box. It does not implement privileged-teacher distillation.

## Data flow and responsibilities

```text
RGB + measured joints -> frozen ACT worker -> base target in simulation radians
RGB -> BoxEstimator -> estimated box position + detection validity
robot forward kinematics -> TCP position
estimate - TCP, validity -> residual actor (alongside the original 25 inputs)
base target + scaled residual -> joint limits -> Isaac environment
reward -> PPO optimizer; frozen policy weights are never updated
```

| File | Responsibility |
| --- | --- |
| `koch_isaac/vision/base.py` | Estimator interface and explicit valid/invalid result |
| `koch_isaac/vision/color_plane.py` | RGB -> HSV mask -> connected component -> planar box estimate |
| `koch_isaac/vision/geometry.py` | Pixel-to-plane homography; camera-derived or measured calibration |
| `koch_isaac/vision/features.py` | Relative position scaling, detection validity, per-episode grasp-attempt gate |
| `koch_isaac/vision/isaac_adapter.py` | Isaac camera calibration and robot FK, independent environment resets |
| `scripts/vision_cli.py` | Standalone calibration and image inspection, without Isaac |
| `scripts/residual_observations.py` | Documented actor/critic input layout |
| `scripts/policy_session.py` | Shared control loop, action conversion, ACT + residual combination |
| `scripts/residual_ppo.py` | SB3 policy/critic, training and checkpoint compatibility |
| `config/vision_color_plane.json` | Colour thresholds, workspace, projection height and gating parameters |

No hidden box pose or box contact sensor is used by the vision actor. The critic
and reward still use simulator state, as before. The adapter reads only camera
calibration, RGB, robot joint state and robot FK. Simulator object positions in
`check_vision_scene.py` are used solely to arrange and score the integration test.

The legacy actor has 25 inputs. The new actor has 29:
joint positions (6), velocities (6), base action (6), previous residual (6),
chunk phase (1), estimated box-minus-TCP / 0.2 m (3), estimate validity (1).
The critic has its separate 31-input suffix. Invalid vision features are all zero.

ACT prediction length and execution horizon remain 100 actions; commands and
residual inference remain 30 Hz, physics 240 Hz. This change isolates the effect
of adding perception; it does not change rewards or the frozen policy horizon.

## Start a new experiment

Commands below run from the project root, using the existing Isaac/LeRobot setup.
OpenCV is required in the Isaac Python environment (tested with OpenCV 4.13.0).

```bash
# Inspect the CV signal with the frozen policy, without training.
./run.sh --mode policy --vision color-plane --episodes 1 --no-save-output

# NEW residual training: do not supply an old 25-input residual checkpoint.
./run.sh --mode residual-ppo --train --vision color-plane \
  --num_envs 16 --headless --no-save-output \
  --total-timesteps 1000000 \
  --checkpoint-out checkpoints/residual_color_plane_v1.zip

# Evaluate with the same vision settings.
./run.sh --mode residual-ppo --vision color-plane \
  --checkpoint checkpoints/residual_color_plane_v1.zip \
  --episodes 10 --no-save-output
```

The million-transition command is an initial experiment, not a promised
convergence budget. Use matched initial seeds/positions to compare frozen ACT
and residual success. The brief implementation smoke test is not a trained model.

`--vision none` is the default and retains the old observation layout/checkpoint
contract. Old checkpoints remain evaluable with that option. A 25-input checkpoint
cannot be resumed into the 29-input experiment. Vision checkpoints also record
feature scale, thresholds, box size and TCP offset, rejecting mismatched settings.

At `--log-every` intervals, the terminal prints `Vision:` with env0's estimate,
detection fraction, actor-valid fraction and disabled fraction. `valid` in the
estimator result means a usable pink component was detected under the plane
assumption; `env0_actor_valid` additionally applies the grasp-attempt gate.

## No training labels required

This CV baseline uses OpenCV colour thresholds, not a trained neural network.
You need:
- HSV bounds that select the pink object under your lighting.
- Camera/table calibration to convert pixels into metres in the robot frame.
- Measurements of the box height and the workspace bounds.

Simulation calibration comes automatically from the configured camera and table.
Do not reuse it on a real camera. OpenCV hue is 0..179, saturation/value 0..255.
Inputs to the estimator are RGB; `vision_cli.py` converts OpenCV's BGR image load.

## Standalone inspection and real-camera calibration

First, export a simulated image/calibration pair for checking the pipeline:
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/niel/miniconda3/envs/isaaclab/bin/python scripts/check_vision_scene.py \
  --headless --diagnostic-dir /tmp/koch_vision

/home/niel/miniconda3/envs/isaaclab/bin/python scripts/vision_cli.py inspect \
  --image /tmp/koch_vision/front.png \
  --calibration /tmp/koch_vision/calibration.json \
  --overlay /tmp/koch_vision/overlay.png
```

For a real fixed camera, undistort images first if lens distortion is material.
Measure at least four non-collinear point correspondences spanning the grasp
workspace; six or more plus held-out validation points are preferable. Coordinates
must use the same robot reference frame as FK. Reference points must lie at the
projection height, not automatically at the bare tabletop height. For this 8 mm
box, the silhouette-centre approximation currently uses a plane 4 mm above the table.

Create a **local** JSON with these fields (replace all pairs with your measurements):
```text
pixels_uv:   [[u1,v1], [u2,v2], ...]     # undistorted pixel coordinates
plane_xy_m:  [[x1,y1], [x2,y2], ...]     # corresponding measured robot-frame metres
image_size: [640,480]                   # width, height; must match actual input
plane_z_m:  0.004                       # calibration plane height in robot frame
center_z_m: 0.004                       # resting box COM height in robot frame
```

Then run:
```bash
python scripts/vision_cli.py calibrate \
  --points config/local/camera_points.json \
  --output config/local/camera_plane.json

python scripts/vision_cli.py inspect \
  --image /path/to/undistorted_frame.png \
  --calibration config/local/camera_plane.json \
  --config config/vision_color_plane.json
```

The fitted error printed by `calibrate` is not an accuracy guarantee: check new,
held-out physical positions. There is no real-robot control code in this CV tool.

## Important limits and replacement boundary

A single table-plane projection cannot measure lift height. The actor receives
an **approach cue**, not a continuously valid 3D box tracker. When the final gripper
command OR measured angle is <= 0.65 rad within 8 cm of the estimate, the cue is
disabled for the rest of that environment's episode. This conservative heuristic
does not prove a grasp, and can disable the cue early. It also disables subsequent
vision-guided retries. Opening the gripper does not silently reactivate a possibly
lifted estimate. An independent environment reset clears only that environment's gate.

While the cue is enabled, disappearance or similar-size competing pink components
produce zero features plus validity=0, never a stale location. Partial occlusion,
reflections and lighting changes can still bias a remaining blob. The silhouette
centre is approximate; `projection_height_fraction=0.5` reduced a measured 6–8 mm
top-surface projection bias to about 0.4–1.4 mm in the six local rendered test cases.
This is not a guarantee for real images or arbitrary camera viewpoints.

To replace CV, implement `BoxEstimator.estimate(rgb) -> BoxEstimate` and wire the
new backend in the adapter/factory/CLI. Keep its coordinates and feature scaling
explicit. A depth-based estimator can use a different feature/gating adapter that
keeps 3D estimates active after lifting. Version the checkpoint contract and train
a new residual when changing observation semantics.

## Validation

- `python -m unittest discover -s tests -v`: geometry, missing/ambiguous
  detections, gating/reset isolation, actor/critic separation, legacy/new SB3 save/load.
- `scripts/check_vision_scene.py --headless`: three poses/yaws in two environments;
  asserts detection and <5 mm XY error.
- Short actual-ACT PPO run: 2 environments, 256 transitions, 4 completed timeout
  episodes; verifies optimizer, reset handling, finite observations and checkpoint
  saving. No pickup success is claimed from this smoke test.

## References

- [OpenCV HSV thresholding](https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html)
- [OpenCV plane homographies](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)
