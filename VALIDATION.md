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
