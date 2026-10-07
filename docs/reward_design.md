# Grasp shaping for a frozen ACT policy

This experiment assumes the robot can grasp the box and leaves ACT/diffusion,
joint conversion, dynamics, observations, residual limits and evaluation criteria
unchanged. PPO learns bounded corrections using more informative simulator
rewards. Simulation success still needs to be measured; real-world ACT data
does not establish the success rate of this residual in simulation.

## Weights and training stages

`--reward-stage grasp` is the new default. It focuses on reaching, closing and
lifting. Once grasping is reliable, use `--reward-stage pick-place` to enable
transport and reduce the incentive to hold the box until timeout.

| Term | Grasp weight | Pick-place weight | Signal |
|---|---:|---:|---|
| reach | 0.5 | 0.5 | `1 - tanh(TCP–box COM distance / 0.02)` |
| alignment | 1 | 1 | Local TCP error and vertical/edge orientation proxy |
| aperture | 0.5 | 0.5 | Open during approach, close as alignment improves |
| contact | 2 | 2 | Weaker opposing finger force, saturated at 0.08 N |
| lift | 8 | 2 | Box-bottom height / 0.04 m, clipped to [0,1], contact gated |
| transport | 0 | 4 | Distance to bin, gated by current grasp and prior stable lift |
| grasp_bonus | +5 once | +5 once | Opposing contact + bottom ≥25 mm for 0.2 s |
| placement_bonus | +20 once | +20 once | Stable released placement after a stable grasp |
| action_rate | −0.002 | −0.002 | Existing squared change in commanded joint angles |

The separate residual penalty remains `0.05 × sum(correction_radians²) × dt`.
Dense positive signals are [0,1]. Isaac multiplies dense weights by control dt;
milestone functions divide by dt so their weights are actual one-time bonuses.
The state is reset per environment and bonuses cannot repeat within an episode.
Reward milestones use fixed shaping settings, independently of CLI evaluation
thresholds (`--grasp-height`, `--grasp-force`, `--grasp-hold`).

## What changes in the objective

The old reach weight was 4; no term encouraged closure/contact, and lift only
became positive after a 25 mm center-height threshold. The new lift signal is
positive at a 5 mm lift. Height uses oriented box corners and table-top height,
not just the box root. No lift/transport reward is paid for an airborne box with
no grasp, a single-finger contact, or nonopposing finger forces.

Approach, alignment, aperture and contact rewards stop after the first stable
grasp. A later drop removes current lift and transport rewards; the one-time
grasp milestone stays latched. This prevents repeated bonus collection through
drop/regrasp cycles. The grasp stage still runs the usual 20 s task horizon and
placement/lost-box terminations; it does not introduce early termination on grasp.

Contact uses existing box-filtered sensors, not table/bin forces or actor inputs.
Normal resultants must be opposing (cosine < −0.25), both above 0.01 N, and the
box within 40 mm of the TCP. Contact shaping saturates with the weaker finger,
so stronger squeezing beyond the target gets no additional contact reward.
At valid dual contact the aperture objective stops asking for further closure.

Alignment is a pose proxy: Gaussian local errors with scales [12,8,4] mm, a
vertical approach-axis score, and a jaw-axis score aligned to either box edge.
It does not measure actual jaw-pad positions. Aperture is joint-angle shaping,
not physical jaw width; the provisional angles are 1 rad open and 0 rad closed,
with a 0.25 rad tolerance. These and other thresholds are centralized in
`GraspShapingSettings` in `koch_isaac/reward_shaping.py`; adjust them if measured
angles/TCP geometry differ. Confirm behavior with existing contact/joint plots.

## Run and compare

On the Ubuntu Isaac machine, run from the repo root with your usual base-policy
path, camera, calibration and vision flags. This example uses the existing cached
ACT baseline (no policy or checkpoint download):

```bash
./run.sh --mode residual-ppo --train --headless --num_envs 2 \
  --reward-stage grasp --total-timesteps 100000 \
  --checkpoint-out checkpoints/grasp_shaping_100k.zip

./run.sh --mode residual-ppo --headless --num_envs 2 --episodes 20 \
  --reward-stage grasp --checkpoint checkpoints/grasp_shaping_100k.zip --seed 123
```

Use `--policy-path` explicitly if your ACT differs from the cached default. All
existing actor/critic and frozen-policy contracts remain unchanged. To use the
optional planar vision cue, include `--vision color-plane` during both training
and evaluation and use a compatible vision residual checkpoint.

After grasp success improves, continue from that residual with a new output file:

```bash
./run.sh --mode residual-ppo --train --headless --num_envs 2 \
  --reward-stage pick-place --checkpoint checkpoints/grasp_shaping_100k.zip \
  --total-timesteps 100000 --checkpoint-out checkpoints/pick_place_shaping_200k.zip
```

Starting a fresh residual gives the cleanest comparison against the old reward.
Resuming an old compatible residual is allowed, but the critic must adapt to the
new returns. Checkpoints record reward stage, weights and shaping settings;
resuming with a changed objective prints a notice. Rewards are experiment metadata,
not part of the actor/base-policy compatibility contract. Evaluation uses the
selected CLI stage, not a stage automatically restored from the checkpoint.

Compare frozen ACT and residual success at identical seeds/scene settings, then
held-out seeds. Inspect sustained contact and lift rather than only episode
returns. Old and new returns, and returns across stages, are not comparable.
Weights are starting values, not validated optima.

## Validation

CPU tests exercise continuous lift, force saturation, closure gating, opposition,
one-time bonuses, timestep normalization, shared-step caching and selective resets.
The Isaac adapter samples physical state directly during reward computation;
it does not read `EvaluatedKochEnv`'s post-reward metrics. Stable placement uses
the termination manager's already-computed success flag before automatic reset.
Contact sensors are now enabled in the default task config, including hold mode.

Run unit tests with the simulator's Python (no Kit launch needed for the shaping
math test): `python -m unittest discover -s tests -p 'test_reward_shaping.py' -v`.
Use `scripts/check_vector_env.py --headless` on Ubuntu for actual physics/reset
integration. No simulator success claim follows from CPU tests alone.
