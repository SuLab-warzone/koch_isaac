# Public Koch follower asset

Source: https://github.com/Maverobot/koch_ros
Pinned commit: `5a6fd8a88b24f7d112399fc3190bfa205dd405b0`
Original: `koch_description/urdf/koch_follower.xacro` and `koch_description/meshes/`.
MIT license and upstream NOTICE are included here. Source credits the
`feature/URDF` branch of https://github.com/tc-huang/low_cost_robot.

The Xacro macro's links and joints were extracted to a standalone URDF,
`${pi}` was expanded, and meshes now use relative paths. ROS hardware and
transmission tags were omitted. No ROS or xacro installation is needed.
Visual color was changed to charcoal. Geometry, mass and inertia are from
the public asset, not measured on your arm.

Upstream uses placeholder limits of ±pi, effort 1000 and velocity 10 for
every revolute joint. This project replaces them with provisional ±2.7 rad
arm limits, [0, 1.5] rad gripper limits, 2.5 N m arm / 0.6 N m gripper
limits and 2 rad/s velocities. These are simulation settings, NOT validated
Dynamixel ratings or hardware operating limits. The arm motor groups should
be calibrated individually before transfer. Self-collisions are initially
disabled; enable only after inspecting adjacent-link collision geometry.

Isaac's importer generates a local USD cache in `usd/` on first launch.
Delete that cache after mesh-only edits, since the converter hashes the URDF
but does not automatically detect changed mesh contents.
