"""Isaac camera/FK adapter. Only this vision module depends on Isaac APIs."""
import numpy as np

from .color_plane import ColorPlaneEstimator
from .geometry import PlaneCalibration
from .features import PlanarBoxFeatures


class IsaacBoxVision:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.features = PlanarBoxFeatures(env.num_envs, config)
        self.estimators = None
        self.last_estimates = []
        self.last_features = np.zeros((env.num_envs, 4), dtype=np.float32)
        self.samples = self.detected = self.usable = 0

    def _calibrate(self, rgb):
        """Read fixed camera extrinsics/intrinsics AFTER the first rendered reset.

        Subtract each environment origin: cloned cameras share local calibration,
        even though their world positions differ. Camera optical axes are ROS:
        +X right, +Y down, +Z forward. Isaac math handles its XYZW quaternion order.
        """
        from isaaclab.utils.math import matrix_from_quat
        from .. import settings as s
        data = self.env.scene["front_camera"].data
        intrinsic = data.intrinsic_matrices.torch.detach().cpu().numpy()
        rotation = matrix_from_quat(data.quat_w_ros.torch).detach().cpu().numpy()
        position = (data.pos_w.torch - self.env.scene.env_origins).detach().cpu().numpy()
        # The mask includes top AND side faces. Its centroid is approximated at
        # mid-height rather than at the top face (which biases XY toward camera).
        # The fraction is configurable; verify it against held-out measured points.
        table_z = s.TABLE_POS[2] + s.TABLE_SIZE[2] / 2
        self.estimators = [
            ColorPlaneEstimator(PlaneCalibration.from_camera(
                k, r, p, (rgb.shape[2], rgb.shape[1]), table_z + self.config["projection_height_fraction"] * s.BOX_SIZE[2],
                table_z + s.BOX_SIZE[2] / 2), self.config)
            for k, r, p in zip(intrinsic, rotation, position)
        ]

    def reset(self, ids=None):
        self.features.reset(ids)

    def observe(self, rgb, measured_gripper):
        from ..mdp.observations import tcp_position
        if self.estimators is None:
            self._calibrate(rgb)
        self.last_estimates = [estimator.estimate(frame)
                               for estimator, frame in zip(self.estimators, rgb)]
        # TCP is robot forward kinematics; no box/contact/critic state is read here.
        tcp = tcp_position(self.env).detach().cpu().numpy()
        self.last_features = self.features.update(self.last_estimates, tcp, measured_gripper)
        self.samples += self.env.num_envs
        self.detected += sum(estimate.valid for estimate in self.last_estimates)
        self.usable += int(self.last_features[:, 3].sum())
        return self.last_features

    def note_action(self, joint_targets):
        self.features.note_gripper(joint_targets[:, -1])

    def report(self):
        first = self.last_estimates[0].as_dict() if self.last_estimates else {}
        return {"env0": first, "env0_actor_valid": bool(self.last_features[0, 3]),
                "detected_fraction": float(np.mean([e.valid for e in self.last_estimates])),
                "actor_valid_fraction": float(self.last_features[:, 3].mean()),
                "grasp_gate_disabled_fraction": float(self.features.disabled.mean())}

    def summary(self):
        return {"estimator": "color-plane",
                "detection_rate": self.detected / self.samples if self.samples else None,
                "actor_cue_rate": self.usable / self.samples if self.samples else None,
                "limitation": "Planar approach cue; disabled after a nearby gripper-close attempt"}
