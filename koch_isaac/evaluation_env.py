"""Capture terminal observations and grasp/place metrics before automatic reset."""
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import matrix_from_quat
from . import settings as s
from .mdp.observations import box_position, tcp_position
from .mdp.terminations import placement_candidate
from .metrics import EpisodeMetrics, ContactForceMetrics


class EvaluatedKochEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, *, grasp_height=0.025, grasp_hold=0.2, grasp_force=0.01, **kwargs):
        self._evaluating_step = False
        self.metrics = None
        self.contact_metrics = None
        self.rerun_logger = None  # Optional live sink attached by run_scene after construction.
        super().__init__(cfg, **kwargs)
        self.grasp_height = grasp_height
        self.grasp_force = grasp_force
        self.metrics = EpisodeMetrics(self.num_envs, self.device, self.step_dt, grasp_hold, 0.5)
        # Use scene values rather than a hard-coded box weight. These are configured
        # properties; if mass/material randomization is added, read runtime values here.
        material = cfg.scene.box.spawn.physics_material
        self.contact_metrics = ContactForceMetrics(
            self.num_envs, self.device, self.step_dt, grasp_force,
            cfg.scene.box.spawn.mass_props.mass,
            sum(component ** 2 for component in cfg.sim.gravity) ** 0.5,
            material.static_friction, material.dynamic_friction,
        )
        self._metric_step = -1
        self.terminal_records = {}
        self.terminal_policy_obs = {}

    def _update_metrics(self):
        # Auto-reset calls this inside step(); do not count that same control step
        # again after step() returns, or sample fresh reset poses as terminal data.
        if self._metric_step == self.common_step_counter:
            return
        self._metric_step = self.common_step_counter
        box = self.scene["box"]
        extent = torch.matmul(matrix_from_quat(box.data.root_quat_w.torch).abs(),
                              torch.tensor(s.BOX_SIZE, device=self.device) / 2)
        lift = box_position(self)[:, 2] - extent[:, 2]
        contacts = []
        normal_forces = []
        for name in ("static_finger_contact", "moving_finger_contact"):
            forces = self.scene[name].data.force_matrix_w
            if forces is None:
                raise RuntimeError("Grasp evaluation requires filtered finger/box contact forces")
            force = forces.torch if hasattr(forces, "torch") else forces
            # force_matrix_w: [env, sensor body, filtered box body, xyz], in N.
            # Each sensor watches one entire finger body against this env's box.
            # It excludes table/bin contacts, and does not include tangential friction.
            pairs = force.reshape(self.num_envs, -1, 3)
            if pairs.shape[1] != 1:
                raise RuntimeError("Expected exactly one finger/box pair per contact sensor")
            normal_forces.append(pairs[:, 0])
            contacts.append(torch.linalg.vector_norm(pairs[:, 0], dim=-1) > self.grasp_force)
        self.contact_metrics.update(torch.stack(normal_forces, dim=1))
        near = torch.linalg.vector_norm(tcp_position(self)-box_position(self), dim=-1) < 0.08
        grasp = contacts[0] & contacts[1] & (lift >= self.grasp_height) & near
        self.metrics.update(grasp, placement_candidate(self), lift)
        if self.rerun_logger is not None:
            # Runs once per control step, including terminal samples BEFORE reset.
            # Logging after env.step() instead would mix terminal forces with reset poses.
            self.rerun_logger.log(self, lift)

    def step(self, action):
        self.terminal_records = {}
        self.terminal_policy_obs = {}
        self._evaluating_step = True
        try:
            result = super().step(action)
            self._update_metrics()
            return result
        finally:
            self._evaluating_step = False

    def _reset_idx(self, env_ids):
        if self.metrics is not None:
            if self._evaluating_step:
                self._update_metrics()
                terminal = self.observation_manager.compute_group("policy")
                for index in env_ids.tolist():
                    self.terminal_policy_obs[index] = terminal[index].detach().cpu().numpy().copy()
                    self.terminal_records[index] = {
                        **self.metrics.outcome(index),
                        "contact_forces": self.contact_metrics.outcome(index),
                        "success": bool(self.termination_manager.get_term("success")[index]),
                        "box_lost": bool(self.termination_manager.get_term("box_lost")[index]),
                        "timeout": bool(self.termination_manager.get_term("time_out")[index]),
                    }
            # Capture forces BEFORE clearing histories and resetting the simulator.
            # PolicySession forwards this terminal snapshot to Evaluation episode.
            self.metrics.reset(env_ids)
            self.contact_metrics.reset(env_ids)
        super()._reset_idx(env_ids)
