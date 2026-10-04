"""Capture terminal observations and grasp/place metrics before automatic reset."""
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import matrix_from_quat
from . import settings as s
from .mdp.observations import box_position, tcp_position
from .mdp.terminations import placement_candidate
from .metrics import EpisodeMetrics


class EvaluatedKochEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, *, grasp_height=0.025, grasp_hold=0.2, grasp_force=0.01, **kwargs):
        self._evaluating_step = False
        self.metrics = None
        super().__init__(cfg, **kwargs)
        self.grasp_height = grasp_height
        self.grasp_force = grasp_force
        self.metrics = EpisodeMetrics(self.num_envs, self.device, self.step_dt, grasp_hold, 0.5)
        self._metric_step = -1
        self.terminal_records = {}
        self.terminal_policy_obs = {}

    def _update_metrics(self):
        if self._metric_step == self.common_step_counter:
            return
        self._metric_step = self.common_step_counter
        box = self.scene["box"]
        extent = torch.matmul(matrix_from_quat(box.data.root_quat_w.torch).abs(),
                              torch.tensor(s.BOX_SIZE, device=self.device) / 2)
        lift = box_position(self)[:, 2] - extent[:, 2]
        contacts = []
        for name in ("static_finger_contact", "moving_finger_contact"):
            forces = self.scene[name].data.force_matrix_w
            if forces is None:
                raise RuntimeError("Grasp evaluation requires filtered finger/box contact forces")
            force = forces.torch if hasattr(forces, "torch") else forces
            contacts.append(torch.linalg.vector_norm(force, dim=-1).reshape(self.num_envs, -1).amax(-1) > self.grasp_force)
        near = torch.linalg.vector_norm(tcp_position(self)-box_position(self), dim=-1) < 0.08
        grasp = contacts[0] & contacts[1] & (lift >= self.grasp_height) & near
        self.metrics.update(grasp, placement_candidate(self), lift)

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
                        "success": bool(self.termination_manager.get_term("success")[index]),
                        "box_lost": bool(self.termination_manager.get_term("box_lost")[index]),
                        "timeout": bool(self.termination_manager.get_term("time_out")[index]),
                    }
            self.metrics.reset(env_ids)
        super()._reset_idx(env_ids)
