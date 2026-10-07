"""Exercise the real train() logging path without Isaac or a frozen vision model."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import residual_ppo
from policy_session import ACTOR_DIM, CRITIC_DIM


class FakeSession:
    """Two five-step episodes with returns 5 and 15, not per-step rewards 1 and 3."""
    def __init__(self, env, args, calibration, output):
        self.num_envs = 2
        self.args = args
        self.contract = {"test": "training logs"}
        self.reward_metadata = {"version": "test", "stage": "grasp"}
        self.last_rgb = None
        self.steps = 0

    def reset(self):
        self.steps = 0
        return np.zeros((2, ACTOR_DIM + CRITIC_DIM), dtype=np.float32)

    def step(self, actions):
        self.steps += 1
        done = self.steps % 5 == 0
        infos = [{"env_id": i, "episode": {"r": r, "l": 5}} if done else {}
                 for i, r in enumerate((5., 15.))]
        return (np.zeros((2, ACTOR_DIM + CRITIC_DIM), dtype=np.float32),
                np.array([1., 3.], dtype=np.float32), np.array([done, done]), infos)

    def summary(self, status):
        return {"status": status}

    def close(self):
        pass


class TrainingLoggingTests(unittest.TestCase):
    def test_completed_episode_mean_and_frequent_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(
                checkpoint_out=root / "trial.zip", checkpoint=None,
                save_output=False, num_envs=2, output_dir=None, snapshot=None, video=False,
                ppo_steps=4, ppo_batch_size=8, ppo_epochs=1, learning_rate=3e-4,
                policy_device="cpu", seed=42, total_timesteps=24, train_log_every=2,
            )
            env = SimpleNamespace(num_envs=2, sim=SimpleNamespace(visualizers=[]))
            with patch.object(residual_ppo, "ROOT", root), patch.object(residual_ppo, "PolicySession", FakeSession):
                residual_ppo.train(env, args, {})
            runs = list((root / "runs").iterdir())
            self.assertEqual(len(runs), 1)
            self.assertTrue((root / "trial.zip").is_file())
            saved = residual_ppo.load_residual(root / "trial.zip", {"test": "training logs"})
            self.assertEqual(saved.reward_metadata, {"version": "test", "stage": "grasp"})
            events = EventAccumulator(str(runs[0])).Reload()
            rewards = events.Scalars("rollout/ep_rew_mean")
            self.assertTrue(rewards)
            self.assertTrue(all(abs(point.value - 10.) < 1e-6 for point in rewards))
            # End at vector step 5; callback sees the completed SB3 buffer at step 6.
            # 6 vector steps * 2 envs = 12 transitions, before the next rollout ends at 16.
            self.assertIn(12, [point.step for point in rewards])
            self.assertTrue(all(point.step >= 10 for point in rewards))
            lengths = events.Scalars("rollout/ep_len_mean")
            self.assertTrue(all(point.value == 5. for point in lengths))
            # The explicit final dump retains loss from the final optimization round.
            self.assertEqual(events.Scalars("train/value_loss")[-1].step, 24)


if __name__ == "__main__":
    unittest.main()
