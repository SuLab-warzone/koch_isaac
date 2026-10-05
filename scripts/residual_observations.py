"""Documented actor/critic layouts, without simulator or policy-worker imports."""
import numpy as np

# Keep the legacy constants/imports working for old checkpoints and tests.
ACTOR_DIM = 25
CRITIC_DIM = 31
VISION_DIM = 4  # relative box XYZ / position_scale_m, followed by valid (0 or 1)


def make_residual_observation(q, velocity, base_action, previous_residual, phase,
                              privileged, vision_features=None):
    """Actor uses deployable inputs; critic receives a separate privileged suffix.

    Legacy actor indices: q[0:6], dq[6:12], ACT[12:18], previous correction[18:24],
    chunk phase[24]. Vision actor appends relative box[25:28], validity[28].
    Invalid estimates have ALL four appended values zero (no stale coordinates).
    """
    actor = np.concatenate((q / np.pi, velocity / 2.0, base_action / np.pi,
                            previous_residual, phase[:, None]), axis=-1)
    if actor.shape[-1] != ACTOR_DIM or privileged.shape[-1] != CRITIC_DIM:
        raise ValueError("Unexpected residual observation layout")
    if vision_features is not None:
        if vision_features.shape != (len(actor), VISION_DIM):
            raise ValueError("Expected vision features [num_envs,4]")
        actor = np.concatenate((actor, vision_features), axis=-1)
    result = np.concatenate((actor, privileged), axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite residual observation")
    return result


def terminal_residual_observation(privileged, actor_dim=ACTOR_DIM):
    # Timeout bootstrapping uses the critic only; NEVER substitute post-reset state.
    return np.concatenate((np.zeros(actor_dim, dtype=np.float32), privileged)).astype(np.float32)
