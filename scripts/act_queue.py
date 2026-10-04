"""Independent ACT chunks for asynchronous vector-environment resets."""
import numpy as np


class ActionQueues:
    def __init__(self, num_envs, action_steps):
        if num_envs < 1 or action_steps < 1: raise ValueError("Invalid queue dimensions")
        self.num_envs, self.action_steps = num_envs, action_steps
        self.chunks = [None] * num_envs
        self.cursors = np.zeros(num_envs, dtype=np.int64)

    def validate_ids(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= self.num_envs)):
            raise ValueError("Expected unique valid environment IDs")
        return ids

    def hungry(self, ids):
        return np.array([i for i in self.validate_ids(ids) if self.chunks[i] is None or self.cursors[i] >= self.action_steps], dtype=np.int64)

    def put(self, ids, chunks):
        ids = self.validate_ids(ids)
        chunks = np.asarray(chunks)
        if chunks.ndim != 3 or chunks.shape[0] != len(ids) or chunks.shape[1] < self.action_steps or chunks.shape[2] != 6:
            raise ValueError("Expected ACT chunks [batch, >=action_steps, 6]")
        if not np.isfinite(chunks).all(): raise ValueError("Nonfinite ACT chunk")
        for index, chunk in zip(ids, chunks):
            self.chunks[index] = chunk[:self.action_steps].copy()
            self.cursors[index] = 0

    def pop(self, ids):
        ids = self.validate_ids(ids)
        if len(self.hungry(ids)): raise RuntimeError("ACT queue empty")
        phase = self.cursors[ids].astype(np.float32) / self.action_steps
        actions = np.stack([self.chunks[i][self.cursors[i]] for i in ids])
        self.cursors[ids] += 1
        return actions, phase

    def reset(self, ids=None):
        ids = np.arange(self.num_envs) if ids is None else self.validate_ids(ids)
        for index in ids:
            self.chunks[index] = None
            self.cursors[index] = 0


class ObservationHistory:
    """Raw per-environment history; pad a fresh episode with its first observation."""
    def __init__(self, num_envs, length):
        from collections import deque
        self.length = length
        self.states = [deque(maxlen=length) for _ in range(num_envs)]
        self.images = [deque(maxlen=length) for _ in range(num_envs)]

    def append(self, ids, states, images):
        for i, state, image in zip(ids, states, images):
            state, image = state.copy(), image.copy()
            if not self.states[i]:
                for _ in range(self.length):
                    self.states[i].append(state); self.images[i].append(image)
            else:
                self.states[i].append(state); self.images[i].append(image)

    def batch(self, ids):
        return (np.stack([np.stack(self.states[i]) for i in ids]),
                np.stack([np.stack(self.images[i]) for i in ids]))

    def reset(self, ids):
        for i in ids: self.states[i].clear(); self.images[i].clear()
