"""Local ACT worker bridge; numpy payloads over a private inherited socket."""
import io
import os
from pathlib import Path
import socket
import struct
import subprocess

import numpy as np

MAX_PACKET = 16 * 1024 * 1024


def send_packet(sock, **arrays):
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    payload = stream.getvalue()
    if len(payload) > MAX_PACKET:
        raise ValueError("ACT packet too large")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def receive_packet(sock):
    def read_exact(size):
        result = bytearray()
        while len(result) < size:
            part = sock.recv(size - len(result))
            if not part:
                raise ConnectionError("ACT worker connection closed; inspect its console error")
            result.extend(part)
        return bytes(result)
    size, = struct.unpack("!I", read_exact(4))
    if not 0 < size <= MAX_PACKET:
        raise ValueError("Invalid ACT packet length")
    with np.load(io.BytesIO(read_exact(size)), allow_pickle=False) as data:
        result = {name: data[name] for name in data.files}
    if "error" in result:
        raise RuntimeError(str(result["error"].item()))
    return result


class ACTWorker:
    def __init__(self, python, policy_path, device="cuda", timeout=120.0, seed=42, num_envs=1, batch_size=4):
        if not 1 <= batch_size <= 8: raise ValueError("ACT batch size must be between 1 and 8")
        self.num_envs, self.batch_size = num_envs, batch_size
        self.phase = np.zeros(num_envs, dtype=np.float32)
        self.process = None
        self.sock, child = socket.socketpair()
        self.sock.settimeout(timeout)
        environment = os.environ.copy()
        # Do not inject Kit's Python packages or libraries into the LeRobot process.
        for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
            environment.pop(key, None)
        environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        try:
            self.process = subprocess.Popen(
                [str(python), "-B", str(Path(__file__).with_name("act_worker.py")),
                 "--fd", str(child.fileno()), "--policy-path", str(policy_path),
                 "--device", device, "--seed", str(seed), "--num-envs", str(num_envs)],
                pass_fds=(child.fileno(),), env=environment,
            )
            child.close()
            self.info = receive_packet(self.sock)
        except BaseException:
            child.close()
            self.close()
            raise

    def reset(self, env_ids=None):
        ids = np.arange(self.num_envs) if env_ids is None else np.asarray(env_ids, dtype=np.int64)
        send_packet(self.sock, command="reset", env_ids=ids)
        self.phase[ids] = 0
        return receive_packet(self.sock)

    def action(self, state, rgb):
        state = np.asarray(state, dtype=np.float32)
        single = state.ndim == 1
        if single: state, rgb = state[None], rgb[None]
        if state.shape != (self.num_envs, 6) or rgb.shape != (self.num_envs, 480, 640, 3):
            raise ValueError("State/image batch does not match environment count")
        actions = np.empty((self.num_envs, 6), dtype=np.float32)
        for start in range(0, self.num_envs, self.batch_size):
            ids = np.arange(start, min(self.num_envs, start+self.batch_size))
            send_packet(self.sock, command="action", env_ids=ids, state=state[ids], rgb=rgb[ids])
            result = receive_packet(self.sock)
            actions[ids] = result["action"]
            self.phase[ids] = result["phase"]
        return actions[0] if single else actions

    def close(self):
        self.sock.close()
        if self.process is not None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
