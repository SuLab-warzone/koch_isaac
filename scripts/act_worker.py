"""Frozen ACT inference in the existing LeRobot environment; no robot connection."""
import argparse
import json
from pathlib import Path
import socket
import traceback

import numpy as np
from act_bridge import receive_packet, send_packet
from act_queue import ActionQueues, ObservationHistory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()
    sock = socket.socket(fileno=args.fd)
    try:
        import torch

        from lerobot.policies.factory import make_pre_post_processors

        torch.manual_seed(args.seed)
        path = str(args.policy_path.expanduser().resolve())
        kind = json.loads((Path(path)/"config.json").read_text())["type"]
        if kind == "act":
            from lerobot.policies.act.configuration_act import ACTConfig as Config
            from lerobot.policies.act.modeling_act import ACTPolicy as Policy
        elif kind == "diffusion":
            from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig as Config
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as Policy
        else:
            raise ValueError(f"Unsupported frozen base policy: {kind}; expected act or diffusion")
        config = Config.from_pretrained(path, local_files_only=True)
        config.device = args.device
        # All backbone weights come from this checkpoint; avoid an ImageNet download.
        config.pretrained_backbone_weights = None
        expected_inputs = {"observation.state": (6,), "observation.images.front": (3, 480, 640)}
        actual_inputs = {k: tuple(v.shape) for k,v in config.input_features.items()}
        if actual_inputs != expected_inputs or tuple(config.output_features["action"].shape) != (6,):
            raise ValueError(f"Unsupported checkpoint features: {actual_inputs}")
        policy = Policy.from_pretrained(path, config=config, local_files_only=True, strict=True)
        policy.eval()
        policy.requires_grad_(False)
        pre, post = make_pre_post_processors(
            config, pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": args.device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        if getattr(config, "temporal_ensemble_coeff", None) is not None:
            raise ValueError("Vector ACT currently supports queued chunks, not temporal ensembling")
        allowed = {"rename_observations_processor", "to_batch_processor", "device_processor",
                   "normalizer_processor", "unnormalizer_processor"}
        for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
            steps = json.loads((Path(path)/filename).read_text())["steps"]
            if any(step["registry_name"] not in allowed for step in steps):
                raise ValueError("Vector ACT requires stateless saved processors")
        queues = ActionQueues(args.num_envs, config.n_action_steps)
        history = ObservationHistory(args.num_envs, config.n_obs_steps)
        send_packet(sock, ready=True, policy_type=kind, n_obs_steps=config.n_obs_steps,
                    chunk_size=getattr(config, "chunk_size", getattr(config, "horizon", 0)),
                    n_action_steps=config.n_action_steps, policy_path=path)
        resets = 0
        while True:
            try:
                request = receive_packet(sock)
            except ConnectionError:
                break
            command = str(request["command"].item())
            ids = queues.validate_ids(request["env_ids"])
            if command == "reset":
                queues.reset(ids)
                history.reset(ids)
                # Saved processors were checked to be stateless; no other environment's queue changes.
                resets += 1
                send_packet(sock, reset_count=resets)
                continue
            if command != "action":
                raise ValueError(f"Unknown worker command: {command}")
            state, rgb = request["state"], request["rgb"]
            if state.shape != (len(ids),6) or not np.isfinite(state).all():
                raise ValueError("Expected finite state [batch,6]")
            if rgb.shape != (len(ids),480,640,3) or rgb.dtype != np.uint8:
                raise ValueError("Expected uint8 RGB [batch,480,640,3]")
            history.append(ids, state, rgb)
            hungry = queues.hungry(ids)
            with torch.inference_mode():
                if len(hungry):
                    states, images = history.batch(hungry)
                    batch_count, history_count = states.shape[:2]
                    batch = {
                        "observation.state": torch.from_numpy(states.reshape(-1,6).astype(np.float32)),
                        "observation.images.front": torch.from_numpy(images.reshape(-1,480,640,3).copy()).permute(0,3,1,2).float()/255.0,
                    }
                    processed = pre(batch)
                    if kind == "diffusion":
                        processed = {key: value.reshape(batch_count, history_count, *value.shape[1:])
                                     for key,value in processed.items() if key in config.input_features}
                    elif history_count != 1:
                        raise ValueError("ACT checkpoint requires unsupported multi-frame observations")
                    chunks = policy.predict_action_chunk(processed).detach().cpu().numpy()
                    queues.put(hungry, chunks)
                normalized, phase = queues.pop(ids)
                action = post(torch.as_tensor(normalized, device=args.device)).detach().cpu().numpy()
            if action.shape != (len(ids),6) or not np.isfinite(action).all():
                raise ValueError("ACT produced an invalid action")
            send_packet(sock, action=action, phase=phase)
    except BaseException as error:
        traceback.print_exc()
        try:
            send_packet(sock, error=f"ACT worker: {type(error).__name__}: {error}")
        except (OSError, ValueError):
            pass
        raise
    finally:
        sock.close()


if __name__ == "__main__":
    main()
