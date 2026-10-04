"""Frozen ACT inference in the existing LeRobot environment; no robot connection."""
import argparse
from pathlib import Path
import socket
import traceback

import numpy as np
from act_bridge import receive_packet, send_packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sock = socket.socket(fileno=args.fd)
    try:
        import torch
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        torch.manual_seed(args.seed)
        path = str(args.policy_path.expanduser().resolve())
        config = ACTConfig.from_pretrained(path, local_files_only=True)
        config.device = args.device
        # All backbone weights come from this checkpoint; avoid an ImageNet download.
        config.pretrained_backbone_weights = None
        expected_inputs = {"observation.state": (6,), "observation.images.front": (3, 480, 640)}
        actual_inputs = {k: tuple(v.shape) for k,v in config.input_features.items()}
        if actual_inputs != expected_inputs or tuple(config.output_features["action"].shape) != (6,):
            raise ValueError(f"Unsupported checkpoint features: {actual_inputs}")
        policy = ACTPolicy.from_pretrained(path, config=config, local_files_only=True, strict=True)
        policy.eval()
        policy.requires_grad_(False)
        pre, post = make_pre_post_processors(
            config, pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": args.device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        send_packet(sock, ready=True, chunk_size=config.chunk_size,
                    n_action_steps=config.n_action_steps, policy_path=path)
        resets = 0
        while True:
            try:
                request = receive_packet(sock)
            except ConnectionError:
                break
            command = str(request["command"].item())
            if command == "reset":
                policy.reset()
                pre.reset()
                post.reset()
                resets += 1
                send_packet(sock, reset_count=resets)
                continue
            if command != "action":
                raise ValueError(f"Unknown worker command: {command}")
            state, rgb = request["state"], request["rgb"]
            if state.shape != (6,) or not np.isfinite(state).all():
                raise ValueError("Expected six finite LeRobot joint values")
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                raise ValueError("Expected uint8 RGB image [480,640,3]")
            batch = {
                "observation.state": torch.from_numpy(state.astype(np.float32)),
                "observation.images.front": torch.from_numpy(rgb.copy()).permute(2,0,1).float() / 255.0,
            }
            with torch.inference_mode():
                processed = pre(batch)
                if processed["observation.state"].shape != (1,6):
                    raise ValueError("Saved preprocessor produced an unexpected state batch shape")
                action = post(policy.select_action(processed)).detach().cpu().numpy()
            if action.shape != (1,6) or not np.isfinite(action).all():
                raise ValueError("ACT produced an invalid action")
            send_packet(sock, action=action[0])
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
