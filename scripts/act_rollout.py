"""Vector evaluation of a chosen frozen base policy, optionally with a PPO residual."""
import json
from pathlib import Path
import numpy as np
from policy_session import PolicySession
from run_output import RunOutput
ROOT = Path(__file__).resolve().parents[1]


def rollout(env,args,calibration):
    output = RunOutput(args,ROOT,'residual_eval' if args.mode == 'residual-ppo' else 'policy_eval')
    session = None
    status = 'error'
    accepted = []
    counts = np.zeros(env.num_envs,dtype=int)
    try:
        session = PolicySession(env,args,calibration,output)
        model = None
        if args.mode == 'residual-ppo':
            from residual_ppo import load_residual
            model = load_residual(args.checkpoint,session.contract,args.ppo_device)
        obs = session.reset()
        while (counts < args.episodes).any() and (not args.steps or session.steps < args.steps):
            if env.sim.visualizers and not any(v.is_running() and not v.is_closed for v in env.sim.visualizers): break
            correction = np.zeros((env.num_envs,6),dtype=np.float32) if model is None else model.predict(obs,deterministic=True)[0]
            obs,_,done,infos = session.step(correction)
            for i in np.flatnonzero(done):
                if counts[i] >= args.episodes: continue
                counts[i] += 1
                row = {k:v for k,v in infos[i].items() if k not in ('episode','terminal_observation','TimeLimit.truncated')}
                accepted.append(row)
                print('Evaluation episode: '+json.dumps(row),flush=True)
        status = 'completed' if (counts >= args.episodes).all() else 'stopped_early'
    except KeyboardInterrupt:
        status = 'interrupted'
    finally:
        summary = session.summary(status,accepted) if session else {'status':status}
        summary['requested_episodes_per_env'] = args.episodes
        summary['completed_per_env'] = counts.tolist()
        if session: session.close()
        output.finish(summary,session.last_rgb[0] if session and session.last_rgb is not None else None)
