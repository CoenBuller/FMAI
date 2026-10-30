import json
import numpy as np
import torch

from pathlib import Path
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from torchrl.envs.utils import set_exploration_type, ExplorationType

from fmai.robustness import WeightedRobustness, Pred, Or, And, Defined

# ------------------------ MONITOR UTILITIES ---------------------------

# Helper function to get the maximum euclidean distance to the goal between agents

def dgoal(obs):                      
    return torch.linalg.vector_norm(obs[..., 4:6], dim=-1)

def dlidar(obs):
    return obs[..., 6:18].amax(dim=-1)



# ------------------------ TRAIN UTILITIES ---------------------------

def with_episode_ids(obs, done):
    """Append the episode index of each step as the last channel. obs: [E,T,A,D], done: [E,T,1]."""
    d = done.squeeze(-1).long()
    seg = d.cumsum(1) - d                                    # index of the episode a step belongs to
    ch = seg[..., None, None].expand(*obs.shape[:3], 1).to(obs.dtype)
    return torch.cat([obs, ch], dim=-1)

def lag(x, s, seg):
    """x[t - s] for x: [E,T,A]; NaN if t < s or t - s lies in an earlier episode. seg: [E,T]."""
    nan = float("nan")
    out = torch.full_like(x, nan)
    out[:, s:] = x[:, :-s]
    same = torch.zeros_like(seg, dtype=torch.bool)
    same[:, s:] = seg[:, s:] == seg[:, :-s]
    return torch.where(same[..., None], out, torch.full_like(out, nan))

def dgoal_i(obs):                                            # |p_i - g_i|, obs[4:6] = pos - goal     -> [E,T,A]
    return torch.linalg.vector_norm(obs[..., 4:6], dim=-1)

def dsep_i(obs):                                             # min distance to any other agent        -> [E,T,A]
    E, T, A, _ = obs.shape
    p = obs[..., 0:2].reshape(E * T, A, 2)
    d = torch.cdist(p, p) + 1e6 * torch.eye(A, device=obs.device)
    return d.min(-1).values.reshape(E, T, A)

def first_episode_mask(obs):                                 # [E,T] True for steps of the first episode
    return obs[..., 0, -1] == 0


V, W, EPS, D = 0.02, 5, 0.1, 0.3     # required progress/step, max lag, goal radius, safety distance

def progress_branch(s):
    """dg(t-s) - dg(t) >= V*s   (margin: gain - V*s)"""
    gain = lambda o: lag(dgoal_i(o), s, o[..., 0, -1]) - dgoal_i(o)
    return Defined(Pred(gain, ">=", V * s))

def build_spec():
    """At every step: (at goal  OR  progressed over some window s<=W)  AND  separated from all other agents."""
    phi = And(
        Or(Pred(dgoal_i, "<=", EPS), *[progress_branch(s) for s in range(1, W + 1)]),
        Pred(dsep_i, ">=", D),
    )
    desc = [f"dg<={EPS} || exists s in [1,{W}]: dg(t-s)-dg(t) >= {V}*s ; and dsep>={D}"]
    return WeightedRobustness([phi], weights=None, clip=1.0), desc

def rho_per_env(spec, rollout):
    """Evaluation robustness: worst margin over the first episode (after the first W steps) and all agents -> [E]."""
    obs = with_episode_ids(rollout["next", "agents", "observation"], rollout["next", "done"])
    rho = spec(obs)                                          # [E,T,A]
    valid = first_episode_mask(obs).clone()
    valid[:, :W] = False                                     # lags undefined at the start
    return rho.masked_fill(~valid[..., None], float("inf")).amin(dim=(1, 2))



#--------------------------- Bookkeeping ----------------------------#
n_logged = 0

def log_skip(it, why, error_path, lv=None, gn=None):
    err_path.parent.mkdir(parents=True, exist_ok=True)
    global n_logged            # drop this and use a list/dict if this lives inside train()
    if n_logged >= 50:
        return
    n_logged += 1
    row = {"it": it, "why": why}
    if lv is not None:
        row["losses"] = {k: v.item() for k, v in lv.items() if v.ndim == 0}
    if gn is not None:
        row["grad_norm"] = float(gn)
    with open(err_path, "a") as f:
        f.write(json.dumps(row) + "\n")

# mean/std that don't give NaN for a single sample
def ms(x): 
    return x.mean().item(), x.std(correction=0).item()

# Saves history of run
def save_history(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    tmp.replace(path)  

def save_gif(eval_env, agent, max_steps, path, fps=30):
    """Render env 0 of eval_env for one deterministic episode and save it as a GIF."""
    frames = []
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def render_cb(env, td):
        frames.append(eval_env.render(mode="rgb_array", agent_index_focus=None, env_index=0))
    
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        eval_env.rollout(
            max_steps=max_steps,
            policy=agent.policy,
            break_when_any_done=False,
            callback=render_cb,
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ImageSequenceClip(frames, fps=fps).write_gif(str(path), logger=None)

def save_agent(agent, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": agent.policy.state_dict(),                  # actor only, enough for evaluation / GIFs
            "loss_module": agent.loss_module.state_dict(),        # actor + critic (+ target params if any)
            "optim": agent.optim.state_dict(),                    # only needed to resume training
        },
        path,
    )
