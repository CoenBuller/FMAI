import json
import numpy as np
import torch

from pathlib import Path
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from torchrl.envs.utils import set_exploration_type, ExplorationType


# ------------------------ MONITOR UTILITIES ---------------------------

# Helper function to get the maximum euclidean distance to the goal between agents
def dgoal_max(obs):                      
    return torch.linalg.vector_norm(obs[..., 4:6], dim=-1).max(-1).values

def dgoal_avg(obs):                      
    return torch.linalg.vector_norm(obs[..., 4:6], dim=-1)

# Get euclidean distance tothe goal for a single agent
def dgoal(a):                            
    return lambda obs: torch.linalg.vector_norm(obs[..., a, 4:6], dim=-1)

def lidar_max(obs):                       # all rays of all agents
    return obs[..., 6:18].amax(dim=(-2, -1))

def lidar_avg(obs):
    return obs[..., 6:18].amax(dim=-1)

# Possible agregation functions, can run natively on the gpu to create signal traces before calculating robustness
AGG = {
    "min": torch.min, 
    "max": torch.max, 
    "mean": torch.mean,
    "dg_max": dgoal_max,
    "dg": dgoal,
    "lmax": lidar_max
    }

# ------------------------ TRAIN UTILITIES ---------------------------



# Function to map formulas to the signal maps
def batch_formulas(formulas, signal_maps):
    phis = dict()
    for i, f in enumerate(formulas):
        phis[f] = signal_maps[i]
    return phis

def with_episode_ids(obs, done):
    """Append the episode index of every step as an extra channel. obs: [E,T,A,D], done: [E,T,1]."""
    d = done.squeeze(-1).long()
    seg = d.cumsum(1) - d                                   # episode index, exclusive of the current done
    return torch.cat([obs, seg[..., None, None].expand(*obs.shape[:3], 1).to(obs.dtype)], -1)

def lag(x, dt, seg):
    """x[t - dt] for x: [E,T,A]; NaN if t < dt or t - dt is in an earlier episode."""
    out = torch.full_like(x, float("nan"))
    out[:, dt:] = x[:, :-dt]
    same = torch.zeros_like(seg, dtype=torch.bool)
    same[:, dt:] = seg[:, dt:] == seg[:, :-dt]
    return torch.where(same[..., None], out, torch.full_like(out, float("nan")))

class Defined:
    def __init__(self, f): self.f = f
    def __call__(self, obs):
        return torch.nan_to_num(self.f(obs), nan=float("-inf"))   # undefined never wins an Or


LIDAR_SAFE = 0.2     # obs = 0.35 - distance to the other agent's surface, so < 0.2 means more than 0.15 away
GOAL_EPS = 0.1       # the env's done radius
V_MIN = 0.01         # required progress per step; check against the diagnostic below

def build_spec():
    da, dg = lidar_avg, dgoal_avg                    # [E,T,A]; rename them, they are per-agent max/distance, not averages

    safety = Pred(da, "<", LIDAR_SAFE)

    def branch(dt):
        gain = lambda obs: lag(dg(obs), dt, obs[..., 0, -1]) - dg(obs)   # last channel = episode id
        return Or(
            Pred(da, ">=", LIDAR_SAFE),              # dodging: progress is not required
            Defined(Pred(gain, ">", V_MIN * dt)),
            Pred(dg, "<", GOAL_EPS),
        )

    phi = And(safety, branch(1))
    desc = [f"safe: da < {LIDAR_SAFE}  AND  exists dt in 1: (da >= {LIDAR_SAFE} || dg(t-dt)-dg(t) > {V_MIN}*dt || dg < {GOAL_EPS})"]
    return WeightedRobustness([phi], weights=None, clip=1.0), desc



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
