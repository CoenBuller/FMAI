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

# Saves history of run
def save_history(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    tmp.replace(path)  

# mean/std that don't give NaN for a single sample
def ms(x): 
    return x.mean().item(), x.std(correction=0).item()

# Function to map formulas to the signal maps
def batch_formulas(formulas, signal_maps):
    phis = dict()
    for i, f in enumerate(formulas):
        phis[f] = signal_maps[i]
    return phis

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
