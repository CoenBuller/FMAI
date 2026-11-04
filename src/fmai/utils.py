import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from torchrl.envs.utils import set_exploration_type, ExplorationType

from fmai.robustness import WeightedRobustness, Pred, Or, And, Defined, Not

W = 1

# ------------------------ MONITOR UTILITIES ---------------------------

def with_episode_ids(obs, done):
    """Append the episode index of each step as the last channel. obs: [E,T,A,D], done: [E,T,1]."""
    d = done.squeeze(-1).long()
    seg = d.cumsum(1) - d                                    # index of the episode a step belongs to
    ch = seg[..., None, None].expand(*obs.shape[:3], 1).to(obs.dtype)
    return torch.cat([obs, ch], dim=-1)

def first_episode_mask(obs):                                 # [E,T] True for steps of the first episode
    return obs[..., 0, -1] == 0

# Used during the evaluation run. 
def rho_per_env(spec, rollout):
    """Evaluation robustness: worst robustness per environment (after the first W steps) -> [E]."""
    obs = with_episode_ids(rollout["next", "agents", "observation"], rollout["next", "done"])
    rho = spec(obs)                                          # [E,T,A]
    valid = first_episode_mask(obs).clone()
    valid[:, :W] = False                                     # lags undefined at the start
    return rho.masked_fill(~valid[..., None], float("inf")).amin(dim=(1, 2))


# ------------------------ SFO specification ---------------------------

# Helper function to get the maximum euclidean distance to the goal between agents

def dgoal(obs):                      
    return torch.linalg.vector_norm(obs[..., 4:6], dim=-1)

def dlidar(obs):
    return obs[..., 6:18].amax(dim=-1)

def lag(x, s, seg):
    """x[t - s] for x: [E,T,A]; NaN if t < s or t - s lies in an earlier episode. seg: [E,T]."""
    nan = float("nan")
    out = torch.full_like(x, nan)
    out[:, s:] = x[:, :-s]
    same = torch.zeros_like(seg, dtype=torch.bool)
    same[:, s:] = seg[:, s:] == seg[:, :-s]
    return torch.where(same[..., None], out, torch.full_like(out, nan))

def vgoal(s):
    """dg(t-s) - dg(t) >= V*s   (margin: gain - V*s)"""
    gain = lambda o: lag(dgoal(o), s, o[..., 0, -1]) - dgoal(o)
    return gain

def vagent(s):
    gain = lambda o: lag(dlidar(o), s, o[..., 0, -1]) - dlidar(o)
    return gain


def build_spec():
    """At every step: (at goal  OR  progressed over some window s<=W)  AND  separated from all other agents."""
    
    # Predicates 
    dg = Pred(dgoal, "<=", 0.1)
    vg = [Defined(Pred(vgoal(s), ">", 0.0)) for s in range(1, 1+W)]
    da = Pred(dlidar, "<", 0.2)
    va = [Defined(Pred(vagent(s), ">", 0.0)) for s in range(1, 1+W)]

    # SFO formula in CNF form
    phi = And(
        Or(Not(da), dg, *vg),
        da,
    )

    desc = [f"dg<={0.1} || dg(t-1)-dg(t) >= {0.05} ; and dsep>={0.2}"]
    return WeightedRobustness([phi], weights=None, clip=1.0), desc


#--------------------------- Bookkeeping ----------------------------#

def create_paths(base_dir, run_id, stem):
    paths = {
        "History": Path(base_dir) / "traces" / f"{run_id}" / f"{stem}.json",
        "Image": Path(base_dir) / "traces" / f"{run_id}" / f"{stem}.png",
        "Agent": Path(base_dir) / "agents" / f"{run_id}" / f"{stem}.pt",
        "Gif": Path(base_dir) / "gifs" / f"{run_id}" / f"{stem}.gif",
        "Error": Path(base_dir) / "error" / f"{run_id}.txt",
    }
    return paths
n_logged = 0


def log_skip(it, why, err_path, lv=None, gn=None):
    err_path = Path(err_path)
    err_path.parent.mkdir(parents=True, exist_ok=True)
    global n_logged
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

def save_plots(history, path):
    """Plot return, robustness, and PPO loss curves from history."""
    if not history:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    eps = [h["episode"] for h in history]
    ret_mean = np.array([h["return_mean"] for h in history])
    ret_std = np.array([h["return_std"] for h in history])
    rho_mean = np.array([h["rho_mean"] for h in history])
    rho_std = np.array([h["rho_std"] for h in history])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("IPPO Training & Evaluation Diagnostics", fontsize=14)

    # 1. Evaluation Return (+/- 1 std) & Train Return
    ax = axes[0, 0]
    ax.plot(eps, ret_mean, label="Eval Return", color="tab:blue")
    ax.fill_between(eps, ret_mean - ret_std, ret_mean + ret_std, color="tab:blue", alpha=0.2)
    if "train_return" in history[0]:
        ax.plot(eps, [h["train_return"] for h in history], "--", label="Train Return", color="tab:cyan", alpha=0.8)
    ax.set_title("Episode Return")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Robustness (rho) & Pass Rate
    ax = axes[0, 1]
    ax.plot(eps, rho_mean, label="Eval Rho", color="tab:green")
    ax.fill_between(eps, rho_mean - rho_std, rho_mean + rho_std, color="tab:green", alpha=0.2)
    ax.axhline(0.0, color="red", linestyle=":", linewidth=1, label="Threshold (0)")
    ax.set_title("STL/SFO Robustness (rho)")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Robustness")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(eps, [h["pass_rate"] for h in history], color="tab:orange", linestyle="--", label="Pass Rate")
    ax2.set_ylabel("Pass Rate", color="tab:orange")
    ax2.set_ylim(-0.05, 1.05)

    # 3. Policy Entropy & KL Divergence
    ax = axes[0, 2]
    if "entropy" in history[0]:
        ax.plot(eps, [h["entropy"] for h in history], color="tab:purple", label="Policy Entropy")
    ax.set_title("Policy Entropy (Exploration)")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Entropy")
    ax.grid(True, alpha=0.3)
    if "kl_approx" in history[0]:
        ax_kl = ax.twinx()
        ax_kl.plot(eps, [h["kl_approx"] for h in history], color="tab:brown", linestyle="--", label="Approx KL")
        ax_kl.set_ylabel("Approx KL", color="tab:brown")

    # 4. Actor (Objective) Loss
    ax = axes[1, 0]
    if "loss_objective" in history[0]:
        ax.plot(eps, [h["loss_objective"] for h in history], color="tab:red")
    ax.set_title("Actor Loss (loss_objective)")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # 5. Critic (Value) Loss
    ax = axes[1, 1]
    if "loss_critic" in history[0]:
        ax.plot(eps, [h["loss_critic"] for h in history], color="tab:orange")
    ax.set_title("Critic Loss (loss_critic)")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # 6. Gradient Norm & Skipped Updates
    ax = axes[1, 2]
    if "grad_norm" in history[0]:
        ax.plot(eps, [h["grad_norm"] for h in history], color="tab:gray", label="Grad Norm (pre-clip)")
    ax.set_title("Gradient Norm & Skipped Steps")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("L2 Norm")
    ax.grid(True, alpha=0.3)
    if "skipped_updates" in history[0]:
        ax_sk = ax.twinx()
        ax_sk.bar(eps, [h["skipped_updates"] for h in history], width=max(eps[-1] // max(len(eps), 1) * 0.4, 1), alpha=0.3, color="red")
        ax_sk.set_ylabel("Skipped Updates", color="red")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
