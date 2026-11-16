import json, warnings
import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from torchrl.envs.utils import set_exploration_type, ExplorationType

from fmai.sfo import TAU1, TAU2


W = TAU1 + TAU2

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
    
    try:                                                     # rendering needs a display; headless nodes may not have one
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            eval_env.rollout(
                max_steps=max_steps,
                policy=agent.policy,
                break_when_any_done=False,
                callback=render_cb,
            )
    except Exception as err:
        print(f"[save_gif] skipped, rendering failed: {err!r}")
        return

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

    eps = [h["step"] for h in history]
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
    ax.set_xlabel("Env steps")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Robustness (rho) & Pass Rate
    ax = axes[0, 1]
    ax.plot(eps, rho_mean, label="Eval Rho", color="tab:green")
    ax.fill_between(eps, rho_mean - rho_std, rho_mean + rho_std, color="tab:green", alpha=0.2)
    ax.axhline(0.0, color="red", linestyle=":", linewidth=1, label="Threshold (0)")
    ax.set_title("STL/SFO Robustness (rho)")
    ax.set_xlabel("Env steps")
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
    ax.set_xlabel("Env steps")
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
    ax.set_xlabel("Env steps")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # 5. Critic (Value) Loss
    ax = axes[1, 1]
    if "loss_critic" in history[0]:
        ax.plot(eps, [h["loss_critic"] for h in history], color="tab:orange")
    ax.set_title("Critic Loss (loss_critic)")
    ax.set_xlabel("Env steps")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # 6. Gradient Norm & Skipped Updates
    ax = axes[1, 2]
    if "grad_norm" in history[0]:
        ax.plot(eps, [h["grad_norm"] for h in history], color="tab:gray", label="Grad Norm (pre-clip)")
    ax.set_title("Gradient Norm & Skipped Steps")
    ax.set_xlabel("Env steps")
    ax.set_ylabel("L2 Norm")
    ax.grid(True, alpha=0.3)
    if "skipped_updates" in history[0]:
        ax_sk = ax.twinx()
        ax_sk.bar(eps, [h["skipped_updates"] for h in history], width=max(eps[-1] // max(len(eps), 1) * 0.4, 1), alpha=0.3, color="red")
        ax_sk.set_ylabel("Skipped Updates", color="red")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


#--------------------------- Compare mode (PLOT_MODE="compare") ----------------------------#

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # fixed categorical order
COMPARE_CURVES = [("return_mean", "Eval return (native)"), ("rho_mean", "Eval robustness ρ"),
                  ("loss_objective", "Actor loss"), ("loss_critic", "Critic loss")]
COMPARE_TABLE = ["return_mean", "rho_mean", "pass_rate", "success_rate", "violation_rate",
                 "min_separation", "collision_steps"]


def option_label(cfg):
    """Name of a training option: 'native' for the plain reward, else '<mode>_<spec>' (e.g. rho_coen)."""
    return "native" if cfg.REWARD_MODE == "none" else f"{cfg.REWARD_MODE}_{cfg.SPEC}"


def smooth(y, k):
    """Trailing moving average over k points that ignores NaNs (the untrained eval has no losses)."""
    y = np.asarray(y, dtype=float)
    if k <= 1:
        return y
    ok = ~np.isnan(y)
    s = np.concatenate([[0.0], np.cumsum(np.where(ok, y, 0.0))])
    n = np.concatenate([[0], np.cumsum(ok)])
    lo = np.maximum(np.arange(len(y)) + 1 - k, 0)
    cnt = n[1:] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (s[1:] - s[lo]) / cnt, np.nan)


def save_compare_plots(results, path, smooth_k=1, title=""):
    """results: {option label: [history of seed 0, seed 1, ...]}. Writes <path>.png (curves, mean ± std over
    seeds, each seed smoothed first) and <path>.csv (final evaluation, mean ± std over seeds)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for k, (label, hists) in enumerate(results.items()):
        n = min(len(h) for h in hists)                                   # cut to the shortest run
        x = np.array([e["step"] for e in hists[0][:n]])
        for ax, (key, name) in zip(axes.flat, COMPARE_CURVES):
            ys = np.stack([smooth([e[key] for e in h[:n]], smooth_k) for h in hists])
            with warnings.catch_warnings():                              # step 0 has no losses (all NaN)
                warnings.simplefilter("ignore", RuntimeWarning)
                mean, std = np.nanmean(ys, 0), np.nanstd(ys, 0)
            ax.plot(x, mean, color=COLORS[k % len(COLORS)], linewidth=2, label=f"{label} ({len(hists)} seeds)")
            ax.fill_between(x, mean - std, mean + std, color=COLORS[k % len(COLORS)], alpha=0.2, linewidth=0)
            ax.set(title=name, xlabel="Env steps")
            ax.grid(True, alpha=0.3)
    axes[0, 1].axhline(0.0, color="gray", linestyle=":", linewidth=1)
    axes[0, 0].legend()
    fig.suptitle(f"{title}  |  mean ± std over seeds, smoothing window {smooth_k}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)

    rows = ["option,n_seeds," + ",".join(f"{m}_mean,{m}_std" for m in COMPARE_TABLE)]
    print(f"\nFinal evaluation (mean +/- std over seeds)\n{'option':14s}" + "".join(f"{m:>22s}" for m in COMPARE_TABLE))
    for label, hists in results.items():
        vals = np.array([[h[-1][m] for m in COMPARE_TABLE] for h in hists])   # [seeds, metrics]
        mean, std = vals.mean(0), vals.std(0)
        rows.append(f"{label},{len(hists)}," + ",".join(f"{a:.4f},{b:.4f}" for a, b in zip(mean, std)))
        print(f"{label:14s}" + "".join(f"{a:>12.3f} +/- {b:<6.3f}" for a, b in zip(mean, std)))
    path.with_suffix(".csv").write_text("\n".join(rows) + "\n")
