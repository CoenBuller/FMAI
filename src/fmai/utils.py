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

@torch.no_grad()
def sub_rho_per_env(spec, rollout):
    """Evaluates each sub-formula in `spec.sub_formulas` to diagnose why an env fails.
    Returns (sub_rho_means, sub_fail_rates) dictionaries mapping clause_name -> float."""
    sub_formulas = getattr(spec, "sub_formulas", None)
    if not sub_formulas:
        return {}, {}
    obs = with_episode_ids(rollout["next", "agents", "observation"], rollout["next", "done"])
    valid = first_episode_mask(obs).clone()
    valid[:, :W] = False

    sub_rho_means, sub_fail_rates = {}, {}
    for name, f in sub_formulas.items():
        r = f(obs).clamp(-spec.clip, spec.clip)              # [E,T,A]
        r_env = r.masked_fill(~valid[..., None], float("inf")).amin(dim=(1, 2))  # [E]
        sub_rho_means[name] = r_env.mean().item()
        sub_fail_rates[name] = (r_env < 0).float().mean().item()
    return sub_rho_means, sub_fail_rates


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

# Results layout. Flat (default):        <RESULTS_DIR>/{agents,gifs,traces,error}/<run_id>/
# SPLIT_RESULTS (one folder per option): <RESULTS_DIR>/<native|robustness>/{agents,gifs,traces,error}/<run_id>/
#                                        <RESULTS_DIR>/comparison/<run_id>/  (compare, analysis, scaling, summary)
OPTION_DIRS = {"none": "native", "rho": "robustness"}

def run_root(cfg):
    """Folder that holds agents/, gifs/, traces/ and error/ of a run."""
    root = Path(cfg.RESULTS_DIR)
    return root / OPTION_DIRS[cfg.REWARD_MODE] if getattr(cfg, "SPLIT_RESULTS", False) else root

def find_traces(results_dir, run_id):
    """The traces/<run_id> folders of a run: one per option folder (split layout) or the single flat one."""
    root = Path(results_dir)
    flat = root / "traces" / run_id
    return sorted(root.glob(f"*/traces/{run_id}")) + ([flat] if flat.is_dir() else [])

def agents_dir(traces_dir):
    """<root>/traces/<run_id> -> <root>/agents/<run_id> (the checkpoints of the same runs)."""
    return Path(traces_dir).parent.parent / "agents" / Path(traces_dir).name

def comparison_dir(results_dir, run_id=None):
    """Where the figures over all options of a run go: comparison/<run_id> (split) or traces/<run_id> (flat)."""
    root = Path(results_dir)
    split = any(root.glob("*/traces")) or not (root / "traces").is_dir()
    base = root / "comparison" if split else root / "traces"
    return base / run_id if run_id else base
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

def save_gif(eval_env, agent, max_steps, path, fps=30, title="", zone_radius=0.25):
    """Roll out env 0 of eval_env for one deterministic episode and save it as a GIF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Use native VMAS rgb_array rendering if 'pos' is not in info, otherwise draw_frames
    frames = []
    def render_cb(env, td):
        frames.append(eval_env.render(mode="rgb_array", agent_index_focus=None, env_index=0))

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = eval_env.rollout(
            max_steps=max_steps,
            policy=None if agent is None else agent.policy,
            break_when_any_done=False,
            callback=render_cb,
        )
    if not frames and "pos" in td["next", "agents", "info"].keys():
        frames = draw_frames(td, title=title, zone_radius=zone_radius)
    if frames:
        ImageSequenceClip(frames, fps=fps).write_gif(str(path), logger=None)

def draw_frames(td, env_index=0, radius=0.1, title="", zone_radius=0.25):
    """Frames of one env's first episode: danger zones (red), goals (pale), trails, agents, catastrophes (X)."""
    inf = td["next", "agents", "info"]
    pos, goal = inf["pos"][env_index].cpu().numpy(), inf["goal"][env_index].cpu().numpy()    # [T, A, 2]
    done = td["next", "done"][env_index].reshape(-1).cpu().numpy()
    T = int(done.argmax()) + 1 if done.any() else len(done)                                  # first episode only
    zones = inf["zones"][env_index, 0, 0].cpu().numpy() if "zones" in inf.keys() else np.zeros((0, 2))   # [K, 2]
    zones = zones[np.abs(zones).max(-1) < 50]                                                # drop unused zones
    cat = inf["catastrophe"][env_index].cpu().numpy() if "catastrophe" in inf.keys() else np.zeros(pos.shape[:2])
    inside = (inf["mu3"][env_index].cpu().numpy() < 0) if "mu3" in inf.keys() else np.zeros(pos.shape[:2], bool)
    pts = np.concatenate([pos[:T].reshape(-1, 2), goal[:T].reshape(-1, 2), zones.reshape(-1, 2), [[-1, -1], [1, 1]]])
    lo, hi = pts.min(0) - 0.35, pts.max(0) + 0.35
    colors = [c for k, c in enumerate(plt.get_cmap("tab10").colors) if k != 3]             # no red: that's the zones

    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=80)
    frames = []
    for t in range(T):
        ax.clear()
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for z in zones:
            ax.add_patch(plt.Circle(z, zone_radius, facecolor="#d62728", edgecolor="#8b0000", alpha=0.3, lw=1.5))
        for i in range(pos.shape[1]):
            c = colors[i % len(colors)]
            ax.add_patch(plt.Circle(goal[t, i], radius, color=c, alpha=0.25))                    # goal
            ax.plot(pos[:t + 1, i, 0], pos[:t + 1, i, 1], color=c, lw=1, alpha=0.5)             # trail
            ax.add_patch(plt.Circle(pos[t, i], radius, color=c))                                 # agent
            for s in np.nonzero(cat[:t + 1, i])[0]:                                              # catastrophes so far
                ax.plot(*pos[s, i], marker="x", color="black", ms=9, mew=2.5)
        n_in, n_cat = int(inside[t].sum()), int(cat[:t + 1].sum())
        ax.set_title(f"{title + chr(10) if title else ''}step {t + 1}/{T} | in zone: {n_in} | catastrophes: {n_cat}",
                     fontsize=8)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return frames

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
    """Plot return, SFO robustness, reward component breakdown, and SFO failure diagnostics (2x2 grid)."""
    if not history:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x_key = "step" if "step" in history[0] else "episode"
    x_label = "Env steps" if x_key == "step" else "Episodes"
    eps = [h[x_key] for h in history]

    ret_mean = np.array([h["return_mean"] for h in history])
    ret_std = np.array([h["return_std"] for h in history])
    rho_mean = np.array([h["rho_mean"] for h in history])
    rho_std = np.array([h["rho_std"] for h in history])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Reward Breakdown & SFO Failure Diagnostics", fontsize=14)

    # 1. Evaluation Return (+/- 1 std), Train Return & Shaped Reward
    ax = axes[0, 0]
    ax.plot(eps, ret_mean, label="Eval Return (Native)", color="tab:blue")
    ax.fill_between(eps, ret_mean - ret_std, ret_mean + ret_std, color="tab:blue", alpha=0.2)
    if "train_return" in history[0]:
        ax.plot(eps, [h["train_return"] for h in history], "--", label="Train Return (Native)", color="tab:cyan", alpha=0.8)
    if "train_reward" in history[0]:
        ax.plot(eps, [h["train_reward"] for h in history], ":", label="Train Reward (Shaped)", color="tab:purple", alpha=0.8)
    ax.set_title("Total Episode Return & Shaped Reward")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Return")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Overall Robustness (rho) & Pass Rate
    ax = axes[0, 1]
    ax.plot(eps, rho_mean, label="Eval Rho (Overall)", color="tab:green")
    ax.fill_between(eps, rho_mean - rho_std, rho_mean + rho_std, color="tab:green", alpha=0.2)
    ax.axhline(0.0, color="red", linestyle=":", linewidth=1, label="Threshold (0)")
    ax.set_title("Overall SFO Robustness (rho) & Pass Rate")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Robustness")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(eps, [h["pass_rate"] for h in history], color="tab:orange", linestyle="--", label="Pass Rate")
    ax2.set_ylabel("Pass Rate", color="tab:orange")
    ax2.set_ylim(-0.05, 1.05)

    # 3. Native Reward Component Breakdown (r_d, r_p, r_v_rel, r_v_max)
    ax = axes[1, 0]
    if "reward_components" in history[0]:
        for ck in history[0]["reward_components"].keys():
            ax.plot(eps, [h["reward_components"].get(ck, float("nan")) for h in history], label=ck, linewidth=1.8)
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
        ax.legend(fontsize=8)
    elif "success_rate" in history[0]:
        ax.plot(eps, [h["success_rate"] for h in history], label="Success Rate (Band/Goal)", color="tab:blue")
        ax.plot(eps, [h["violation_rate"] for h in history], label="Violation Rate (Sep < 0)", color="tab:red")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
    ax.set_title("Individual Reward Components (Eval Episode Sum)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Component Return")
    ax.grid(True, alpha=0.3)

    # 4. Why an Env Fails: Per-Clause Failure Rates & Sub-Formula Robustness
    ax = axes[1, 1]
    if "sub_fail_rate" in history[0]:
        for name in history[0]["sub_fail_rate"].keys():
            ax.plot(eps, [h["sub_fail_rate"].get(name, float("nan")) for h in history], label=f"Fail: {name}", linewidth=1.8)
    if "violation_rate" in history[0]:
        ax.plot(eps, [h["violation_rate"] for h in history], "--", color="black", alpha=0.6, label="Lidar Viol (mu2 < 0)")
    ax.set_title("Why Envs Fail (Failure Rate per SFO Clause)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Fraction of Envs Failing [0, 1]")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


#--------------------------- Compare mode (PLOT_MODE="compare") ----------------------------#

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#8a5cd1", "#17becf", "#8c564b",
          "#7a7a7a"]
OPTION_ORDER = ["native", "rho_robustness"]

def option_color(label):
    """Same colour for an option in every figure, however many options a run has."""
    k = OPTION_ORDER.index(label) if label in OPTION_ORDER else len(OPTION_ORDER)
    return COLORS[k % len(COLORS)]

COMPARE_CURVES = [("return_mean", "Eval return (native)"), ("rho_mean", "Eval robustness ρ"),
                  ("pass_rate", "SFO Pass Rate"), ("violation_rate", "Safety Violation Rate")]
COMPARE_TABLE = ["return_mean", "rho_mean", "pass_rate", "success_rate", "violation_rate",
                 "min_separation", "collision_steps", "zone_entry_rate", "catastrophes"]


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
    """results: {option label: [history of seed 0, seed 1, ...]}. Writes <path>.png and <path>.csv."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for k, (label, hists) in enumerate(results.items()):
        n = min(len(h) for h in hists)
        x_key = "step" if "step" in hists[0][0] else "episode"
        x = np.array([e[x_key] for e in hists[0][:n]])
        for ax, (key, name) in zip(axes.flat, COMPARE_CURVES):
            ys = np.stack([smooth([e[key] for e in h[:n]], smooth_k) for h in hists])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean, std = np.nanmean(ys, 0), np.nanstd(ys, 0)
            ax.plot(x, mean, color=option_color(label), linewidth=2, label=f"{label} ({len(hists)} seeds)")
            ax.fill_between(x, mean - std, mean + std, color=option_color(label), alpha=0.2, linewidth=0)
            ax.set(title=name, xlabel="Env steps" if x_key == "step" else "Episodes")
            ax.grid(True, alpha=0.3)
    axes[0, 1].axhline(0.0, color="gray", linestyle=":", linewidth=1)
    axes[0, 0].legend()
    fig.suptitle(f"{title}  |  mean ± std over seeds, smoothing window {smooth_k}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)

    metrics = [m for m in COMPARE_TABLE if m in next(iter(results.values()))[0][-1]]
    rows = ["option,n_seeds," + ",".join(f"{m}_mean,{m}_std" for m in metrics)]
    print(f"\nFinal evaluation (mean +/- std over seeds)\n{'option':14s}" + "".join(f"{m:>22s}" for m in metrics))
    for label, hists in results.items():
        vals = np.array([[h[-1][m] for m in metrics] for h in hists])
        mean, std = vals.mean(0), vals.std(0)
        rows.append(f"{label},{len(hists)}," + ",".join(f"{a:.4f},{b:.4f}" for a, b in zip(mean, std)))
        print(f"{label:14s}" + "".join(f"{a:>12.3f} +/- {b:<6.3f}" for a, b in zip(mean, std)))
    path.with_suffix(".csv").write_text("\n".join(rows) + "\n")