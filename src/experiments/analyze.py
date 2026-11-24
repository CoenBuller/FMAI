"""Post-training robustness analysis: every checkpoint of a run is rolled out on the same EPISODES environments
(fixed seed, deterministic policy, full MAX_STEPS), and the terms of the gated spec (GatedSpec: Prog, Agent,
Zone) and the total robustness are plotted over time, per option: mean +/- 1 std over all episodes of
all seeds. The value per episode and step is the worst agent (min over agents), as the spec must hold for all.

    PYTHONPATH=src python -m experiments.analyze --results results/navigation --run-id <run_id>

Finds every traces/<run_id> folder of the run (one per option, see utils.find_traces; or pass them with --traces)
and the checkpoints next to them (agents/<run_id>). Writes analysis.png and analysis.csv into
<results>/comparison/<run_id> (or --out).
"""
import argparse, csv, dataclasses, json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchrl.envs.utils import set_exploration_type, ExplorationType

from config import Config
from environment.environment import make_env
from fmai.agents import Agent
from fmai.sfo import GatedSpec
from fmai.utils import option_label, option_color, OPTION_ORDER, find_traces, agents_dir, comparison_dir

TERMS = [("prog", r"$\rho_{Prog}$: goal progress"), ("agent", r"$\rho_{Agent}$: agent spacing"),
         ("total", r"$\rho$: total robustness")]
ZONE_TERMS = [("prog", r"$\rho_{Prog}$: goal progress"), ("agent", r"$\rho_{Agent}$: agent spacing"),
              ("zone", r"$\rho_{Zone}$: danger zones"), ("total", r"$\rho$: total robustness"),
              ("in_zone", "fraction of agents inside a zone")]


def rollout_terms(cfg_dict, ckpt, episodes, seed, device):
    fields = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in fields})
    cfg.DEVICE = device
    cfg.ACTIVATION = {"ReLu": "ReLU"}.get(cfg.ACTIVATION, cfg.ACTIVATION)   # runs from before the yaml fix
    env = make_env(cfg, episodes, seed)                                  # same seed -> same spawns for every model
    agent = Agent()
    agent.make_agent(env=env, cfg=cfg)
    agent.policy.load_state_dict(torch.load(ckpt, map_location=device)["policy"])
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = env.rollout(max_steps=cfg.MAX_STEPS, policy=agent.policy, break_when_any_done=False)
    inf = td["next", "agents", "info"]
    zones = inf["zones"][..., 0, :, :] if cfg.N_ZONES else None
    terms = GatedSpec.from_config(cfg).terms(inf["pos"], inf["goal"], td["agents", "info", "pos"], zones)
    out = {k: v.amin(dim=-1).cpu().numpy() for k, v in terms.items()}   # worst agent -> [E, T]
    if cfg.N_ZONES:
        inside = inf["mu3"].reshape(*inf["mu3"].shape[:3]) < 0           # [E, T, A]
        out["in_zone"] = inside.float().mean(-1).cpu().numpy()          # fraction of agents inside, [E, T]
        out["zone_entry"] = inside.any(-1).any(-1).float().cpu().numpy()   # [E]
        out["catastrophes"] = inf["catastrophe"].reshape(*inside.shape).sum((1, 2)).cpu().numpy()   # [E]

    dg = torch.linalg.vector_norm(inf["pos"] - inf["goal"], dim=-1)      # progress per step while not at the goal
    step = (dg[:, :-1] - dg[:, 1:])[dg[:, :-1] > cfg.GOAL_EPS]
    out["speed"] = step.cpu().numpy()
    out["at_goal_end"] = (dg[:, -1] <= cfg.GOAL_EPS).all(dim=-1).float().cpu().numpy()   # [E]
    # Steps to goal: first step an agent is within GOAL_EPS (per agent), and first step all agents are (per episode);
    # NaN when never reached within MAX_STEPS
    at = dg <= cfg.GOAL_EPS                                              # [E, T, A]
    first = torch.where(at.any(1), at.float().argmax(1) + 1.0, torch.nan)                       # [E, A]
    all_at = at.all(-1)                                                  # [E, T]
    out["t_goal_agent"] = first.flatten().cpu().numpy()                 # [E * A]
    out["t_goal_all"] = torch.where(all_at.any(1), all_at.float().argmax(1) + 1.0, torch.nan).cpu().numpy()   # [E]
    return option_label(cfg), cfg, out


def add_run_args(p):
    """--results + --run-id (finds the traces folders), or --traces directly."""
    p.add_argument("--results", help="RESULTS_DIR of the run, e.g. results/navigation")
    p.add_argument("--run-id", help="run id")
    p.add_argument("--traces", nargs="+", help="or: the traces/<run_id> folder(s) themselves")
    p.add_argument("--out", help="output folder (default: <results>/comparison/<run_id>, or the first --traces)")


def run_folders(p, a):
    """-> (traces folders, output folder) from the arguments of add_run_args."""
    if a.traces:
        dirs, out = [Path(t) for t in a.traces], Path(a.traces[0])
    elif a.results and a.run_id:
        dirs, out = find_traces(a.results, a.run_id), comparison_dir(a.results, a.run_id)
    else:
        p.error("give --results with --run-id, or --traces")
    if not dirs:
        raise SystemExit(f"[{p.prog}] no traces folders found for {a.run_id} under {a.results}")
    return dirs, Path(a.out) if a.out else out


def main():
    p = argparse.ArgumentParser(prog="analyze")
    add_run_args(p)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=12345, help="env seed, the same for every model")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    trace_dirs, out_dir = run_folders(p, a)

    res, T = {}, 0
    for js in sorted(js for d in trace_dirs for js in d.glob("*.json")):
        ckpt = agents_dir(js.parent) / js.with_suffix(".pt").name
        if not ckpt.exists():
            print(f"[analyze] no checkpoint for {js.name}, skipped")
            continue
        label, cfg, out = rollout_terms(json.load(open(js))["config"], ckpt, a.episodes, a.seed, a.device)
        T = cfg.MAX_STEPS
        r = res.setdefault(label, {k: [] for k in out})
        for k, v in out.items():
            r[k].append(v)
        print(f"[analyze] {js.stem}: mean total rho {out['total'].mean():+.4f}")
    if not res:
        raise SystemExit("[analyze] nothing to analyse")

    labels = sorted(res, key=lambda l: OPTION_ORDER.index(l) if l in OPTION_ORDER else 99)
    zones = "zone_entry" in next(iter(res.values()))
    terms = ZONE_TERMS if zones else TERMS
    fig, axes = plt.subplots(2, 3 if zones else 2, figsize=(17 if zones else 12, 8), sharex=True)
    t = np.arange(1, T + 1)
    rows = []
    for label in labels:
        r = {k: np.concatenate(v) for k, v in res[label].items()}       # seeds pooled: [seeds * episodes, T]
        n_seeds = len(res[label]["total"])
        for ax, (key, name) in zip(axes.flat, terms):
            m, s = r[key].mean(0), r[key].std(0)
            ax.plot(t, m, color=option_color(label), lw=2, label=f"{label} ({n_seeds} seeds)")
            ax.fill_between(t, m - s, m + s, color=option_color(label), alpha=0.15, lw=0)
            ax.set_title(name)
        rows.append({"option": label, "seeds": n_seeds, "episodes": r["total"].shape[0],
                     **{f"{k}_mean": round(float(r[k].mean()), 4) for k, _ in terms},
                     "frac_steps_rho>=0": round(float((r["total"] >= 0).mean()), 4),
                     "frac_episodes_rho>=0_all_steps": round(float((r["total"] >= 0).all(1).mean()), 4),
                     "all_at_goal_at_end": round(float(r["at_goal_end"].mean()), 4),
                     "progress_per_step": round(float(r["speed"].mean()), 4),
                     **({"zone_entry_rate": round(float(r["zone_entry"].mean()), 4),
                         "catastrophes_per_episode": round(float(r["catastrophes"].mean()), 4)} if zones else {})})
    for ax in axes.flat:
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.grid(True, alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Time step")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"IPPO | gated spec terms over time | worst agent per episode | mean ± 1 std over "
                 f"{a.episodes} episodes × seeds (same envs)", fontsize=10)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "analysis.png", dpi=150)
    plt.close(fig)

    with open(out_dir / "analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        print("  ".join(f"{k}={v}" for k, v in row.items()))
    print(f"[analyze] wrote {out_dir / 'analysis.png'} and analysis.csv")


if __name__ == "__main__":
    main()
