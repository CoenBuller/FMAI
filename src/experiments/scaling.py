"""Zero-shot scaling test: every checkpoint of a run, trained with NUM_AGENTS agents, is rolled out without any
retraining on environments with other numbers of agents (--n-agents). This works because the policy shares its
parameters over the agents and an agent's observation does not depend on how many agents there are. Every model
sees the same environments per number of agents (fixed seed, deterministic policy, full MAX_STEPS); the metrics are
the ones of the normal evaluation (train.evaluate, spec-independent robustness), as mean +/- std over seeds.

    PYTHONPATH=src python -m experiments.scaling --results results/navigation --run-id <run_id> --n-agents 2 4 6 8

Finds the run like analyze.py (--results + --run-id, or --traces). Writes scaling.png and scaling.csv into
<results>/comparison/<run_id> (or --out).
"""
import argparse, csv, dataclasses, json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from environment.environment import make_env
from fmai.agents import Agent
from fmai.train import evaluate
from fmai.sfo import build_spec
from fmai.utils import option_label, option_color, OPTION_ORDER, agents_dir
from experiments.analyze import add_run_args, run_folders

METRICS = [("success_rate", "Success (all agents at goal at the end)"), ("pass_rate", "Pass rate (ρ ≥ 0)"),
           ("agent_pass_rate", "Per-agent pass rate"),
           ("return_mean", "Eval return (native)"), ("violation_rate", "Safety violation rate"),
           ("collision_steps", "Collisions per episode"), ("zone_entry_rate", "Zone entry rate")]


def zero_shot(cfg_dict, ckpt, n_agents, episodes, seed, device):
    """Roll out the policy in `ckpt` on `episodes` envs with `n_agents` agents -> (cfg, evaluate() dict)."""
    fields = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in fields})
    cfg.ACTIVATION = {"ReLu": "ReLU"}.get(cfg.ACTIVATION, cfg.ACTIVATION)   # runs from before the yaml fix
    trained = cfg.NUM_AGENTS
    cfg.NUM_AGENTS, cfg.DEVICE, cfg.EVAL_EPISODES = n_agents, device, episodes
    env = make_env(cfg, num_envs=episodes, seed=seed, spec=None, mode="none")
    agent = Agent()
    agent.make_agent(env=env, cfg=cfg)                                   # the critic is built for n_agents but unused
    agent.policy.load_state_dict(torch.load(ckpt, map_location=device)["policy"])
    cfg.NUM_AGENTS = trained                                             # keep the label/title on the trained setting
    spec = build_spec(cfg)[0] if cfg.SCENARIO == "flocking" else None     # as in training: flocking spec, else sfo_rho
    return cfg, evaluate(env, agent, spec, cfg, training_episode=0)


def main():
    p = argparse.ArgumentParser(prog="scaling")
    add_run_args(p)
    p.add_argument("--n-agents", type=int, nargs="+", required=True, help="numbers of agents to test on")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=12345, help="env seed, the same for every model")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    trace_dirs, out_dir = run_folders(p, a)

    res, trained = {}, set()                                   # res[label][n] = [info per seed]
    for js in sorted(js for d in trace_dirs for js in d.glob("*.json")):
        ckpt = agents_dir(js.parent) / js.with_suffix(".pt").name
        if not ckpt.exists():
            print(f"[scaling] no checkpoint for {js.name}, skipped")
            continue
        cfg_dict = json.load(open(js))["config"]
        for n in a.n_agents:
            cfg, info = zero_shot(cfg_dict, ckpt, n, a.episodes, a.seed, a.device)
            res.setdefault(option_label(cfg), {}).setdefault(n, []).append(info)
            print(f"[scaling] {js.stem} (trained on {cfg.NUM_AGENTS}) -> {n} agents: "
                  f"success {info['success_rate']:.2f}, pass {info['pass_rate']:.2f}, return {info['return_mean']:.2f}")
        trained.add(cfg.NUM_AGENTS)
    if not res:
        raise SystemExit("[scaling] nothing to test")

    labels = sorted(res, key=lambda l: OPTION_ORDER.index(l) if l in OPTION_ORDER else 99)
    metrics = [(k, name) for k, name in METRICS if k in next(iter(next(iter(res.values())).values()))[0]]
    fig, axes = plt.subplots(2, 4, figsize=(22, 8), sharex=True)
    ns = np.array(a.n_agents)
    rows = []
    for label in labels:
        for ax, (key, name) in zip(axes.flat, metrics):
            v = np.array([[i[key] for i in res[label][n]] for n in a.n_agents])   # [len(ns), seeds]
            m, s = v.mean(1), v.std(1)
            ax.plot(ns, m, "o-", color=option_color(label), lw=2, label=f"{label} ({v.shape[1]} seeds)")
            ax.fill_between(ns, m - s, m + s, color=option_color(label), alpha=0.15, lw=0)
            ax.set_title(name)
        for n in a.n_agents:
            infos = res[label][n]
            rows.append({"option": label, "trained_agents": "/".join(map(str, sorted(trained))), "test_agents": n,
                         "seeds": len(infos),
                         **{f"{k}_{st}": round(float(f([i[k] for i in infos])), 4)
                            for k, _ in metrics for st, f in (("mean", np.mean), ("std", np.std))}})
    for ax in axes.flat[:len(metrics)]:
        for t in trained:
            ax.axvline(t, color="gray", ls="--", lw=1)                   # the number of agents it was trained on
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ns)
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Number of agents at test time")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"IPPO | zero-shot scaling (trained on {', '.join(map(str, sorted(trained)))} agents, dashed) | "
                 f"mean ± std over seeds | {a.episodes} episodes per point (same envs)", fontsize=10)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "scaling.png", dpi=150)
    plt.close(fig)

    with open(out_dir / "scaling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[scaling] wrote {out_dir / 'scaling.png'} and scaling.csv")


if __name__ == "__main__":
    main()
