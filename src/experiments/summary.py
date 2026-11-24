"""Summary of one run: every reward option (native, robustness) in one figure and one table.

    PYTHONPATH=src python -m experiments.summary --results results/navigation --run-id <run_id>

Finds the run like analyze.py (--results + --run-id, or --traces). Writes summary.png, summary.csv and summary.md
into <results>/comparison/<run_id> (or --out).

- summary.png: learning curves (eval metrics vs env steps), colour = reward option, mean +/- std over seeds,
  each seed smoothed first (--smooth).
- summary.csv / summary.md: per option
  * final evaluation (mean +/- std over seeds): native return, spec-independent robustness rho, pass rate, success,
    safety violations, collisions, zone entries, catastrophes;
  * rollouts of the final checkpoints on the same --episodes envs (as analyze.py): steps until an agent reaches its
    goal and until all agents have (mean +/- std over the agents/episodes that got there, plus the fraction that
    got there), and the spec robustness per step (worst agent; mean +/- std over all steps of all episodes and seeds).
The checkpoints are taken from the matching agents/ folder (traces/<id> -> agents/<id>); --no-rollouts skips that part.
"""
import argparse, dataclasses, json, warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from fmai.utils import option_label, option_color, smooth, OPTION_ORDER, agents_dir
from experiments.analyze import rollout_terms, add_run_args, run_folders

CURVES = [("return_mean", "Eval return (native)"), ("rho_mean", "Eval robustness ρ (spec-independent)"),
          ("pass_rate", "Pass rate (ρ ≥ 0)"), ("success_rate", "Success (all agents at goal)"),
          ("violation_rate", "Safety violation rate"), ("zone_entry_rate", "Zone entry rate")]
FINAL = [("return_mean", "return"), ("rho_mean", "rho"), ("pass_rate", "pass"), ("success_rate", "success"),
         ("violation_rate", "violations"), ("collision_steps", "collisions"), ("zone_entry_rate", "zone_entries"),
         ("catastrophes", "catastrophes")]

def load(trace_dirs):
    """-> {option: [(history, cfg_dict, json path), ...one per seed]}"""
    fields = {f.name for f in dataclasses.fields(Config)}
    runs = {}
    for d in trace_dirs:
        for js in sorted(Path(d).glob("*.json")):
            data = json.load(open(js))
            cfg = Config(**{k: v for k, v in data["config"].items() if k in fields})
            runs.setdefault(option_label(cfg), []).append((data["history"], data["config"], js))
    order = lambda k: OPTION_ORDER.index(k) if k in OPTION_ORDER else 99
    return {k: runs[k] for k in sorted(runs, key=order)}


def ms(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return (float(x.mean()), float(x.std())) if x.size else (float("nan"), float("nan"))


def main():
    p = argparse.ArgumentParser(prog="summary")
    add_run_args(p)
    p.add_argument("--smooth", type=int, default=5)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--no-rollouts", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    trace_dirs, out = run_folders(p, a)
    runs = load(trace_dirs)
    if not runs:
        raise SystemExit("[summary] no runs found")

    # ------------------------------ learning curves ------------------------------
    curves = [c for c in CURVES if c[0] in next(iter(runs.values()))[0][0][-1]]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for label, seeds in runs.items():
        hists = [h for h, _, _ in seeds]
        n = min(len(h) for h in hists)
        x = np.array([e["step"] for e in hists[0][:n]])
        for ax, (key, name) in zip(axes.flat, curves):
            ys = np.stack([smooth([e.get(key, np.nan) for e in h[:n]], a.smooth) for h in hists])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m, s = np.nanmean(ys, 0), np.nanstd(ys, 0)
            ax.plot(x, m, color=option_color(label), lw=2, label=f"{label} ({len(hists)} seeds)")
            ax.fill_between(x, m - s, m + s, color=option_color(label), alpha=0.12, lw=0)
            ax.set(title=name, xlabel="Env steps")
    for ax in axes.flat[len(curves):]:
        ax.axis("off")
    for ax in axes.flat[:len(curves)]:
        ax.grid(True, alpha=0.3)
    axes.flat[1].axhline(0, color="gray", ls=":", lw=1)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"IPPO | native vs robustness reward | mean ± std over seeds, smoothing window {a.smooth}", fontsize=12)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "summary.png", dpi=150)
    plt.close(fig)

    # ------------------------------ table ------------------------------
    rows = []
    for label, seeds in runs.items():
        row = {"option": label, "seeds": len(seeds)}
        for key, name in FINAL:
            if key in seeds[0][0][-1]:
                row[f"{name}_mean"], row[f"{name}_std"] = ms([h[-1][key] for h, _, _ in seeds])
        if not a.no_rollouts:
            r = {}
            for _, cfg_dict, js in seeds:
                ckpt = agents_dir(js.parent) / js.with_suffix(".pt").name
                if not ckpt.exists():
                    print(f"[summary] no checkpoint {ckpt}, skipped")
                    continue
                _, _, o = rollout_terms(cfg_dict, ckpt, a.episodes, a.seed, a.device)
                for k in ["t_goal_agent", "t_goal_all", "total"]:
                    r.setdefault(k, []).append(o[k].ravel())
            if r:
                r = {k: np.concatenate(v) for k, v in r.items()}
                row["steps_to_goal_agent_mean"], row["steps_to_goal_agent_std"] = ms(r["t_goal_agent"])
                row["frac_agents_reached"] = float((~np.isnan(r["t_goal_agent"])).mean())
                row["steps_all_at_goal_mean"], row["steps_all_at_goal_std"] = ms(r["t_goal_all"])
                row["frac_episodes_all_reached"] = float((~np.isnan(r["t_goal_all"])).mean())
                row["spec_rho_per_step_mean"], row["spec_rho_per_step_std"] = ms(r["total"])
                row["frac_steps_spec_rho>=0"] = float((r["total"] >= 0).mean())
        rows.append(row)
        print(f"[summary] {label} done")

    cols = list(dict.fromkeys(k for row in rows for k in row))
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    lines = [",".join(cols)] + [",".join(fmt(row.get(c, "")) for c in cols) for row in rows]
    (out / "summary.csv").write_text("\n".join(lines) + "\n")

    # Markdown: "mean ± std" per metric
    names = [c[:-5] for c in cols if c.endswith("_mean")]
    extra = [c for c in cols if c.startswith("frac_")]
    hdr = ["option", "seeds"] + names + extra
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for row in rows:
        cells = [row["option"], str(row["seeds"])]
        cells += [f"{row[n + '_mean']:.3f} ± {row[n + '_std']:.3f}" if n + "_mean" in row else "" for n in names]
        cells += [f"{row[c]:.3f}" if c in row else "" for c in extra]
        md.append("| " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"[summary] wrote {out}/summary.png, .csv and .md")


if __name__ == "__main__":
    main()
