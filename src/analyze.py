"""Figures and a summary table from the runs written by train.py. Edit the lists below to choose what is shown;
nothing needs to be retrained.

Run from the repository root:  uv run python src/analyze.py
Reads results/<reward>_seed<seed>.csv and _traces.npz, writes results/<figure>.png
"""
import csv
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# =============================== PARAMETERS ===============================
RESULTS_DIR = "results"
FIGURES = ["table", "curves", "final", "tradeoff", "robustness_distribution", "trajectories"]
CURVES = ["native_return", "robustness_team", "success_rate", "violation_rate", "collision_steps", "min_separation"]
FINAL = ["native_return", "robustness_team", "spec_satisfied", "success_rate", "violation_rate", "time_to_goal"]
TRADEOFF = ("violation_rate", "success_rate")  # x and y of the one-point-per-run scatter
EXAMPLE_SEED, EXAMPLE_EPISODE = 0, 0            # which evaluation episode the trajectory plot shows
# Available metrics: train_return, native_return, robustness_team, robustness_agent, robustness_goal,
# robustness_separation, spec_satisfied, success_rate, violation_rate, collision_steps, min_separation, time_to_goal
# ==========================================================================


def load_runs():
    """reward condition -> one dict per seed, mapping each metric to its values over the evaluations."""
    runs = defaultdict(list)
    for path in sorted(glob.glob(f"{RESULTS_DIR}/*_seed*.csv")):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        runs[rows[0]["reward"]].append({key: np.array([float(row[key]) for row in rows]) for key in rows[0]
                                        if key != "reward"})
    return runs


def load_traces(condition):
    """The final evaluation of every seed of a condition (per step: mu1, mu2, pos, goal, rewards, collisions)."""
    return [np.load(path) for path in sorted(glob.glob(f"{RESULTS_DIR}/{condition}_seed*_traces.npz"))]


def grid(n):
    rows, cols = (n + 2) // 3, min(n, 3)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, axes.flat


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/{name}.png", dpi=150)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR}/{name}.png")


def table(runs):
    """Final evaluation, mean ± std over seeds."""
    print(f"{'metric':24s}" + "".join(f"{condition:>22s}" for condition in runs))
    for metric in FINAL:
        cells = [np.array([seed[metric][-1] for seed in seeds]) for seeds in runs.values()]
        print(f"{metric:24s}" + "".join(f"{v.mean():>14.3f} ± {v.std():<5.3f}" for v in cells))


def curves(runs):
    fig, axes = grid(len(CURVES))
    for ax, metric in zip(axes, CURVES):
        for condition, seeds in runs.items():
            n = min(len(seed["frames"]) for seed in seeds)  # cut to the shortest run, so unfinished runs also work
            values = np.stack([seed[metric][:n] for seed in seeds])
            frames, mean, std = seeds[0]["frames"][:n], values.mean(0), values.std(0)
            ax.plot(frames, mean, label=f"{condition} ({len(seeds)} seeds)")
            ax.fill_between(frames, mean - std, mean + std, alpha=0.2)
        ax.set(title=metric, xlabel="environment frames")
    axes[0].legend()
    save(fig, "curves")


def final(runs):
    fig, axes = grid(len(FINAL))
    for ax, metric in zip(axes, FINAL):
        for k, seeds in enumerate(runs.values()):
            values = [seed[metric][-1] for seed in seeds]
            ax.bar(k, np.mean(values), yerr=np.std(values), color=f"C{k}", alpha=0.5)
            ax.scatter([k] * len(values), values, color=f"C{k}", zorder=3)
        ax.set_xticks(range(len(runs)), list(runs))
        ax.set_title(metric)
    save(fig, "final")


def tradeoff(runs):
    x, y = TRADEOFF
    fig, ax = plt.subplots(figsize=(5, 4))
    for k, (condition, seeds) in enumerate(runs.items()):
        ax.scatter([seed[x][-1] for seed in seeds], [seed[y][-1] for seed in seeds], color=f"C{k}", label=condition)
    ax.set(xlabel=x, ylabel=y, title="Final evaluation, one point per run")
    ax.legend()
    save(fig, "tradeoff")


def robustness_distribution(runs):
    """Per-agent SFO robustness over all final-evaluation episodes; right of the dashed line = spec satisfied."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, condition in enumerate(runs):
        rho = [np.minimum(t["mu1"][:, -1], t["mu2"].min(axis=1)).ravel() for t in load_traces(condition)]
        ax.hist(np.concatenate(rho), bins=50, alpha=0.5, color=f"C{k}", label=condition)
    ax.axvline(0, color="black", linestyle="--")
    ax.set(xlabel="per-agent SFO robustness", ylabel="agent-episodes")
    ax.legend()
    save(fig, "robustness_distribution")


def trajectories(runs):
    """One evaluation episode per condition: paths from start (o) to goal (x), red where an agent is closer than d."""
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 5), squeeze=False)
    for ax, condition in zip(axes.flat, runs):
        t = np.load(f"{RESULTS_DIR}/{condition}_seed{EXAMPLE_SEED}_traces.npz")
        pos, goal, mu2 = t["pos"][EXAMPLE_EPISODE], t["goal"][EXAMPLE_EPISODE], t["mu2"][EXAMPLE_EPISODE]
        for agent in range(pos.shape[1]):
            ax.plot(pos[:, agent, 0], pos[:, agent, 1], color=f"C{agent}")
            ax.scatter(*pos[0, agent], marker="o", color=f"C{agent}")
            ax.scatter(*goal[0, agent], marker="x", s=80, color=f"C{agent}")
        ax.scatter(*pos[mu2 < 0].T, s=10, color="red", zorder=3, label="closer than d")
        ax.set(title=f"{condition}, seed {EXAMPLE_SEED}, episode {EXAMPLE_EPISODE}", aspect="equal")
        ax.legend()
    save(fig, "trajectories")


if __name__ == "__main__":
    runs = load_runs()
    for figure in FIGURES:
        globals()[figure](runs)
