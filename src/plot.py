"""Combine the per-run CSVs written by train.py into one figure: results/compare.png.

Run from the repository root:  uv run python src/plot.py
Each curve is the mean ± std over the seeds of an algorithm (cut to the shortest run, so unfinished runs also work).
"""
import csv
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import torch

RESULTS_DIR = "results"

runs = defaultdict(list)  # algorithm -> one list of (frames, return, success, fell) rows per seed
for path in sorted(glob.glob(f"{RESULTS_DIR}/*_seed*.csv")):
    with open(path) as f:
        rows = list(csv.reader(f))[1:]
    runs[rows[0][0]].append([[float(value) for value in row[2:]] for row in rows])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for algo, seeds in runs.items():
    n = min(len(seed) for seed in seeds)
    data = torch.tensor([seed[:n] for seed in seeds])  # [seeds, evaluations, 4]
    for ax, column in zip(axes, (1, 2)):
        mean, std = data[..., column].mean(0), data[..., column].std(0, correction=0)
        ax.plot(data[0, :, 0], mean, label=f"{algo} ({len(seeds)} seeds)")
        ax.fill_between(data[0, :, 0], mean - std, mean + std, alpha=0.2)
for ax, label in zip(axes, ("evaluation team return", "success rate")):
    ax.set_xlabel("environment frames")
    ax.set_ylabel(label)
axes[0].legend()
fig.savefig(f"{RESULTS_DIR}/compare.png", dpi=150, bbox_inches="tight")
print(f"Saved {RESULTS_DIR}/compare.png")
