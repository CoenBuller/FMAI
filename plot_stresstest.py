import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_stresstest_results(
    root_dir: str = "results/stresstest",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads all stress-test JSONs and returns (raw_df, aggregated_df)."""
    pattern = os.path.join(root_dir, "seed_*", "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No JSON files found matching {pattern}")

    rows = []
    for path in files:
        fname = os.path.basename(path)
        # Check 'nowindtrained' first because 'windtrained' is a substring of it
        wind_trained = "nowindtrained" not in fname

        with open(path, "r") as f:
            data = json.load(f)

        cfg = data["config"]
        info = data["history"][0]

        row = {
            "seed": cfg["SEED"],
            "mode": cfg["REWARD_MODE"],
            "wind_trained": wind_trained,
            "n_agents": cfg["NUM_AGENTS"],
            "wind_strength": cfg["WIND_STRENGTH"],
            "wind_gust": cfg["WIND_GUST_STRENGTH"],
            "pass_rate": info["pass_rate"],
            "return_mean": info["return_mean"],
            "rho_mean": info["rho_mean"],
            "violation_rate": info["violation_rate"],
        }
        # Also pull individual sub-formula failure rates if present
        for k, v in info.get("sub_fail_rate", {}).items():
            row[f"fail_{k}"] = v
        rows.append(row)

    raw_df = pd.DataFrame(rows)

    group_cols = ["mode", "wind_trained", "n_agents", "wind_strength", "wind_gust"]
    metric_cols = [c for c in raw_df.columns if c not in group_cols + ["seed"]]

    agg_df = raw_df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    agg_df.columns = [
        f"{c[0]}_{c[1]}" if c[1] else c[0] for c in agg_df.columns
    ]
    return raw_df, agg_df


def plot_stresstest_matrices(
    agg_df: pd.DataFrame,
    metric: str = "pass_rate",
    title: str = "Stress Test: Pass Rate (Mean ± Std across 5 Seeds)",
    save_path: str = "stresstest_matrix.png",
):
    """3x4 horizontal grid of (wind_strength x wind_gust) heatmaps:
    Rows = n_agents, Cols = (mode, wind_trained).
    """
    col_specs = [
        ("none", False, "Native\n(No-Wind Trained)"),
        ("rho", False, "Rho\n(No-Wind Trained)"),
        ("none", True, "Native\n(Wind Trained)"),
        ("rho", True, "Rho\n(Wind Trained)"),
    ]
    n_agents_list = sorted(agg_df["n_agents"].unique())
    winds = sorted(agg_df["wind_strength"].unique())
    gusts = sorted(agg_df["wind_gust"].unique())

    mean_col, std_col = f"{metric}_mean", f"{metric}_std"
    if metric == "pass_rate":
        vmin, vmax, cmap, fmt = 0.0, 1.0, "YlGn", ".2f"
    else:
        vmin = agg_df[mean_col].min()
        vmax = agg_df[mean_col].max()
        cmap, fmt = "viridis", ".1f"

    n_rows, n_cols = len(n_agents_list), len(col_specs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols + 1.5, 3.4 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )

    im = None
    for r, n_ag in enumerate(n_agents_list):
        for c, (mode, wt, col_label) in enumerate(col_specs):
            ax = axes[r, c]
            sub = agg_df[
                (agg_df["mode"] == mode)
                & (agg_df["wind_trained"] == wt)
                & (agg_df["n_agents"] == n_ag)
            ]

            mean_mat = sub.pivot(
                index="wind_strength", columns="wind_gust", values=mean_col
            ).reindex(index=winds, columns=gusts)
            std_mat = sub.pivot(
                index="wind_strength", columns="wind_gust", values=std_col
            ).reindex(index=winds, columns=gusts)

            im = ax.imshow(
                mean_mat.values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto"
            )

            for i in range(len(winds)):
                for j in range(len(gusts)):
                    m_val = mean_mat.values[i, j]
                    s_val = std_mat.values[i, j]
                    if np.isnan(m_val):
                        continue
                    norm_val = (m_val - vmin) / max(vmax - vmin, 1e-8)
                    txt_color = "white" if norm_val > 0.6 else "black"
                    s_str = f"\n±{s_val:{fmt}}" if not np.isnan(s_val) else ""
                    ax.text(
                        j,
                        i,
                        f"{m_val:{fmt}}{s_str}",
                        ha="center",
                        va="center",
                        color=txt_color,
                        fontsize=9,
                        fontweight="bold",
                    )

            if r == 0:
                ax.set_title(col_label, fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(
                    f"N = {n_ag} Agents\n\nWind Strength",
                    fontsize=11,
                    fontweight="bold",
                )
            if r == n_rows - 1:
                ax.set_xlabel("Wind Gust Strength", fontsize=11, fontweight="bold")

            ax.set_xticks(range(len(gusts)))
            ax.set_xticklabels([f"{g:.2f}" for g in gusts])
            ax.set_yticks(range(len(winds)))
            ax.set_yticklabels([f"{w:.2f}" for w in winds])

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_delta_matrices(
    agg_df: pd.DataFrame,
    metric: str = "pass_rate",
    save_path: str = "stresstest_delta.png",
):
    """2x3 diverging heatmap showing (Rho - Native) advantage."""
    row_specs = [
        (False, "No-Wind Trained\n(Rho − Native)"),
        (True, "Wind Trained\n(Rho − Native)"),
    ]
    n_agents_list = sorted(agg_df["n_agents"].unique())
    winds = sorted(agg_df["wind_strength"].unique())
    gusts = sorted(agg_df["wind_gust"].unique())
    mean_col = f"{metric}_mean"

    deltas = {}
    max_abs = 1e-6
    for wt, _ in row_specs:
        for n_ag in n_agents_list:
            sub_rho = agg_df[
                (agg_df["mode"] == "rho")
                & (agg_df["wind_trained"] == wt)
                & (agg_df["n_agents"] == n_ag)
            ]
            sub_nat = agg_df[
                (agg_df["mode"] == "none")
                & (agg_df["wind_trained"] == wt)
                & (agg_df["n_agents"] == n_ag)
            ]
            m_rho = sub_rho.pivot(
                index="wind_strength", columns="wind_gust", values=mean_col
            ).reindex(index=winds, columns=gusts)
            m_nat = sub_nat.pivot(
                index="wind_strength", columns="wind_gust", values=mean_col
            ).reindex(index=winds, columns=gusts)
            diff = m_rho - m_nat
            deltas[(wt, n_ag)] = diff
            if not diff.isna().all().all():
                max_abs = max(max_abs, np.nanmax(np.abs(diff.values)))

    fig, axes = plt.subplots(
        len(row_specs),
        len(n_agents_list),
        figsize=(4.5 * len(n_agents_list) + 1.5, 3.8 * len(row_specs)),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )

    fmt = "+.2f" if metric == "pass_rate" else "+.1f"
    im = None
    for r, (wt, row_label) in enumerate(row_specs):
        for c, n_ag in enumerate(n_agents_list):
            ax = axes[r, c]
            diff = deltas[(wt, n_ag)].values
            im = ax.imshow(
                diff, vmin=-max_abs, vmax=max_abs, cmap="RdYlGn", aspect="auto"
            )

            for i in range(len(winds)):
                for j in range(len(gusts)):
                    val = diff[i, j]
                    if np.isnan(val):
                        continue
                    txt_color = "white" if abs(val) > 0.65 * max_abs else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:{fmt}}",
                        ha="center",
                        va="center",
                        color=txt_color,
                        fontsize=10,
                        fontweight="bold",
                    )

            if r == 0:
                ax.set_title(f"N = {n_ag} Agents", fontsize=13, fontweight="bold")
            if c == 0:
                ax.set_ylabel(
                    f"{row_label}\n\nWind Strength", fontsize=11, fontweight="bold"
                )
            if r == len(row_specs) - 1:
                ax.set_xlabel("Wind Gust Strength", fontsize=11, fontweight="bold")

            ax.set_xticks(range(len(gusts)))
            ax.set_xticklabels([f"{g:.2f}" for g in gusts])
            ax.set_yticks(range(len(winds)))
            ax.set_yticklabels([f"{w:.2f}" for w in winds])

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(
        f"Δ {metric.replace('_', ' ').title()} (Rho − Native)",
        fontsize=12,
        fontweight="bold",
    )
    fig.suptitle(
        f"Advantage of Robustness Training: Δ {metric.replace('_', ' ').title()}",
        fontsize=15,
        fontweight="bold",
    )
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def print_compact_summary(raw_df: pd.DataFrame):
    """Compact table collapsing across all wind/gust levels to show high-level trends."""
    summary = (
        raw_df.groupby(["wind_trained", "n_agents", "mode"])[
            ["pass_rate", "return_mean", "rho_mean"]
        ]
        .agg(["mean", "std"])
    )
    print("\n=== High-Level Summary (Averaged Across All Wind & Gust Conditions) ===")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    root_dir = "results/images"
    raw_df, agg_df = load_stresstest_results("results/stresstest")
    print_compact_summary(raw_df)
    os.makedirs(root_dir, exist_ok=True)

    # 1. Full 3x4 horizontal matrices
    plot_stresstest_matrices(
        agg_df,
        metric="pass_rate",
        title="Pass Rate Across Wind, Gusts & Swarm Size",
        save_path=os.path.join(root_dir, "matrix_pass_rate.png"),
    )
    plot_stresstest_matrices(
        agg_df,
        metric="return_mean",
        title="Native Return Across Wind, Gusts & Swarm Size",
        save_path=os.path.join(root_dir, "matrix_return.png"),
    )

    # 2. Direct Rho - Native comparison matrices
    plot_delta_matrices(
        agg_df,
        metric="pass_rate",
        save_path=os.path.join(root_dir, "delta_pass_rate.png"),
    )
    plot_delta_matrices(
        agg_df,
        metric="return_mean",
        save_path=os.path.join(root_dir, "delta_return.png"),
    )