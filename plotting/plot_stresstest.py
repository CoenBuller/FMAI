import argparse
import glob
import itertools
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# Consistent styling for the 4 policy regimes across all plots
REGIME_STYLES = {
    ("none", False): {"label": "Native (No-Wind)", "color": "#2b5c8f", "ls": "--", "marker": "o"},
    ("rho", False):  {"label": "Rho (No-Wind)",    "color": "#1b9e77", "ls": "--", "marker": "s"},
    ("none", True):  {"label": "Native (Wind)",    "color": "#d95f02", "ls": "-",  "marker": "^"},
    ("rho", True):   {"label": "Rho (Wind)",       "color": "#b2182b", "ls": "-",  "marker": "D"},
}

PRETTY_SUBFORMULAS = {
    "fail_phi_c (collision)": r"$\varphi_c$ (Collision)",
    "fail_phi_vm (v_max)": r"$\varphi_{v_m}$ ($v_{\max}$)",
    "fail_phi_sd (dist_band)": r"$\varphi_{s_d}$ (Dist. Band)",
    "fail_phi_sv (tan_speed)": r"$\varphi_{s_v}$ (Tang. Speed)",
}


def load_stresstest_results(
    root_dir: str = "results/flocking/stresstest",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads all stress-test JSONs (including tau_flock) and returns (raw_df, agg_df)."""
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
            "tau_flock": cfg.get("TAU_FLOCK", 1),
            "wind_strength": cfg["WIND_STRENGTH"],
            "wind_gust": cfg["WIND_GUST_STRENGTH"],
            "pass_rate": info["pass_rate"],
            "return_mean": info["return_mean"],
            "rho_mean": info["rho_mean"],
            "violation_rate": info["violation_rate"],
        }
        for k, v in info.get("sub_fail_rate", {}).items():
            row[f"fail_{k}"] = v
        rows.append(row)

    raw_df = pd.DataFrame(rows)

    group_cols = ["mode", "wind_trained", "n_agents", "tau_flock", "wind_strength", "wind_gust"]
    metric_cols = [c for c in raw_df.columns if c not in group_cols + ["seed"]]

    agg_df = raw_df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    agg_df.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in agg_df.columns]
    return raw_df, agg_df


# =============================================================================
# 1. MAIN EFFECTS DASHBOARD (Swept factors x 4 policy regimes)
# =============================================================================
def plot_main_effects(
    raw_df: pd.DataFrame,
    metric: str,
    save_path: str,
    include_tau: bool = True,
):
    """Shows marginal main effects of (optional tau), n_agents, wind_strength, wind_gust."""
    factors = [
        ("n_agents", "Swarm Size (N Agents)"),
        ("wind_strength", "Wind Strength"),
        ("wind_gust", "Wind Gust Strength"),
    ]
    if include_tau:
        factors.insert(0, ("tau_flock", r"Recovery Horizon ($\tau$)"))

    fig, axes = plt.subplots(
        1, len(factors),
        figsize=(5.5 * len(factors), 5.2),
        sharey=True,
        constrained_layout=True,
    )

    for ax, (col, xlabel) in zip(axes, factors):
        x_vals = sorted(raw_df[col].unique())
        for (mode, wt), style in REGIME_STYLES.items():
            sub = raw_df[(raw_df["mode"] == mode) & (raw_df["wind_trained"] == wt)]
            # Average over all other factors per seed first so error bars reflect seed variance
            per_seed = sub.groupby([col, "seed"])[metric].mean().reset_index()
            stats = per_seed.groupby(col)[metric].agg(["mean", "std"]).reindex(x_vals)

            y = stats["mean"].values
            e = stats["std"].values
            ax.plot(
                x_vals, y,
                label=style["label"],
                color=style["color"],
                linestyle=style["ls"],
                marker=style["marker"],
                linewidth=2.2,
                markersize=7,
            )
            ax.fill_between(x_vals, y - e, y + e, color=style["color"], alpha=0.14)

        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
        ax.set_xticks(x_vals)
        ax.grid(True, alpha=0.35)
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel(metric.replace("_", " ").title(), fontsize=13, fontweight="bold")
    if metric == "pass_rate":
        axes[0].set_ylim(-0.02, 1.05)
    axes[-1].legend(loc="best", fontsize=10, frameon=True)
    fig.suptitle(
        f"Main Effects Analysis: {metric.replace('_', ' ').title()} (Marginalized Mean ± Seed Std)",
        fontsize=15,
        fontweight="bold",
    )
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# =============================================================================
# 2. FACTORIAL EFFECT-SIZE DECOMPOSITION (Exact Eta-Squared / % Variance Explained)
# =============================================================================
def plot_factorial_eta_squared(
    raw_df: pd.DataFrame,
    metric: str,
    save_path: str,
    include_tau: bool = True,
):
    """Computes exact balanced-factorial Sum of Squares (eta^2) for main effects & 2-way interactions."""
    factors = ["mode", "wind_trained", "n_agents", "wind_strength", "wind_gust"]
    if include_tau:
        factors.append("tau_flock")

    pretty = {
        "mode": "Reward Mode (Rho vs Native)",
        "wind_trained": "Wind Trained",
        "n_agents": "N Agents",
        "tau_flock": r"Recovery Horizon ($\tau$)",
        "wind_strength": "Wind Strength",
        "wind_gust": "Wind Gust",
    }

    y = raw_df[metric].values
    grand_mean = np.mean(y)
    ss_total = np.sum((y - grand_mean) ** 2)
    if ss_total <= 1e-12:
        return

    main_ss = {}
    for f in factors:
        grp = raw_df.groupby(f)[metric].agg(["count", "mean"])
        main_ss[f] = np.sum(grp["count"] * (grp["mean"] - grand_mean) ** 2)

    inter_ss = {}
    for f1, f2 in itertools.combinations(factors, 2):
        grp = raw_df.groupby([f1, f2])[metric].agg(["count", "mean"])
        ss_cell = np.sum(grp["count"] * (grp["mean"] - grand_mean) ** 2)
        inter_ss[(f1, f2)] = max(0.0, ss_cell - main_ss[f1] - main_ss[f2])

    records = [(pretty[f], 100.0 * ss / ss_total, "Main Effect") for f, ss in main_ss.items()]
    for (f1, f2), ss in inter_ss.items():
        records.append((f"{pretty[f1]} × {pretty[f2]}", 100.0 * ss / ss_total, "2-Way Interaction"))

    eff_df = pd.DataFrame(records, columns=["Term", "PctVar", "Type"])
    top_df = pd.concat([
        eff_df[eff_df["Type"] == "Main Effect"],
        eff_df[eff_df["Type"] == "2-Way Interaction"].nlargest(8, "PctVar"),
    ]).sort_values("PctVar", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    colors = ["#2b5c8f" if t == "Main Effect" else "#d95f02" for t in top_df["Type"]]
    bars = ax.barh(top_df["Term"], top_df["PctVar"], color=colors, edgecolor="black", alpha=0.88)

    for bar, val in zip(bars, top_df["PctVar"]):
        ax.text(
            val + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_xlabel(r"Variance Explained (% of Total SS, $\eta^2 \times 100$)", fontsize=12, fontweight="bold")
    ax.set_title(f"Factorial Variance Decomposition: {metric.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.set_xlim(0, top_df["PctVar"].max() * 1.15)
    ax.grid(True, axis="x", alpha=0.35)
    ax.legend(
        handles=[
            Patch(facecolor="#2b5c8f", edgecolor="black", label="Main Effect"),
            Patch(facecolor="#d95f02", edgecolor="black", label="2-Way Interaction"),
        ],
        loc="lower right",
        frameon=True,
        fontsize=10,
    )
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# =============================================================================
# 3. SUB-FORMULA VIOLATION BREAKDOWNS (vs. Tau & vs. Swarm Size)
# =============================================================================
def _plot_subformulas_vs_factor(
    raw_df: pd.DataFrame,
    factor: str,
    xlabel: str,
    suptitle: str,
    save_path: str,
):
    fail_cols = [c for c in raw_df.columns if c.startswith("fail_")]
    if not fail_cols:
        return

    x_vals = sorted(raw_df[factor].unique())
    fig, axes = plt.subplots(
        1, len(fail_cols),
        figsize=(5.2 * len(fail_cols), 4.8),
        sharey=True,
        constrained_layout=True,
    )
    if len(fail_cols) == 1:
        axes = [axes]

    for ax, fcol in zip(axes, fail_cols):
        for (mode, wt), style in REGIME_STYLES.items():
            sub = raw_df[(raw_df["mode"] == mode) & (raw_df["wind_trained"] == wt)]
            per_seed = sub.groupby([factor, "seed"])[fcol].mean().reset_index()
            stats = per_seed.groupby(factor)[fcol].agg(["mean", "std"]).reindex(x_vals)

            y, e = stats["mean"].values, stats["std"].values
            ax.plot(
                x_vals, y,
                label=style["label"],
                color=style["color"],
                ls=style["ls"],
                marker=style["marker"],
                lw=2.2,
                markersize=7,
            )
            ax.fill_between(
                x_vals,
                np.clip(y - e, 0.0, 1.0),
                np.clip(y + e, 0.0, 1.0),
                color=style["color"],
                alpha=0.14,
            )

        ax.set_title(PRETTY_SUBFORMULAS.get(fcol, fcol.replace("fail_", "")), fontsize=12.5, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
        ax.set_xticks(x_vals)
        ax.grid(True, alpha=0.35)

    axes[0].set_ylabel("Sub-Formula Failure Rate [0, 1]", fontsize=12, fontweight="bold")
    axes[0].set_ylim(-0.02, 1.05)
    axes[-1].legend(loc="best", fontsize=9.5, frameon=True)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_tau_subformulas(raw_df: pd.DataFrame, save_path: str):
    _plot_subformulas_vs_factor(
        raw_df,
        factor="tau_flock",
        xlabel=r"Recovery Horizon ($\tau$)",
        suptitle=r"Effect of $\tau$ on Individual Sub-Formula Failure Rates (Mean ± Seed Std)",
        save_path=save_path,
    )


def plot_n_agents_subformulas(raw_df: pd.DataFrame, save_path: str):
    _plot_subformulas_vs_factor(
        raw_df,
        factor="n_agents",
        xlabel="Swarm Size (N Agents)",
        suptitle="Sub-Formula Violations vs. Swarm Size (Mean ± Seed Std)",
        save_path=save_path,
    )


# =============================================================================
# 4. PAIRWISE 2-WAY INTERACTION DELTA HEATMAPS (Rho - Native)
# =============================================================================
def plot_pairwise_delta_interactions(
    raw_df: pd.DataFrame,
    metric: str,
    save_path: str,
    include_tau: bool = True,
):
    """2xN grid of (Rho - Native) heatmaps across key 2-way factor pairs.
    Row 0 = No-Wind Trained, Row 1 = Wind Trained.
    """
    if include_tau:
        pairs = [
            ("wind_strength", "wind_gust", "Wind Strength", "Wind Gust"),
            ("n_agents", "tau_flock", "N Agents", r"$\tau$"),
            ("wind_strength", "tau_flock", "Wind Strength", r"$\tau$"),
            ("n_agents", "wind_strength", "N Agents", "Wind Strength"),
        ]
    else:
        pairs = [
            ("wind_strength", "wind_gust", "Wind Strength", "Wind Gust"),
            ("n_agents", "wind_strength", "N Agents", "Wind Strength"),
            ("n_agents", "wind_gust", "N Agents", "Wind Gust"),
        ]

    row_specs = [(False, "No-Wind Trained"), (True, "Wind Trained")]
    fmt = "+.2f" if metric == "pass_rate" else "+.1f"

    deltas = {}
    max_abs = 1e-6
    for wt, _ in row_specs:
        sub = raw_df[raw_df["wind_trained"] == wt]
        for f_row, f_col, _, _ in pairs:
            m_rho = sub[sub["mode"] == "rho"].groupby([f_row, f_col])[metric].mean().unstack()
            m_nat = sub[sub["mode"] == "none"].groupby([f_row, f_col])[metric].mean().unstack()
            diff = m_rho - m_nat
            deltas[(wt, f_row, f_col)] = diff
            if not diff.isna().all().all():
                max_abs = max(max_abs, np.nanmax(np.abs(diff.values)))

    fig, axes = plt.subplots(
        2, len(pairs),
        figsize=(4.8 * len(pairs) + 1.2, 8.2),
        constrained_layout=True,
    )
    im = None
    for r, (wt, rlabel) in enumerate(row_specs):
        for c, (f_row, f_col, y_lbl, x_lbl) in enumerate(pairs):
            ax = axes[r, c]
            diff_df = deltas[(wt, f_row, f_col)]
            vals = diff_df.values
            im = ax.imshow(vals, vmin=-max_abs, vmax=max_abs, cmap="RdYlGn", aspect="auto")

            for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    v = vals[i, j]
                    if np.isnan(v):
                        continue
                    txt_c = "white" if abs(v) > 0.65 * max_abs else "black"
                    ax.text(j, i, f"{v:{fmt}}", ha="center", va="center", color=txt_c, fontsize=9.5, fontweight="bold")

            ax.set_xticks(range(len(diff_df.columns)))
            ax.set_xticklabels([f"{x:g}" for x in diff_df.columns])
            ax.set_yticks(range(len(diff_df.index)))
            ax.set_yticklabels([f"{y:g}" for y in diff_df.index])
            ax.set_xlabel(x_lbl, fontsize=11, fontweight="bold")
            ax.set_ylabel(f"{rlabel}\n\n{y_lbl}" if c == 0 else y_lbl, fontsize=10.5, fontweight="bold")
            if r == 0:
                ax.set_title(f"{y_lbl} × {x_lbl}", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(f"Δ {metric.replace('_', ' ').title()} (Rho − Native)", fontsize=12, fontweight="bold")
    fig.suptitle(
        f"2-Way Interaction Advantage (Rho − Native): {metric.replace('_', ' ').title()}",
        fontsize=15,
        fontweight="bold",
    )
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# =============================================================================
# 5. FULL (WIND x GUST) MATRICES (Sliced by tau or marginalized)
# =============================================================================
def _marginalize_tau(raw_df: pd.DataFrame, tau: int | None) -> pd.DataFrame:
    df = raw_df if tau is None else raw_df[raw_df["tau_flock"] == tau]
    group_cols = ["mode", "wind_trained", "n_agents", "wind_strength", "wind_gust"]
    metric_cols = ["pass_rate", "return_mean", "rho_mean", "violation_rate"]
    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in agg.columns]
    return agg


def plot_stresstest_matrices(
    raw_df: pd.DataFrame,
    metric: str = "pass_rate",
    tau: int | None = None,
    save_path: str = "stresstest_matrix.png",
):
    """Rows = n_agents, Cols = (mode, wind_trained) grid of (wind_strength x wind_gust) heatmaps."""
    agg_df = _marginalize_tau(raw_df, tau)
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
        vmin, vmax, cmap, fmt = agg_df[mean_col].min(), agg_df[mean_col].max(), "viridis", ".1f"

    n_rows, n_cols = len(n_agents_list), len(col_specs)
    fig, axes = plt.subplots(
        n_rows, n_cols,
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
            mean_mat = sub.pivot(index="wind_strength", columns="wind_gust", values=mean_col).reindex(index=winds, columns=gusts)
            std_mat = sub.pivot(index="wind_strength", columns="wind_gust", values=std_col).reindex(index=winds, columns=gusts)

            im = ax.imshow(mean_mat.values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
            for i in range(len(winds)):
                for j in range(len(gusts)):
                    m_val, s_val = mean_mat.values[i, j], std_mat.values[i, j]
                    if np.isnan(m_val):
                        continue
                    norm_val = (m_val - vmin) / max(vmax - vmin, 1e-8)
                    txt_color = "white" if norm_val > 0.6 else "black"
                    s_str = f"\n±{s_val:{fmt}}" if not np.isnan(s_val) else ""
                    ax.text(j, i, f"{m_val:{fmt}}{s_str}", ha="center", va="center", color=txt_color, fontsize=9, fontweight="bold")

            if r == 0:
                ax.set_title(col_label, fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"N = {n_ag} Agents\n\nWind Strength", fontsize=11, fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("Wind Gust Strength", fontsize=11, fontweight="bold")
            ax.set_xticks(range(len(gusts)))
            ax.set_xticklabels([f"{g:.2f}" for g in gusts])
            ax.set_yticks(range(len(winds)))
            ax.set_yticklabels([f"{w:.2f}" for w in winds])

    tau_str = f"τ = {tau}" if tau is not None else "All τ Marginalized"
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
    fig.suptitle(f"{metric.replace('_', ' ').title()} ({tau_str})", fontsize=15, fontweight="bold")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")

def plot_wind_subformulas(raw_df: pd.DataFrame, save_path: str):
    _plot_subformulas_vs_factor(
        raw_df,
        factor="wind_strength",
        xlabel="Wind Strength",
        suptitle="Sub-Formula Violations vs. Wind Strength (Mean ± Seed Std)",
        save_path=save_path,
    )

def plot_gust_subformulas(raw_df: pd.DataFrame, save_path: str):
    _plot_subformulas_vs_factor(
        raw_df,
        factor="wind_gust",
        xlabel="Wind Gust Strength",
        suptitle="Sub-Formula Violations vs. Wind Strength (Mean ± Seed Std)",
        save_path=save_path,
    )


def plot_oat_effects(
    raw_df: pd.DataFrame,
    metric: str,
    save_path: str,
    base_n: int = 3,
    base_wind: float = 0.17,
    base_gust: float = 0.17,
    base_tau: int = 1,
):
    """1x3 One-at-a-Time (OAT) simple effects plot:
    Sweeps one factor while holding the other two fixed at (base_n, base_wind, base_gust) and tau=base_tau.
    """
    df_tau = raw_df[raw_df["tau_flock"] == base_tau]
    if df_tau.empty:
        raise ValueError(f"No rows found with tau_flock == {base_tau}")

    # Define each panel: (swept_col, xlabel, fixed_mask, subtitle)
    panels = [
        (
            "n_agents",
            "Swarm Size (N Agents)",
            np.isclose(df_tau["wind_strength"], base_wind) & np.isclose(df_tau["wind_gust"], base_gust),
            f"Fixed: Wind={base_wind}, Gust={base_gust}",
        ),
        (
            "wind_strength",
            "Wind Strength",
            (df_tau["n_agents"] == base_n) & np.isclose(df_tau["wind_gust"], base_gust),
            f"Fixed: N={base_n}, Gust={base_gust}",
        ),
        (
            "wind_gust",
            "Wind Gust Strength",
            (df_tau["n_agents"] == base_n) & np.isclose(df_tau["wind_strength"], base_wind),
            f"Fixed: N={base_n}, Wind={base_wind}",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharey=True, constrained_layout=True)

    for ax, (col, xlabel, mask, subtitle) in zip(axes, panels):
        sliced = df_tau[mask]
        x_vals = sorted(sliced[col].unique())

        for (mode, wt), style in REGIME_STYLES.items():
            sub = sliced[(sliced["mode"] == mode) & (sliced["wind_trained"] == wt)]
            stats = sub.groupby(col)[metric].agg(["mean", "std"]).reindex(x_vals)

            y, e = stats["mean"].values, stats["std"].values
            ax.plot(
                x_vals, y,
                label=style["label"],
                color=style["color"],
                linestyle=style["ls"],
                marker=style["marker"],
                linewidth=2.2,
                markersize=7,
            )
            ax.fill_between(x_vals, y - e, y + e, color=style["color"], alpha=0.14)

        ax.set_title(subtitle, fontsize=11.5, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
        ax.set_xticks(x_vals)
        ax.grid(True, alpha=0.35)
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel(metric.replace("_", " ").title(), fontsize=13, fontweight="bold")
    if metric == "pass_rate":
        axes[0].set_ylim(-0.02, 1.05)
    axes[-1].legend(loc="best", fontsize=10, frameon=True)
    fig.suptitle(
        f"One-at-a-Time (Simple) Effects: {metric.replace('_', ' ').title()} (τ = {base_tau}, Mean ± Seed Std)",
        fontsize=14.5,
        fontweight="bold",
    )
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")

def print_compact_summary(raw_df: pd.DataFrame, include_tau: bool = True):
    """Prints high-level summary tables for main effects."""
    factors = ["tau_flock", "n_agents"] if include_tau else ["n_agents"]
    for factor in factors:
        summary = (
            raw_df.groupby(["wind_trained", factor, "mode"])[
                ["pass_rate", "return_mean", "rho_mean"]
            ]
            .agg(["mean", "std"])
        )
        print(f"\n=== Summary by {factor} (Averaged Across Other Factors) ===")
        print(summary.round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Full factorial stress-test analysis & plots")
    p.add_argument("--results", default="results/flocking", help="RESULTS_DIR of the flocking runs")
    p.add_argument(
        "--ignore-tau",
        action="store_true",
        help="Filter strictly to tau_flock=1 instead of marginalizing/sweeping tau",
    )
    a = p.parse_args()

    include_tau = not a.ignore_tau
    sub_folder = "stresstest" if include_tau else "stresstest_tau1"
    out_root = os.path.join(a.results, "comparison", sub_folder)
    os.makedirs(out_root, exist_ok=True)

    raw_df, _ = load_stresstest_results(os.path.join(a.results, "stresstest"))

    if not include_tau:
        raw_df = raw_df[raw_df["tau_flock"] == 1].copy()
        if raw_df.empty:
            raise ValueError("No stress-test results found with tau_flock == 1.")

    print_compact_summary(raw_df, include_tau=include_tau)

    for metric in ["pass_rate", "return_mean", "rho_mean"]:
        plot_main_effects(raw_df, metric, os.path.join(out_root, f"main_effects_{metric}.png"), include_tau=include_tau)
        plot_oat_effects(raw_df, metric, os.path.join(out_root, f"oat_effects_{metric}.png"))
        plot_factorial_eta_squared(raw_df, metric, os.path.join(out_root, f"effect_sizes_{metric}.png"), include_tau=include_tau)
        plot_pairwise_delta_interactions(raw_df, metric, os.path.join(out_root, f"interactions_delta_{metric}.png"), include_tau=include_tau)

    plot_n_agents_subformulas(raw_df, os.path.join(out_root, "n_agents_subformulas.png"))
    plot_wind_subformulas(raw_df, os.path.join(out_root, "wind_subformulas.png"))
    plot_gust_subformulas(raw_df, os.path.join(out_root, "gust_subformulas.png"))

    if include_tau:
        plot_tau_subformulas(raw_df, os.path.join(out_root, "tau_subformulas.png"))
        os.makedirs(os.path.join(out_root, "per_tau"), exist_ok=True)
        for metric in ["pass_rate", "return_mean"]:
            plot_stresstest_matrices(raw_df, metric=metric, tau=None, save_path=os.path.join(out_root, f"matrix_{metric}_all_tau.png"))
            for tau in sorted(raw_df["tau_flock"].unique()):
                plot_stresstest_matrices(raw_df, metric=metric, tau=tau, save_path=os.path.join(out_root, "per_tau", f"matrix_{metric}_tau{tau}.png"))
    else:
        for metric in ["pass_rate", "return_mean"]:
            plot_stresstest_matrices(raw_df, metric=metric, tau=1, save_path=os.path.join(out_root, f"matrix_{metric}_tau1.png"))