import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

METRIC_LABELS = {
    "pass_rate": "Pass Rate [0, 1]",
    "return_mean": "Native Return",
    "rho_mean": r"Robustness ($\rho$)",
}


def get_active_styles(trained: str) -> dict:
    if trained == "wind":
        return {k: v for k, v in REGIME_STYLES.items() if k[1] is True}
    if trained == "no_wind":
        return {k: v for k, v in REGIME_STYLES.items() if k[1] is False}
    return REGIME_STYLES


def load_stresstest_results(root_dir: str) -> pd.DataFrame:
    pattern = os.path.join(root_dir, "seed_*", "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No JSON files found matching {pattern}")

    rows = []
    for path in files:
        fname = os.path.basename(path)
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
        }
        for k, v in info.get("sub_fail_rate", {}).items():
            row[f"fail_{k}"] = v
        rows.append(row)

    return pd.DataFrame(rows)


def _oat_slices(df_tau: pd.DataFrame, base_n: int, base_wind: float, base_gust: float):
    """Returns the 3 OAT slices: (swept_col, xlabel, subtitle, sliced_df)."""
    return [
        (
            "n_agents",
            "Swarm Size (N Agents)",
            f"Fixed: Wind={base_wind}, Gust={base_gust}",
            df_tau[np.isclose(df_tau["wind_strength"], base_wind) & np.isclose(df_tau["wind_gust"], base_gust)],
        ),
        (
            "wind_strength",
            "Wind Strength",
            f"Fixed: N={base_n}, Gust={base_gust}",
            df_tau[(df_tau["n_agents"] == base_n) & np.isclose(df_tau["wind_gust"], base_gust)],
        ),
        (
            "wind_gust",
            "Wind Gust Strength",
            f"Fixed: N={base_n}, Wind={base_wind}",
            df_tau[(df_tau["n_agents"] == base_n) & np.isclose(df_tau["wind_strength"], base_wind)],
        ),
    ]


def plot_oat_metric(
    df_tau: pd.DataFrame,
    metric: str,
    save_path: str,
    base_n: int,
    base_wind: float,
    base_gust: float,
    base_tau: int,
    styles: dict = REGIME_STYLES,
):
    """1x3 OAT plot for a single scalar metric (pass_rate, return_mean, or rho_mean)."""
    slices = _oat_slices(df_tau, base_n, base_wind, base_gust)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), sharey=True, constrained_layout=True)

    for ax, (col, xlabel, subtitle, sliced) in zip(axes, slices):
        x_vals = sorted(sliced[col].unique())
        for (mode, wt), style in styles.items():
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

    axes[0].set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=13, fontweight="bold")
    if metric == "pass_rate":
        axes[0].set_ylim(-0.02, 1.05)
    axes[-1].legend(loc="best", fontsize=10, frameon=True)
    fig.suptitle(
        f"One-at-a-Time Effects: {METRIC_LABELS.get(metric, metric)} (τ = {base_tau}, Mean ± Seed Std)",
        fontsize=14.5,
        fontweight="bold",
    )
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_oat_subformulas(
    df_tau: pd.DataFrame,
    save_path: str,
    base_n: int,
    base_wind: float,
    base_gust: float,
    base_tau: int,
    styles: dict = REGIME_STYLES,
):
    """3x3 grid comparing sub-formula failure rates (excluding v_max) under OAT sweeps.
    Rows = swept factor (N Agents, Wind Strength, Wind Gust), Cols = sub-formulas.
    """
    fail_cols = [
        c for c in df_tau.columns
        if c.startswith("fail_") and "phi_vm" not in c
    ]
    if not fail_cols:
        print("No sub_fail_rate columns found; skipping sub-formula plot.")
        return

    slices = _oat_slices(df_tau, base_n, base_wind, base_gust)
    fig, axes = plt.subplots(
        len(slices), len(fail_cols),
        figsize=(4.8 * len(fail_cols), 3.8 * len(slices)),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )

    for r, (col, xlabel, subtitle, sliced) in enumerate(slices):
        x_vals = sorted(sliced[col].unique())
        for c, fcol in enumerate(fail_cols):
            ax = axes[r, c]
            for (mode, wt), style in styles.items():
                sub = sliced[(sliced["mode"] == mode) & (sliced["wind_trained"] == wt)]
                stats = sub.groupby(col)[fcol].agg(["mean", "std"]).reindex(x_vals)
                y, e = stats["mean"].values, stats["std"].values

                ax.plot(
                    x_vals, y,
                    label=style["label"],
                    color=style["color"],
                    linestyle=style["ls"],
                    marker=style["marker"],
                    linewidth=2.0,
                    markersize=6.5,
                )
                ax.fill_between(
                    x_vals,
                    np.clip(y - e, 0.0, 1.0),
                    np.clip(y + e, 0.0, 1.0),
                    color=style["color"],
                    alpha=0.14,
                )

            if r == 0:
                ax.set_title(PRETTY_SUBFORMULAS.get(fcol, fcol.replace("fail_", "")), fontsize=12.5, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{subtitle}\n\nFailure Rate [0, 1]", fontsize=10.5, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
            ax.set_xticks(x_vals)
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.35)
            ax.tick_params(labelsize=9.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=len(styles),
        fontsize=11,
        frameon=True,
    )
    fig.suptitle(
        f"One-at-a-Time Sub-Formula Violations (τ = {base_tau}, Mean ± Seed Std)",
        fontsize=15,
        fontweight="bold",
        y=1.08,
    )
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="One-at-a-Time (OAT) stress-test plots")
    p.add_argument("--results", default="results/flocking", help="RESULTS_DIR of the flocking runs")
    p.add_argument("--base-n", type=int, default=3, help="Default swarm size")
    p.add_argument("--base-wind", type=float, default=0.17, help="Default wind strength")
    p.add_argument("--base-gust", type=float, default=0.17, help="Default wind gust strength")
    p.add_argument("--base-tau", type=int, default=1, help="Fixed tau_flock value")
    p.add_argument(
        "--trained",
        choices=["both", "wind", "no_wind"],
        default="both",
        help="Which trained policies to plot (default: both)",
    )
    a = p.parse_args()

    styles = get_active_styles(a.trained)
    suffix = "" if a.trained == "both" else f"_{a.trained}"
    out_dir = os.path.join(a.results, "comparison", f"stresstest_oat{suffix}")
    os.makedirs(out_dir, exist_ok=True)

    raw_df = load_stresstest_results(os.path.join(a.results, "stresstest"))
    df_tau = raw_df[raw_df["tau_flock"] == a.base_tau].copy()
    if df_tau.empty:
        raise ValueError(f"No stress-test results found with tau_flock == {a.base_tau}")

    for metric in ["pass_rate", "return_mean", "rho_mean"]:
        plot_oat_metric(
            df_tau,
            metric=metric,
            save_path=os.path.join(out_dir, f"oat_{metric}.png"),
            base_n=a.base_n,
            base_wind=a.base_wind,
            base_gust=a.base_gust,
            base_tau=a.base_tau,
            styles=styles,
        )

    plot_oat_subformulas(
        df_tau,
        save_path=os.path.join(out_dir, "oat_subformulas.png"),
        base_n=a.base_n,
        base_wind=a.base_wind,
        base_gust=a.base_gust,
        base_tau=a.base_tau,
        styles=styles,
    )