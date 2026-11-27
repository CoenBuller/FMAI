import matplotlib.pyplot as plt
import glob
import os
import json
import numpy as np

from matplotlib.gridspec import GridSpec

OPTION_DIRS = {"none": "native", "rho": "robustness"}


def get_jsons(results: str, run_id: str, mode: str = "none"):
    """The history JSONs of every seed: <results>/<native|robustness>/traces/<run_id>/flocking_<mode>_seed<N>.json"""
    return sorted(glob.glob(os.path.join(results, OPTION_DIRS[mode], "traces", run_id, f"flocking_{mode}_seed*.json")))


def aggregate_reward_mode(results: str, run_id: str, reward_mode: str) -> dict:
    """Loads all seed JSONs using get_jsons and aggregates training curves

    (mean & std across seeds) into a dictionary.
    """
    base_folder = os.path.join(results, "*", "traces", run_id)
    paths = get_jsons(results, run_id, mode=reward_mode)
    seed_histories = []

    for json_path in paths:
        if os.path.isfile(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                seed_histories.append(data["history"])

    if not seed_histories:
        raise FileNotFoundError(
            f"No JSON traces found for reward_mode='{reward_mode}' in {base_folder}"
        )

    def agg_curve(extractor_fn):
        arr = np.array(
            [[extractor_fn(entry) for entry in hist] for hist in seed_histories],
            dtype=float,
        )
        mean = np.full(arr.shape[1], np.nan)
        std = np.full(arr.shape[1], np.nan)
        valid = ~np.all(np.isnan(arr), axis=0)
        if np.any(valid):
            mean[valid] = np.nanmean(arr[:, valid], axis=0)
            std[valid] = np.nanstd(arr[:, valid], axis=0)
        return {"mean": mean, "std": std}

    ref_hist = seed_histories[0]
    sub_rho_keys = list(ref_hist[0]["sub_rho"].keys())
    sub_fail_keys = list(ref_hist[0]["sub_fail_rate"].keys())
    rew_comp_keys = list(ref_hist[0]["reward_components"].keys())

    agg_dict = {
        "steps": np.array([entry["step"] for entry in ref_hist]),
        "episodes": np.array([entry["episode"] for entry in ref_hist]),
        "return": agg_curve(lambda e: e["return_mean"]),
        "native_return": agg_curve(lambda e: e["native_return"]),
        "train_return": agg_curve(lambda e: e["train_return"]),
        "train_reward": agg_curve(lambda e: e["train_reward"]),
        "rho": agg_curve(lambda e: e["rho_mean"]),
        "pass_rate": agg_curve(lambda e: e["pass_rate"]),
        "violation_rate": agg_curve(lambda e: e["violation_rate"]),
        "sub_rho": {
            k: agg_curve(lambda e, key=k: e["sub_rho"][key]) for k in sub_rho_keys
        },
        "sub_fail_rate": {
            k: agg_curve(lambda e, key=k: e["sub_fail_rate"][key])
            for k in sub_fail_keys
        },
        "reward_components": {
            k: agg_curve(lambda e, key=k: e["reward_components"][key])
            for k in rew_comp_keys
        },
    }

    agg_dict["final"] = {
        "return": {
            "mean": float(agg_dict["return"]["mean"][-1]),
            "std": float(agg_dict["return"]["std"][-1]),
        },
        "pass_rate": {
            "mean": float(agg_dict["pass_rate"]["mean"][-1]),
            "std": float(agg_dict["pass_rate"]["std"][-1]),
        },
        "rho": {
            "mean": float(agg_dict["rho"]["mean"][-1]),
            "std": float(agg_dict["rho"]["std"][-1]),
        },
    }

    return agg_dict


def plot_flocking(results: str, run_id: str, save_path: str):
    fig = plt.figure(figsize=(32, 8))
    label_size = 20
    marker_size = 17
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    # Outer grid: 3 main groups (left group = 2x width of the two bar plots)
    outer = GridSpec(1, 3, width_ratios=[2, 1, 1], wspace=0.18)

    # Inner grid for ax[0] and ax[1]: wspace=0 squashes them together
    left_sub = outer[0].subgridspec(1, 2, wspace=0)

    ax0 = fig.add_subplot(left_sub[0])
    ax0.set_title(r"Native Reward", size=label_size + 2)
    ax1 = fig.add_subplot(left_sub[1], sharey=ax0)
    ax1.set_title(r"Robustness Reward ($\rho$)", size=label_size + 2)
    ax2 = fig.add_subplot(outer[1])
    ax2.set_title("Final Pass Rate", size=label_size + 2)
    ax3 = fig.add_subplot(outer[2])
    ax3.set_title("Final Evaluation Return", size=label_size + 2)
    ax = [ax0, ax1, ax2, ax3]

    # Cohesive color palette across line plots and bar plots
    spec_labels = {
        "phi_c (collision)": r"$\varphi_c$ (collision)",
        "phi_vm (v_max)": r"$\varphi_{v_m}$ (v$_{max}$)",
        "phi_sd (dist_band)": r"$\varphi_{s_d}$ (dist. band)",
        "phi_sv (tan_speed)": r"$\varphi_{s_v}$ (tang. speed)",
    }
    spec_colors = {
        "phi_c (collision)": "#2b5c8f",  # Deep steel blue
        "phi_vm (v_max)": "#d95f02",  # Muted burnt orange
        "phi_sd (dist_band)": "#2b83ba",  # Teal/cyan blue (or "#1b9e77" sage green)
        "phi_sv (tan_speed)": "#b2182b",  # Muted crimson
    }
    # Match bar colors to the primary blue & green tones of the line plots
    bar_colors = ["#2b5c8f", "#1b9e77"]

    native = aggregate_reward_mode(results, run_id, reward_mode="none")
    rho = aggregate_reward_mode(results, run_id, reward_mode="rho")

    for key in native["sub_fail_rate"].keys():
        c = spec_colors[key]
        lbl = spec_labels[key]

        # --- ax[0]: Native ---
        x = native["steps"]
        y = native["sub_fail_rate"][key]["mean"]
        e = native["sub_fail_rate"][key]["std"]
        ax[0].plot(x, y, linewidth=2.0, color=c, label=lbl)
        ax[0].fill_between(x, y - e, y + e, color=c, alpha=0.18)

        # --- ax[1]: Rho ---
        x = rho["steps"]
        y = rho["sub_fail_rate"][key]["mean"]
        e = rho["sub_fail_rate"][key]["std"]
        ax[1].plot(x, y, linewidth=2.0, color=c, label=lbl)
        ax[1].fill_between(x, y - e, y + e, color=c, alpha=0.18)

    bar_labels = ["Native", r"Rho ($\rho$)"]

    # --- ax[2]: Final Pass Rate ---
    pass_means = [
        native["final"]["pass_rate"]["mean"],
        rho["final"]["pass_rate"]["mean"],
    ]
    pass_stds = [
        native["final"]["pass_rate"]["std"],
        rho["final"]["pass_rate"]["std"],
    ]
    ax[2].bar(
        bar_labels,
        pass_means,
        yerr=pass_stds,
        width=0.48,
        color=bar_colors,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.9,
        error_kw=dict(elinewidth=2.0, capsize=6, capthick=2.0),
    )

    # --- ax[3]: Final Return ---
    ret_means = [
        native["final"]["return"]["mean"],
        rho["final"]["return"]["mean"],
    ]
    ret_stds = [
        native["final"]["return"]["std"],
        rho["final"]["return"]["std"],
    ]
    ax[3].bar(
        bar_labels,
        ret_means,
        yerr=ret_stds,
        width=0.48,
        color=bar_colors,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.9,
        error_kw=dict(elinewidth=2.0, capsize=6, capthick=2.0),
    )

    # --- Formatting & Styling ---
    ax[0].set_xlabel("Env Step", size=label_size)
    ax[0].set_ylabel(r"Sub-formula failure rate [0, 1]", size=label_size)
    ax[0].tick_params(axis="both", which="major", labelsize=marker_size)
    ax[0].grid(True, alpha=0.4)
    ax[0].legend(fontsize=label_size - 3, loc="upper right", frameon=True)

    ax[1].set_xlabel("Env Step", size=label_size)
    ax[1].grid(True, alpha=0.4)
    ax[1].tick_params(axis="y", left=False, labelleft=False)
    ax[1].tick_params(axis="x", which="major", labelsize=marker_size)
    # Hide the first x-tick label on ax[1] so it doesn't collide with ax[0]'s last x-tick
    ax[1].xaxis.get_major_ticks()[0].label1.set_visible(False)

    ax[2].set_ylabel("Final Pass Rate [0, 1]", size=label_size)
    ax[2].set_ylim(0, 1.08)
    ax[2].tick_params(axis="both", which="major", labelsize=marker_size)
    ax[2].grid(True, axis="y", alpha=0.4)

    ax[3].set_ylabel("Final Native Return", size=label_size)
    ax[3].axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.5)
    ax[3].tick_params(axis="both", which="major", labelsize=marker_size)
    ax[3].grid(True, axis="y", alpha=0.4)

    ax[2].set_xticks(range(len(bar_labels)))
    ax[2].set_xticklabels(bar_labels, fontweight="bold")
    ax[3].set_xticks(range(len(bar_labels)))
    ax[3].set_xticklabels(bar_labels, fontweight="bold")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[plots] wrote {save_path}")




if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Flocking learning curves (sub-formula failure rates) and final eval")
    p.add_argument("--results", default="results/flocking", help="RESULTS_DIR of the flocking runs")
    p.add_argument("--run-ids", nargs="+", default=["no_wind", "wind"])
    a = p.parse_args()
    for rid in a.run_ids:
        plot_flocking(a.results, rid, os.path.join(a.results, "comparison", rid, "flocking.png"))    
