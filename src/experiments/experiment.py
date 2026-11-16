import time, yaml, dataclasses, torch
import argparse, itertools, json
from types import SimpleNamespace

from environment.environment import make_env
from fmai.train import train
from fmai.agents import Agent
from fmai.sfo import build_spec, TAU2, TAU1
from fmai.utils import option_label, save_compare_plots
from pathlib import Path
from config import Config

from torchrl.envs.libs.vmas import VmasEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Path to experiment .yaml config",
    )
    # Optional overrides of the yaml (None = keep the yaml / Config value)
    parser.add_argument("--steps", type=int, help="Training budget in env steps (STEPS)")
    parser.add_argument("--agents", type=int, help="Number of agents (NUM_AGENTS)")
    parser.add_argument("--seeds", type=int, help="Number of seeds per option (N_SEEDS)")
    parser.add_argument("--smooth", type=int, help="Moving-average window in eval points for compare plots (SMOOTH)")
    parser.add_argument("--plot", choices=["diagnostic", "compare"],
                        help="diagnostic: per-run 6-panel PNG. compare: also one overlay figure + table of all options")
    parser.add_argument("--eval-spec", help="Eval robustness: 'sfo' (spec-independent) or a build_spec name (EVAL_SPEC)")
    parser.add_argument("--modes", nargs="+", help="Options, run in order: none | rho[:spec] | mixed[:spec]")
    parser.add_argument("--no-early-stop", action="store_true", help="Only the time limit ends an episode")
    # For parallel runs (cluster.sbatch): one process per option/seed, all sharing --run-id, then --compare-only
    parser.add_argument("--seed", type=int, help="First seed (SEED)")
    parser.add_argument("--run-id", help="Output folder name under traces/agents/gifs (default: timestamp)")
    parser.add_argument("--compare-only", action="store_true",
                        help="No training: build the compare figure + table from the JSONs of --run-id")

    return parser.parse_args()

def load_experiment(path="experiment.yaml"):
    with open(path, encoding="utf-8") as f:
        y = yaml.safe_load(f)    

    # catch typos in Config field names early
    valid = {f.name for f in dataclasses.fields(Config)}
    bad = set(y["config"]) - valid
    if bad:
        raise KeyError(f"Not Config fields: {sorted(bad)}")

    return dict(
        formula_weights=y.get("formula_weights"),   # None -> equal weights
        modes=y["modes"],
        config=y["config"],
    )

def run_all(
    path="experiment.yaml",
    overrides=None,
    modes=None,
    run_id=None,
    compare_only=False,
    ):

    e = load_experiment(path)
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    base = {**e["config"], **(overrides or {})}
    base0 = Config(**base)                                   # resolved defaults (STEPS, N_SEEDS, ...)

    if compare_only:                                         # merge the per-process histories of a parallel run
        results = {}
        for f in sorted((Path(base0.RESULTS_DIR) / "traces" / run_id).glob("*.json")):
            d = json.loads(f.read_text())
            results.setdefault(option_label(SimpleNamespace(**d["config"])), []).append(d["history"])
        return save_compare(results, base0, run_id)

    # Eval spec: None -> spec-independent reference robustness (see train.evaluate)
    eval_spec = None if base0.EVAL_SPEC == "sfo" else build_spec(base0.EVAL_SPEC)[0]
    results = {}                                             # option label -> [history per seed]

    for s, entry in itertools.product(range(base0.N_SEEDS), modes or e["modes"]):   # all options per seed, in order
        mode, _, spec_name = entry.partition(":")            # "rho:coen" -> ("rho", "coen")
        cfg = Config(**{**base, "SEED": base0.SEED + s, "SPEC": spec_name or base0.SPEC},
                     REWARD_MODE=mode, RUN_ID=run_id)
        spec, spec_desc = build_spec(cfg.SPEC) if mode in ["rho", "mixed"] else (None, [])

        shaped = mode in ["rho", "mixed"]
        env = make_env(
            cfg, 
            num_envs=cfg.NUM_ENVS, 
            seed=cfg.SEED, 
            spec=spec if shaped else None, 
            mode=mode, 
            window=TAU2+TAU1
            )
        eval_env = make_env(
            cfg, 
            num_envs=cfg.EVAL_EPISODES, 
            seed=cfg.SEED + 1000,
            spec=None,
            mode="none",
            window=TAU2+TAU1,
            )        

        torch.manual_seed(cfg.SEED)
        agent = Agent()
        agent.make_agent(env=env, cfg=cfg)

        training_hist = train(env=env, eval_env=eval_env, agent=agent, spec=eval_spec, spec_desc=spec_desc, cfg=cfg)
        results.setdefault(option_label(cfg), []).append(training_hist)

    if base0.PLOT_MODE == "compare":
        save_compare(results, base0, run_id)


def save_compare(results, cfg, run_id):
    title = (f"{cfg.SCENARIO} | {'MAPPO' if cfg.MAPPO else 'IPPO'} | {cfg.NUM_AGENTS} agents | "
             f"{cfg.STEPS:,} env steps | eval ρ: {cfg.EVAL_SPEC}")
    save_compare_plots(results, Path(cfg.RESULTS_DIR) / "traces" / run_id / "compare",
                       smooth_k=cfg.SMOOTH, title=title)


if __name__ == "__main__":
    args = parse_args()
    path = f"src/experiments/{args.exp}"
    flags = {"STEPS": args.steps, "NUM_AGENTS": args.agents, "N_SEEDS": args.seeds, "SMOOTH": args.smooth,
             "PLOT_MODE": args.plot, "EVAL_SPEC": args.eval_spec, "SEED": args.seed}
    overrides = {k: v for k, v in flags.items() if v is not None}
    if args.no_early_stop:
        overrides["EARLY_STOP"] = False
    run_all(path=path, overrides=overrides, modes=args.modes, run_id=args.run_id, compare_only=args.compare_only)