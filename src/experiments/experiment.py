import time, yaml, dataclasses, torch, itertools
from pathlib import Path
from types import SimpleNamespace
import argparse
import warnings
import os
import json

# Ignore those suckers
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


from environment.environment import make_env
from fmai.train import train, evaluate
from fmai.agents import Agent
from fmai.sfo import build_spec, TAU2, TAU1, GatedSpec
from fmai.utils import option_label, save_compare_plots
from config import Config


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}.")

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Path to experiment .yaml config",
    )
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--n_agents", type=int, default=None)
    parser.add_argument("--mappo", type=str_to_bool, default=None)
    parser.add_argument("--shared_reward", type=str_to_bool, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--wind_strength", type=float, default=None)
    parser.add_argument("--wind_gust_strength", type=float, default=None)
    parser.add_argument("--train", type=str_to_bool, default=False)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--wind_trained", type=str_to_bool, default=True)
    # Several options and seeds per call; for parallel runs (cluster.sbatch) one process per option/seed, all sharing
    # --run-id, then --compare-only
    parser.add_argument("--modes", nargs="+", help="Options, run in order: none | rho[:spec] | mixed[:spec]")
    parser.add_argument("--seeds", type=int, help="Number of seeds per option (N_SEEDS)")
    parser.add_argument("--smooth", type=int, help="Moving-average window in eval points for compare plots (SMOOTH)")
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

from fmai.sfo import build_spec, TAU2, TAU1

def cli_overrides(args):
    overrides = {}
    if args.scenario is not None:
        overrides["SCENARIO"] = args.scenario
    if args.n_agents is not None:
        overrides["NUM_AGENTS"] = args.n_agents
    if args.seed is not None:
        overrides["SEED"] = args.seed
    if args.mappo is not None:
        overrides["MAPPO"] = args.mappo
    if args.shared_reward is not None:
        overrides["SHARED_REW"] = args.shared_reward
    if args.results_dir is not None:
        overrides["RESULTS_DIR"] = args.results_dir
    if args.wind_strength is not None:
        overrides["WIND_STRENGTH"] = args.wind_strength
    if args.wind_gust_strength is not None:
        overrides["WIND_GUST_STRENGTH"] = args.wind_gust_strength
    if args.mode is not None:
        overrides["REWARD_MODE"] = args.mode
    if args.seeds is not None:
        overrides["N_SEEDS"] = args.seeds
    if args.smooth is not None:
        overrides["SMOOTH"] = args.smooth
    return overrides

def run_all(args, path="experiment.yaml"):
    e = load_experiment(path)
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    overwrites = cli_overrides(args=args)
    base = Config(**{**e["config"], **overwrites})                   # resolved SEED, N_SEEDS

    for s, entry in itertools.product(range(base.N_SEEDS), args.modes or e["modes"]):   # all options per seed
        mode, _, spec_name = entry.partition(":")                    # "rho:robustness" -> ("rho", "robustness")
        merged_cfg = {**e["config"], **overwrites, "SEED": base.SEED + s, "REWARD_MODE": mode, "RUN_ID": run_id}
        if spec_name:
            merged_cfg["SPEC"] = spec_name
        cfg = Config(**merged_cfg)

        spec, spec_desc = build_spec(cfg)
        reward_fn = None
        if isinstance(spec, GatedSpec):                              # GatedSpec: scored in train(), not in the env
            reward_fn = spec.reward if mode in ["rho", "mixed"] else None
            spec = None                                              # eval then uses the spec-independent sfo_rho
        window = cfg.TAU_FLOCK if cfg.SCENARIO == "flocking" else (TAU1 + TAU2)

        shaped = mode in ["rho", "mixed"]
        env = make_env(
            cfg,
            num_envs=cfg.NUM_ENVS,
            seed=cfg.SEED,
            spec=spec if shaped else None,
            mode=mode,
            window=window,
        )
        eval_env = make_env(
            cfg,
            num_envs=cfg.EVAL_EPISODES,
            seed=cfg.SEED + 1000,
            spec=None,
            mode="none",
            window=window,
        )

        torch.manual_seed(cfg.SEED)
        agent = Agent()
        agent.make_agent(env=env, cfg=cfg)

        training_hist = train(
            env=env, eval_env=eval_env, agent=agent, spec=spec, spec_desc=spec_desc, cfg=cfg, reward_fn=reward_fn
        )


def compare(args, path):
    """Merge the per-process histories of a parallel run (traces/<run-id>/*.json) into compare.png / .csv."""
    e = load_experiment(path)
    cfg = Config(**{**e["config"], **cli_overrides(args=args)})
    results = {}
    for f in sorted((Path(cfg.RESULTS_DIR) / "traces" / args.run_id).glob("*.json")):
        d = json.loads(f.read_text())
        results.setdefault(option_label(SimpleNamespace(**d["config"])), []).append(d["history"])
    title = (f"{cfg.SCENARIO} | {'MAPPO' if cfg.MAPPO else 'IPPO'} | {cfg.NUM_AGENTS} agents | "
             f"{int(cfg.EPISODES):,} episodes")
    save_compare_plots(results, Path(cfg.RESULTS_DIR) / "traces" / args.run_id / "compare",
                       smooth_k=cfg.SMOOTH, title=title)



def stresstest(
    mode: str,
    seed: int,
    wind: bool,
    path: str,
    args,
    ):
    e = load_experiment(path=path)
    overwrites = cli_overrides(args=args)

    merged_cfg = {**e["config"], **overwrites, "SEED": seed, "REWARD_MODE": mode}
    cfg = Config(**merged_cfg)

    spec, spec_desc = build_spec(cfg)
    window = cfg.TAU_FLOCK if cfg.SCENARIO == "flocking" else (TAU1 + TAU2)
    device = cfg.DEVICE

    testing_env = make_env(
        cfg=cfg,
        num_envs=cfg.EVAL_EPISODES,
        seed=seed + 1000,
        spec=None,
        mode="none",
        window=window,
    )

    agent = Agent()
    agent.make_agent(env=testing_env, cfg=cfg)
    agent.policy.to(device)  # type: ignore

    base_dir = "results/flocking_wind" if wind else "results/flocking"
    agents_dir = os.path.join(base_dir, f"seed_{seed}", "agents")
    timestamps = sorted([
        t
        for t in os.listdir(agents_dir)
        if os.path.isdir(os.path.join(agents_dir, t))
    ])
    assert timestamps, f"No timestamp directories found in {agents_dir}"

    agent_path = os.path.join(agents_dir, timestamps[-1], f"flocking_{mode}_seed{seed}.pt")
    assert os.path.exists(agent_path), f"Given path does not exist. Path: {agent_path}"

    checkpoint = torch.load(f=agent_path, map_location=device, weights_only=True)
    agent.policy.load_state_dict(checkpoint["policy"])  # type: ignore
    agent.policy.eval()  # type: ignore

    info = evaluate(
        eval_env=testing_env,
        agent=agent,
        spec=spec,
        cfg=cfg,
        training_episode=0,
    )
    return info, cfg, spec_desc


if __name__ == "__main__":
  args = parse_args()
  path = f"src/experiments/{args.exp}"

  if args.compare_only:
    compare(args, path=path)
  elif args.train:
    run_all(args, path=path)
  else:
    info, cfg, spec_desc = stresstest(seed=args.seed, mode=args.mode, wind=args.wind_trained, path=path, args=args)

    out_dir = os.path.join(cfg.RESULTS_DIR, "stresstest", f"seed_{cfg.SEED}" )
    os.makedirs(out_dir, exist_ok=True)
    train_tag = "windtrained" if args.wind_trained else "nowindtrained"
    out_file = os.path.join(
        out_dir,
        f"{cfg.REWARD_MODE}_{train_tag}_n{cfg.NUM_AGENTS}_wind_{cfg.WIND_STRENGTH}_gust_{cfg.WIND_GUST_STRENGTH}.json",
        )
    
    with open(out_file, "w") as f:
        json.dump(
            {"config": vars(cfg), "formulas": spec_desc, "history": [info]},
            f,
            indent=2,
        )
    print(
        f"[Done] mode={cfg.REWARD_MODE}, seed={cfg.SEED}, n_agents={cfg.NUM_AGENTS} ->"
        f" pass_rate={info['pass_rate']:.2f},"
        f" return={info['return_mean']:.2f}"
    )