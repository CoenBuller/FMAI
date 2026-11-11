import time, yaml, dataclasses, torch
import argparse
import warnings
# Ignore those suckers
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


from environment.environment import make_env
from fmai.train import train
from fmai.agents import Agent
from fmai.sfo import build_spec, TAU2, TAU1
from config import Config

from torchrl.envs.libs.vmas import VmasEnv


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
    parser.add_argument("--n_agents", type=int, default=None)
    parser.add_argument("--mappo", type=str_to_bool, default=None)
    parser.add_argument("--shared_reward", type=str_to_bool, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    return parser.parse_args()

def cli_overrides(args):
    overrides = {}
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
    return overrides

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
    args,
    path="experiment.yaml",
    ):

    e = load_experiment(path)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    overwrites = cli_overrides(args=args)

    spec, spec_desc = build_spec()

    for mode in e["modes"]:
        merged_cfg = {**e["config"], **overwrites, "REWARD_MODE": mode, "RUN_ID": run_id}
        cfg = Config(**merged_cfg)
        
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

        training_hist = train(env=env, eval_env=eval_env, agent=agent, spec=spec, spec_desc=spec_desc, cfg=cfg)


if __name__ == "__main__":
    args = parse_args()
    path = f"src/experiments/{args.exp}"
    run_all(args, path=path)