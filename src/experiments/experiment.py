import time, yaml, dataclasses, torch
import argparse

from environment.environment import make_env
from fmai.train import train
from fmai.agents import Agent
from fmai.sfo import build_spec, TAU2, TAU1
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

    parser.add_argument("--n_agents", type=int, default=None)
    parser.add_argument("--mappo", type=bool, default=True)
    parser.add_argument("--shared_reward", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()

def cli_overrides(args):
    cli_overrides = {}
    if args.env is not None:
        cli_overrides["N_AGENTS"] = args.n_agents
    if args.seed is not None:
        cli_overrides["SEED"] = args.seed
    if args.mappo is not None:
        cli_overrides["MAPPO"] = args.mappo
    if args.shared_reward is not None:
        cli_overrides["SHARED_REW"] = args.shared_rewards

    return cli_overrides

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
        cfg = Config(**e["config"], REWARD_MODE=mode, RUN_ID=run_id)
        for key, value in overwrites.items():
            cfg[key] = value
        
        
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