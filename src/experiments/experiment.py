import time, yaml, dataclasses, torch
import argparse

from environment.environment import make_env
from fmai.train import train
from fmai.utils import build_spec, W
from fmai.agents import Agent
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
    ):

    e = load_experiment(path)
    run_id = time.strftime("%Y%m%d_%H%M%S")

    spec, spec_desc = build_spec()

    for mode in e["modes"]:
        cfg = Config(**e["config"], REWARD_MODE=mode, RUN_ID=run_id)
        
        shaped = mode in ["rho", "mixed"]
        env = make_env(cfg, cfg.NUM_ENVS, cfg.SEED, spec=spec if shaped else None, mode=mode, window=W)
        eval_env = make_env(cfg, cfg.EVAL_EPISODES, cfg.SEED + 1000)         # native reward, no shaping

        torch.manual_seed(cfg.SEED)
        agent = Agent()
        agent.make_agent(env=env, cfg=cfg)

        training_hist = train(env=env, eval_env=eval_env, agent=agent, spec=spec, spec_desc=spec_desc, cfg=cfg)


if __name__ == "__main__":
    args = parse_args()
    path = f"src/experiments/{args.exp}"
    run_all(path=path)