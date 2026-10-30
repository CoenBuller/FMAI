import time, yaml, dataclasses

from fmai.train import train, build_spec
from fmai.agents import Agent
from config import Config

from torchrl.envs.libs.vmas import VmasEnv


def load_experiment(path="experiment.yaml"):
    with open(path, encoding="utf-8") as f:
        y = yaml.safe_load(f)    


    # catch typos in Config field names early
    valid = {f.name for f in dataclasses.fields(Config)}
    bad = set(y["config"]) - valid
    if bad:
        raise KeyError(f"Not Config fields: {sorted(bad)}")

    return dict(
        formulas=y["formulas"],
        formula_weights=y.get("formula_weights"),   # None -> equal weights
        horizon=y["horizon"],
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
        
        env = VmasEnv(
            scenario=cfg.SCENARIO,
            num_envs=cfg.NUM_ENVS,
            continuous_actions=cfg.CONTINUOUS_ACTIONS,
            max_steps=cfg.MAX_STEPS,
            seed=cfg.SEED,
            device=cfg.DEVICE
        )

        eval_env = VmasEnv(
            scenario=cfg.SCENARIO,
            num_envs=cfg.EVAL_EPISODES,
            continuous_actions=cfg.CONTINUOUS_ACTIONS,
            max_steps=cfg.MAX_STEPS,
            seed=cfg.SEED + 1, 
            device=cfg.DEVICE
        )
        assert env.max_steps == cfg.MAX_STEPS == eval_env.max_steps

        agent = Agent()
        agent.make_agent(env=env, cfg=cfg)
        g = agent.optim.param_groups[0]

        trainnig_hist = train(
            env=env,
            eval_env=eval_env,
            agent=agent,
            spec=spec,
            spec_desc=spec_desc,
            horizon=cfg.HORIZON,
            cfg=cfg,
            # verbose_eval=True,
        )


if __name__ == "__main__":
    path = "src/experiments/experiment_1.yaml"
    run_all(path=path)


