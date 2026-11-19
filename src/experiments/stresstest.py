import os
import torch

from fmai.train import evaluate
from fmai.agents import Agent
from environment.environment import make_env
from config import Config

def stresstest(
        seed: int, 
        mode: str, 
        wind: bool, 
        cfg: Config,
        spec
        ):
    
    device = cfg.DEVICE
    testing_env = make_env(
        cfg=cfg, 
        num_envs=cfg.NUM_ENVS, 
        seed=seed, 
        spec=spec, 
        mode=mode, 
        window=4
        )

    agent = Agent()
    agent.make_agent(env=testing_env, cfg=cfg)
    agent.actor_net.to(device) # type: ignore

    agent_path = "results/flocking"+ f"_wind/seed_{seed}/agents/" if wind else f"/seed_{seed}/agents/" 
    agent_path += os.listdir(agent_path)[0] + f"/flocking_{mode}_seed{seed}.pt"
    assert os.path.exists(agent_path), f"Given path does not exist. Path: {agent_path}"

    state_dict = torch.load(f=agent_path, map_location=device, weights_only=True)
    agent.actor_net.load_state_dict(state_dict=state_dict)
    agent.actor_net.eval()

    info = evaluate(
        eval_env=testing_env,
        agent=agent,
        spec=spec, 
        cfg=cfg,
        training_episode=0
        )

    return info


if __name__ == "__main__":
    stresstest(seed=30, mode="rho", wind=True)