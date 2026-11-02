from torchrl.envs import StepCounter, TransformedEnv
from navigation_sfo import NavigationSFO

def make_env(cfg, num_envs, seed):
    env = VmasEnv(
        scenario=NavigationSFO(), 
        num_envs=num_envs, 
        continuous_actions=True,
        device=cfg.DEVICE, 
        seed=seed, 
        n_agents=4, 
        shared_rew=True, 
        reward="native",
        eps=0.1, 
        safety_distance=0.3, 
        spawn_spacing=0.4,
        lambda1=1.0, 
        lambda2=1 / 0.3, 
        mu2_cap=0.2,
    )
    return TransformedEnv(env, StepCounter(max_steps=cfg.MAX_STEPS))