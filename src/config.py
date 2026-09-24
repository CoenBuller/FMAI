import dataclasses 

class Config:
    env_name: str = "balance""
    num_envs: int = "32"
    device: str = "cuda"
    continuous_actions: bool = True,
    wrapper: None|str = None,                   # One of: None, "rllib", "gym", "gymnasium", "gymnasium_vec"
    max_steps: None|int = None,                 # Defines the horizon. None is infinite horizon.
    seed: None|int = 67,                        # Seed of the environment
    dict_spaces: bool = False,                  # By default tuple spaces are used with each element in the tuple being an agent.
                                                # If dict_spaces=True, the spaces will become Dict with each key being the agent's name
    grad_enabled: bool = True,                  # If grad_enabled the simulator is differentiable and gradients can flow from output to input
    terminated_truncated: bool = False,         # If terminated_truncated the simulator will return separate `terminated` and `truncated` flags in the `done()`, `step()`, and `get_from_scenario()` functions instead of a single `done` flag
