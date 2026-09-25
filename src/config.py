from dataclasses import dataclass

@dataclass
class Config:
    results_dir: str = "results"
    render: bool = False
    n_steps: int = 600

    scenario: str = "balance"
    num_envs: int = 4096
    device: str = "cuda"
    continuous_actions: bool = True
    wrapper: None|str = None                   # One of: None, "rllib", "gym", "gymnasium", "gymnasium_vec"
    max_steps: None|int = None                 # Defines the horizon. None is infinite horizon.
    seed: None|int = 67                        # Seed of the environment
    dict_spaces: bool = False                  # By default tuple spaces are used with each element in the tuple being an agent.
                                                # If dict_spaces=True, the spaces will become Dict with each key being the agent's name
    grad_enabled: bool = True                  # If grad_enabled the simulator is differentiable and gradients can flow from output to input
    terminated_truncated: bool = False         # If terminated_truncated the simulator will return separate `terminated` and `truncated` flags in the `done()`, `step()`, and `get_from_scenario()` functions instead of a single `done` flag

    @property
    def variables(self):
        return self.get_variables()

    @property
    def attributes(self):
        return self.get_attributes()
    
    @property
    def values(self):
        return self.get_values()

    def get_variables(self):
        return {key: value for (key, value) in vars(self).items()}

    def get_attributes(self):
        return [key for (key, value) in vars(self).items()]
    
    def get_values(self):
        return [value for (key, value) in vars(self).items()]

if __name__ == "__main__":
    print(Config().variables)       # Retrieves all attributes, value pairs of class instance
    print(Config().values)          # Retrieves all values of class instance
    print(Config().attributes)      # Retrieves all attributes of class instance