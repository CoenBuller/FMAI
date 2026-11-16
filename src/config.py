import torch.nn as nn

from typing import Literal
from dataclasses import dataclass
from torch.nn import Tanh, ReLU, Sigmoid

@dataclass
class Config:
    RESULTS_DIR: str = "results/luc"
    RENDER: bool = False
    DEVICE: str = "cuda"
    EVAL_INTERVAL: int = 100                                # After how many rollouts to do an evaluation run
    SEED: int = 1                                           # Seed of the environment
    EVAL_EPISODES: int = 10                                 # How many episodes for evaluation
    EPISODES: int|float = 1e6                               # Number of episodes to train for
    REWARD_MODE: str = "rho"                                # Agent can learn either using robustness (rho), using reward (rew), or mixed
    RHO_COEF: float = 1
    RUN_ID: str = ""
    HORIZON: int = 1000
    EVAL_GIF: bool = False
    STEPS: int = 0                                          # Training budget in env steps (frames). 0 -> EPISODES * MAX_STEPS
    SPEC: str = "coen"                                      # Training spec (sfo.build_spec) for REWARD_MODE rho / mixed
    EVAL_SPEC: str = "sfo"                                  # Eval robustness: "sfo" = spec-independent reference (GOAL_EPS, SAFETY_DIST), or a build_spec name
    N_SEEDS: int = 1                                        # Seeds SEED .. SEED + N_SEEDS - 1 are run per option (experiment.py)
    SMOOTH: int = 1                                         # Moving-average window (in eval points) for the compare plots; 1 = off
    PLOT_MODE: str = "diagnostic"                           # "diagnostic": per-run 6-panel PNG only. "compare": also one overlay figure + table over all options/seeds
    EARLY_STOP: bool = True                                 # End an episode once every agent is within GOAL_EPS of its goal
    GOAL_EPS: float = 0.1                                   # Goal radius for early stopping and the reference robustness (mu1)
    SAFETY_DIST: float = 0.3                                # Centre-to-centre safety distance for the reference robustness (mu2)

    #---------------------------------- Env Config ----------------------------------#
    SCENARIO: str = "navigation"
    NUM_ENVS: int = 5
    NUM_AGENTS: int = 4
    MAX_STEPS: int = 1000                                   # Defines the horizon. None is infinite horizon.
    CONTINUOUS_ACTIONS: bool = True                         # Defines if actions are continuous or discrete
    SHARED_REW: bool = False                                # Added flag for shared/common reward

    
    #---------------------------------- Agent Config ----------------------------------#
    
    # Agent architecture
    MAPPO: bool = False                                     # Set to True to use MAPPO. False defaults to IPPO
    DEPTH: int = 2                                          # Number of hidden layers
    N_NODES: int = 256                                      # Number of neurons per layer
    ACTIVATION: Literal["Tanh", "ReLU", "Sigmoid"] = "ReLU" # Activaton function

    # Helper to get activation
    def get_activation(self) -> nn.Module:
        activations = {
            "Tanh": Tanh,
            "ReLU": ReLU,
            "Sigmoid": Sigmoid,
        }
        return {k.lower(): v for k, v in activations.items()}[self.ACTIVATION.lower()]   # yaml has "ReLu"

    # PPO Loss
    CLIP_EPS: float = 0.2                                   # Clipping parameter for PPO loss
    ENTROPY_COEF: float = 1e-4                              # Entropy coeficient to regularize action probabilities
    GAMMA: float = 0.99                                     # Reward decay coefficient 
    LAMBDA: float =0.9                                      # Lambda parameter for Generalized Advantage Estimate (GAE)

    # Optimizer
    LR: float = 3e-4                                        # Learning rate
    EPOCHS: int = 10                                        # Parameter update epochs
    MINIBATCH: int = 20                                     # How many steps to update over at once

    def __post_init__(self):
        if self.MAPPO:
            self.SHARED_REW = True    # MAPPO implies a team reward
        if not self.STEPS:
            self.STEPS = int(self.EPISODES * self.MAX_STEPS)   # old yaml files only give EPISODES
    #---------------------------------- Retrieving variables ----------------------------------#
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