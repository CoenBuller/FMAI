import torch.nn as nn

from typing import Literal
from dataclasses import dataclass
from torch.nn import Tanh, ReLU, Sigmoid

@dataclass
class Config:
    RESULTS_DIR: str = "results/second"
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

    #---------------------------------- Env Config ----------------------------------#
    SCENARIO: str = "navigation"
    NUM_ENVS: int = 5
    NUM_AGENTS: int = 4
    MAX_STEPS: int = 1000                                   # Defines the horizon. None is infinite horizon.
    CONTINUOUS_ACTIONS: bool = True                         # Defines if actions are continuous or discrete
    SHARED_REW: bool = False                                # Added flag for shared/common reward
    OBSERVE_PEERS: bool = False

    # Flocking configs 
    AGENT_DISTANCE: float = 0.15
    TARGET_RADIUS: float = 0.3                              # Central target radius at which the agents should flock around the target
    EPSILON_T: float = 0.05                                 # Band width of the target radius
    SPAWN_IN_BAND: bool = True                              # If the agents should spawn in the band around the target
    MIN_PROXIMITY: float = 0.15                             # Minimal proximity the agents should keep between them

    TARGET_REL_SPEED: float = 0.25                          # Target relative speed the agents should keep
    EPSILON_V: float = 0.05                                 # Tolerance band around target velocity
    V_MAX: float = 0.85                                      # Maximum speed of the agents 

    W_DIST: float = 1
    W_PROX: float = 1
    W_V_REL: float = 1
    W_V_MAX: float = 1

    WIND_STRENGTH: float = 0.08
    WIND_GUST_STRENGTH: float = 0.04
    RANDOM_WIND_DIRECTION: bool = True
    
    TARGET_SPEED_ACTION: float = 0.5
    
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
        return activations[self.ACTIVATION]    

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

    def __setitem__(self, key, value):
        self.__dict__[key] = value
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