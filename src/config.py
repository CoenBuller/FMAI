import torch.nn as nn

from typing import Literal
from dataclasses import dataclass
from torch.nn import Tanh, ReLU, Sigmoid



@dataclass
class Config:
    RESULTS_DIR: str = "results"
    RENDER: bool = False

    SCENARIO: str = "navigation"
    NUM_ENVS: int = 5
    DEVICE: str = "cuda"
    MAX_STEPS: int = 1000                                   # Defines the horizon. None is infinite horizon.
    EPISODES: int|float = 1e6                               # Number of episodes to train for
    SEED: None|int = 67                                     # Seed of the environment


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