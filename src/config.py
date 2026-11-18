import math
import torch.nn as nn

from typing import Literal
from dataclasses import dataclass
from torch.nn import Tanh, ReLU, Sigmoid


@dataclass
class Config:
    RESULTS_DIR: str = "results/flocking"
    RENDER: bool = False
    DEVICE: str = "cuda"
    EVAL_INTERVAL: int = 20                                 # Evaluate every 20 collector rollouts
    SEED: int = 1                                           # Seed of the environment
    EVAL_EPISODES: int = 10                                 # How many parallel episodes for evaluation
    EPISODES: int | float = 50_000                          # Total training episodes
    REWARD_MODE: str = "rho"                                # 'rho' (SFO robustness), 'none' (native reward), or 'mixed'
    RHO_COEF: float = 1.0
    RUN_ID: str = ""
    HORIZON: int = 200
    EVAL_GIF: bool = False

    #---------------------------------- Env Config ----------------------------------#
    SCENARIO: str = "flocking"
    NUM_ENVS: int = 32                                      # Parallel vectorized environments on GPU
    NUM_AGENTS: int = 4                                     # Default swarm size (scales up to N_MAX = 6)
    MAX_STEPS: int = 200                                    # Episode length (200 steps = ~2 full orbits)
    CONTINUOUS_ACTIONS: bool = True                         # Continuous 2D thrust forces
    SHARED_REW: bool = False                                # Individual agent robustness by default (True if MAPPO)
    OBSERVE_PEERS: bool = False                         # Include peer Lidar readings in observations
    N_LIDAR_RAYS: int = 12                              # Number of Lidar rays per agent
    FLAT_BAND_BONUS: bool = False                       # Use a flat distance bonus inside the orbital band

    #---------------------------------- Flocking SFO Configs ------------------------#
    # 1. Geometry & Orbital Band:
    #    Scaled for max swarm size N_max = 6: r_t = eps_a / (2 * sin(pi / 6)) + eps_T + 0.05
    #    With eps_a = 0.20 and eps_T = 0.05 -> r_t = 0.20 / 1.0 + 0.05 + 0.05 = 0.30
    AGENT_DISTANCE: float = 0.20                            # epsilon_a: minimum pairwise center-to-center separation
    EPSILON_T: float = 0.05                                 # epsilon_T: orbital band half-width
    TARGET_RADIUS: float = 0.30                             # r_t: central orbital radius around moving target
    SPAWN_IN_BAND: bool = True                              # Spawn agents inside [r_t - eps_T, r_t + eps_T] at t=0

    # 2. Lidar & Collision Avoidance (phi_c):
    #    With lidar_max_range = 0.35 and agent radius = 0.05, center-to-center distance eps_a = 0.20
    #    equals surface-to-surface clearance 0.10 -> normalized proximity p_t = 1 - (0.20 / 0.35) ≈ 0.45
    LIDAR_MAX_RANGE: float = 0.35
    MIN_PROXIMITY: float = 2/3                              # p_t: max allowed normalized Lidar proximity in [0, 1]

    # 3. Velocity & Target Dynamics (phi_v^m & phi_s^v):
    #    target_speed_action = 0.30 under VMAS damping (0.25) yields steady target speed v_t ≈ 0.20.
    #    v_max = v_r + eps_v^rel + v_t + 0.15 = 0.20 + 0.05 + 0.20 + 0.15 = 0.60
    TARGET_SPEED_ACTION: float = 0.20                       # Target thrust (produces steady speed v_t ≈ 0.20)
    TARGET_TURN_RATE: float = 0.15                          # Smooth random-walk angular drift rate
    TARGET_BOUNDARY_MARGIN: float = 0.60                    # Soft arena boundary for target random walk
    TARGET_REL_SPEED: float = 0.20                          # v_r: desired steady relative speed to target
    EPSILON_V: float = 0.05                                 # epsilon_v^rel: tolerance band around v_r
    V_MAX: float = 0.60                                     # v_max: strict global speed ceiling

    # 4. Temporal Recovery Horizon & Wind Perturbations (phi_s^d & phi_s^v):
    TAU_FLOCK: int = 3                                      # Bounded recovery horizon tau_d = tau = 3 steps
    WIND_STRENGTH: float = 0.08                             # Base steady wind force
    WIND_GUST_STRENGTH: float = 0.03                        # Spatially/temporally varying gust amplitude
    RANDOM_WIND_DIRECTION: bool = True

    # 5. Native Reward Weights (used when REWARD_MODE is 'none' or 'mixed'):
    W_DIST: float = 1.0
    W_PROX: float = 1.0
    W_V_REL: float = 1.0
    W_V_MAX: float = 1.0

    #---------------------------------- Agent Config --------------------------------#
    # Agent architecture
    MAPPO: bool = False                                     # Set to True to use MAPPO. False defaults to IPPO
    DEPTH: int = 2                                          # Number of hidden layers
    N_NODES: int = 256                                      # Number of neurons per layer
    ACTIVATION: Literal["Tanh", "ReLU", "Sigmoid"] = "Tanh" # Tanh works smoothly for bounded continuous control

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
    ENTROPY_COEF: float = 1e-4                              # Entropy coefficient to regularize action probabilities
    GAMMA: float = 0.99                                     # Reward decay coefficient
    LAMBDA: float = 0.9                                     # Lambda parameter for Generalized Advantage Estimate (GAE)

    # Optimizer
    LR: float = 3e-4                                        # Learning rate
    EPOCHS: int = 10                                        # Parameter update epochs per rollout
    MINIBATCH: int = 512                                    # Frames per minibatch (with 32 envs * 200 steps = 6400 frames)

    def __post_init__(self):
        if self.MAPPO:
            self.SHARED_REW = True                          # MAPPO implies a team reward

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    #---------------------------------- Retrieving variables ------------------------#
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
    print(Config().variables)