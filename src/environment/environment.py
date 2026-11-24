import torch

from torchrl.envs import StepCounter, TransformedEnv, Compose
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.transforms import Transform

from navigation_sfo import NavigationSFO

from vmas.scenarios.navigation import Scenario
from vmas.simulator.core import Agent

from environment.flocking import StableFlocking

class RobustnessReward(Transform):
    """Adds the team robustness of `spec` to the reward at every step, computed on the GPU inside the env.

    Keeps the last `window + 1` next-observations per env (NaN where not yet available) and evaluates the
    spec on that window. Only the last step is used. Undefined lags are NaN, which `Defined` turns into -inf.
    """

    def __init__(self, spec, window, shared_rew=False):
        super().__init__(in_keys=[], out_keys=[])
        self.spec = spec
        self.window = window
        self.hist = None                                             # [E, window+1, A, D]
        self.shared_rew = shared_rew

    def _reset(self, tensordict, tensordict_reset):
        if self.hist is not None:                                    # clear the history of the envs that reset
            mask = None if tensordict is None else tensordict.get("_reset", None)
            if mask is None:
                self.hist.fill_(float("nan"))
            else:
                self.hist[mask.reshape(self.hist.shape[0], -1).any(-1)] = float("nan")
        return tensordict_reset

    def _step(self, tensordict, next_tensordict):
        obs = next_tensordict.get(("agents", "observation"))         # [E, A, D]
        if self.hist is None or self.hist.shape[0] != obs.shape[0]:
            self.hist = torch.full((obs.shape[0], self.window + 1, *obs.shape[1:]),
                                   float("nan"), device=obs.device, dtype=obs.dtype)
        self.hist = torch.roll(self.hist, -1, dims=1)
        self.hist[:, -1] = obs

        o = torch.cat([self.hist, torch.zeros_like(self.hist[..., :1])], dim=-1)    # zero episode-id channel
        rho = self.spec(o)[:, -1]                                                   # [E, A]: margins at the current step

        rew = next_tensordict.get(("agents", "reward"))              # [E, A, 1]
        if self.shared_rew:
            rho_r = rho.mean(-1)[:, None, None].expand_as(rew)       # team value, same for every agent
        else:
            rho_r = rho[..., None].expand_as(rew)

        next_tensordict.set(("agents", "reward"), rho_r)
        return next_tensordict


def make_env(
    cfg,
    num_envs=None,
    seed=None,
    spec=None, 
    mode=None,
    window=None
    ):

    """NavigationSFO with native reward. If `spec` is given and mode is 'rho', the robustness is
    added to the reward inside the env. The eval env is built without a spec, so its reward stays native."""

    # Use Config defaults while allowing callers (e.g. evaluation) to override them.
    if num_envs is None:
        num_envs = cfg.NUM_ENVS
    if seed is None:
        seed = cfg.SEED
    if mode is None:
        mode = cfg.REWARD_MODE
    if window is None:
        window = cfg.TAU_FLOCK

    if cfg["SCENARIO"] == "navigation":
        env = VmasEnv(
            scenario=NavigationSFO(),
            num_envs=num_envs,
            continuous_actions=cfg.CONTINUOUS_ACTIONS,
            device=cfg.DEVICE,
            seed=seed,
            n_agents=cfg.NUM_AGENTS,
            shared_rew=cfg.SHARED_REW,
            eps=cfg.GOAL_EPS,
            safety_distance=cfg.SAFETY_DIST,
            n_zones=cfg.N_ZONES,                                     # danger zones + inertia (navigation_sfo.py)
            zone_radius=cfg.ZONE_RADIUS,
            zone_lidar_range=cfg.ZONE_LIDAR_RANGE,
            cat_prob=cfg.CAT_PROB,
            cat_penalty=cfg.CAT_PENALTY,
            agent_mass=cfg.AGENT_MASS,
            drag=cfg.DRAG,
            spawn_spacing=0.4,
        )

    elif cfg["SCENARIO"] == "flocking":
        env = VmasEnv(
            scenario=StableFlocking(),
            num_envs=num_envs,
            continuous_actions=cfg.CONTINUOUS_ACTIONS,
            device=cfg.DEVICE,
            seed=seed,
            n_agents=cfg.NUM_AGENTS,
            min_dist_between_entities=cfg.AGENT_DISTANCE,
            n_lidar_rays=cfg.N_LIDAR_RAYS,
            lidar_max_range=cfg.LIDAR_MAX_RANGE,
            target_radius=cfg.TARGET_RADIUS,
            epsilon_t=cfg.EPSILON_T,
            spawn_in_band=cfg.SPAWN_IN_BAND,
            flat_band_bonus=cfg.FLAT_BAND_BONUS,
            observe_peers=cfg.OBSERVE_PEERS,
            min_proximity=cfg.MIN_PROXIMITY,
            target_rel_speed=cfg.TARGET_REL_SPEED,
            epsilon_v=cfg.EPSILON_V,
            v_max=cfg.V_MAX,
            w_dist=cfg.W_DIST,
            w_prox=cfg.W_PROX,
            w_v_rel=cfg.W_V_REL,
            w_v_max=cfg.W_V_MAX,
            wind_strength=cfg.WIND_STRENGTH,
            wind_gust_strength=cfg.WIND_GUST_STRENGTH,
            random_wind_direction=cfg.RANDOM_WIND_DIRECTION,
            target_speed_action=cfg.TARGET_SPEED_ACTION,
            target_turn_rate=cfg.TARGET_TURN_RATE,
            target_boundary_margin=cfg.TARGET_BOUNDARY_MARGIN,
        )

        
    transforms = [StepCounter(max_steps=cfg.MAX_STEPS)]              # time limit = truncation
    if spec is not None and mode == "rho":
        rho_rew = RobustnessReward(spec=spec, window=window, shared_rew=cfg.SHARED_REW)
        transforms.append(rho_rew)
    return TransformedEnv(env, Compose(*transforms))
