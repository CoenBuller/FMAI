import torch

from torchrl.envs import StepCounter, TransformedEnv, Compose
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.transforms import Transform

from navigation_sfo import NavigationSFO


class RobustnessReward(Transform):
    """Adds the team robustness of `spec` to the reward at every step, computed on the GPU inside the env.

    Keeps the last `window + 1` next-observations per env (NaN where not yet available) and evaluates the
    spec on that window. Only the last step is used. Undefined lags are NaN, which `Defined` turns into -inf.
    """

    def __init__(self, spec, window, mode, coef, shared_rew=False):
        super().__init__(in_keys=[], out_keys=[])
        self.spec, self.window, self.mode, self.coef = spec, window, mode, coef
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

        new = rew + self.coef * rho_r if self.mode == "mixed" else rho_r
        next_tensordict.set(("agents", "reward"), new)
        return next_tensordict


def make_env(cfg, num_envs, seed, spec=None, mode="none", window=5):
    """NavigationSFO with native reward. If `spec` is given and mode is 'rho' or 'mixed', the robustness is
    added to the reward inside the env. The eval env is built without a spec, so its reward stays native."""
    env = VmasEnv(
        scenario=NavigationSFO(),
        num_envs=num_envs,
        continuous_actions=cfg.CONTINUOUS_ACTIONS,
        device=cfg.DEVICE,
        seed=seed,
        n_agents=cfg.NUM_AGENTS,
        shared_rew=cfg.SHARED_REW,
        reward="native",
        eps=0.1,
        safety_distance=0.3,
        spawn_spacing=0.4,
        lambda1=1.0,
        lambda2=1 / 0.3,
        mu2_cap=0.2,
    )
    transforms = [StepCounter(max_steps=cfg.MAX_STEPS)]              # time limit = truncation
    if spec is not None and mode in ("rho", "mixed"):
        rho_rew = RobustnessReward(
            spec=spec,
            window=window,
            mode=mode,
            coef=cfg.RHO_COEF,
            shared_rew=cfg.SHARED_REW,
        )
        transforms.append(rho_rew)
    return TransformedEnv(env, Compose(*transforms))

if __name__ == "__main__":
    from src.config import Config
    from src.fmai.utils import *

    cfg = Config()
    spec, _ = build_spec()
    env = make_env(cfg, spec=spec, mode="rho", window=W)
    td = env.rollout(150, break_when_any_done=False)                 # random policy
    obs = with_episode_ids(td["next", "agents", "observation"], td["next", "done"])
    off = spec(obs).mean(-1)                                          # [E, T]
    on = td["next", "agents", "reward"][..., 0, 0]                    # [E, T]
    print((off - on).abs().max())   