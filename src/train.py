"""Train and compare multi-agent RL algorithms on the VMAS Balance scenario with TorchRL.

Run from the repository root:  uv run python src/train.py
Results: results/compare.csv, results/compare.png and one GIF per run (results/<algorithm>_seed<seed>.gif)
"""
import csv

import matplotlib.pyplot as plt
import torch
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import AdditiveGaussianModule, MultiAgentMLP, ProbabilisticActor, TanhDelta, TanhNormal
from torchrl.objectives import ClipPPOLoss, DDPGLoss, SACLoss, SoftUpdate, TD3Loss, ValueEstimators

# =============================== PARAMETERS ===============================
ALGORITHMS = ["mappo", "ippo", "maddpg", "matd3", "masac"]  # any of: mappo, ippo, maddpg, iddpg, matd3, itd3, masac, isac
SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "results"
SAVE_GIF = True        # GIF of one deterministic episode after training (needs a display; use xvfb-run on a server)
GIF_FPS = 30

# Environment
SCENARIO = "balance"   # VMAS scenario name, or a BaseScenario instance (e.g. Balance with an STL reward)
N_AGENTS = 3
MAX_STEPS = 100        # episode horizon
NUM_ENVS = 600         # parallel training environments
EVAL_NUM_ENVS = 200    # parallel evaluation environments (one deterministic episode each)

# Training budget, identical for every algorithm (counted in environment steps)
FRAMES_PER_BATCH = 60_000  # must be a multiple of NUM_ENVS
TOTAL_FRAMES = 6_000_000
EVAL_EVERY = 5             # evaluate every N iterations

# Networks
SHARE_PARAMS = True    # one network shared by all agents (False = one network per agent)
DEPTH = 2
NUM_CELLS = 256
LR = 3e-4
GAMMA = 0.99
MAX_GRAD_NORM = 1.0

# On-policy: MAPPO / IPPO
PPO_EPOCHS = 10
PPO_MINIBATCH_SIZE = 4096
CLIP_EPS = 0.2
GAE_LAMBDA = 0.9
ENTROPY_COEF = 1e-4

# Off-policy: MADDPG / IDDPG / MATD3 / ITD3 / MASAC / ISAC
BUFFER_SIZE = 1_000_000
OFF_POLICY_BATCH_SIZE = 4096
OFF_POLICY_UPDATES = 500   # gradient steps per collected batch
TAU = 0.005                # soft target-network update rate
NOISE_SIGMA_INIT = 0.9     # DDPG/TD3 Gaussian exploration noise, annealed linearly ...
NOISE_SIGMA_END = 0.1
NOISE_ANNEAL_FRAMES = TOTAL_FRAMES // 2  # ... over this many frames
TD3_POLICY_DELAY = 2       # TD3: actor updated once per this many critic updates
TD3_POLICY_NOISE = 0.2     # TD3: noise added to the target action ...
TD3_NOISE_CLIP = 0.5       # ... clipped to this range
# ==========================================================================

# algorithm -> (family, centralised critic)
ALGOS = {
    "mappo": ("ppo", True), "ippo": ("ppo", False),
    "maddpg": ("ddpg", True), "iddpg": ("ddpg", False),
    "matd3": ("td3", True), "itd3": ("td3", False),
    "masac": ("sac", True), "isac": ("sac", False),
}


def make_env(num_envs, seed):
    return VmasEnv(scenario=SCENARIO, num_envs=num_envs, continuous_actions=True, max_steps=MAX_STEPS,
                   device=DEVICE, seed=seed, n_agents=N_AGENTS)


def mlp(n_in, n_out, centralized):
    return MultiAgentMLP(n_agent_inputs=n_in, n_agent_outputs=n_out, n_agents=N_AGENTS, centralized=centralized,
                         share_params=SHARE_PARAMS, device=DEVICE, depth=DEPTH, num_cells=NUM_CELLS,
                         activation_class=torch.nn.Tanh)


def make_actor(env, family, obs_dim, act_dim):
    """Decentralised actor: each agent acts on its own observation."""
    spec = env.full_action_spec_unbatched
    bounds = {"low": spec[env.action_key].space.low, "high": spec[env.action_key].space.high}
    if family in ("ddpg", "td3"):  # deterministic policy
        net = TensorDictModule(mlp(obs_dim, act_dim, False), in_keys=[("agents", "observation")],
                               out_keys=[("agents", "param")])
        return ProbabilisticActor(net, spec=spec, in_keys=[("agents", "param")], out_keys=[env.action_key],
                                  distribution_class=TanhDelta, distribution_kwargs=bounds, return_log_prob=False)
    net = TensorDictModule(torch.nn.Sequential(mlp(obs_dim, 2 * act_dim, False), NormalParamExtractor()),
                           in_keys=[("agents", "observation")], out_keys=[("agents", "loc"), ("agents", "scale")])
    return ProbabilisticActor(net, spec=spec, in_keys=[("agents", "loc"), ("agents", "scale")],
                              out_keys=[env.action_key], distribution_class=TanhNormal,
                              distribution_kwargs=bounds, return_log_prob=True)


def make_critic(env, family, centralized, obs_dim, act_dim):
    """Centralised critic (MA*) sees all agents' inputs; independent critic (I*) only the agent's own."""
    if family == "ppo":  # state value V(o)
        return TensorDictModule(mlp(obs_dim, 1, centralized), in_keys=[("agents", "observation")],
                                out_keys=[("agents", "state_value")])
    cat = TensorDictModule(lambda obs, act: torch.cat([obs, act], dim=-1),  # action value Q(o, a)
                           in_keys=[("agents", "observation"), env.action_key], out_keys=[("agents", "obs_action")])
    q = TensorDictModule(mlp(obs_dim + act_dim, 1, centralized), in_keys=[("agents", "obs_action")],
                         out_keys=[("agents", "state_action_value")])
    return TensorDictSequential(cat, q)


def make_loss(env, family, actor, critic):
    keys = {"reward": env.reward_key, "done": ("agents", "done"), "terminated": ("agents", "terminated")}
    q_key = ("agents", "state_action_value")
    if family == "ppo":
        loss = ClipPPOLoss(actor, critic, clip_epsilon=CLIP_EPS, entropy_coeff=ENTROPY_COEF)
        loss.set_keys(value=("agents", "state_value"), sample_log_prob=("agents", "action_log_prob"),
                      action=env.action_key, **keys)
        loss.make_value_estimator(ValueEstimators.GAE, gamma=GAMMA, lmbda=GAE_LAMBDA)
        return loss
    if family == "ddpg":
        loss = DDPGLoss(actor, critic, loss_function="l2", delay_value=True)
        loss.set_keys(state_action_value=q_key, **keys)
    elif family == "td3":
        loss = TD3Loss(actor, critic, action_spec=env.full_action_spec_unbatched[env.action_key], num_qvalue_nets=2,
                       policy_noise=TD3_POLICY_NOISE, noise_clip=TD3_NOISE_CLIP, loss_function="l2")
        loss.set_keys(state_action_value=q_key, action=env.action_key, **keys)
    else:
        loss = SACLoss(actor, critic, num_qvalue_nets=2, loss_function="l2", delay_qvalue=True,
                       action_spec=env.full_action_spec_unbatched)
        loss.set_keys(state_action_value=q_key, log_prob=("agents", "action_log_prob"), action=env.action_key, **keys)
    loss.make_value_estimator(ValueEstimators.TD0, gamma=GAMMA)
    return loss


def expand_done(td):
    """VMAS gives one done flag per environment; the losses need one per agent, like the reward."""
    shape = td.get(("next", "agents", "reward")).shape
    for key in ("done", "terminated"):
        td.set(("next", "agents", key), td.get(("next", key)).unsqueeze(-1).expand(shape))


def evaluate(env, actor):
    """Team return, success rate and fall rate of the first deterministic episode in each evaluation env."""
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = env.rollout(MAX_STEPS, actor, break_when_any_done=False)
    first_done = td.get(("next", "done"))[..., 0].float().argmax(dim=1)  # step at which each first episode ended
    reward = td.get(("next", "agents", "reward"))[..., 0, 0]  # shared team reward, shape [envs, steps]
    returns = (reward * (torch.arange(MAX_STEPS, device=DEVICE) <= first_done[:, None])).sum(dim=1)
    fell = td.get(("next", "agents", "info", "ground_rew"))[torch.arange(len(first_done)), first_done, 0, 0] < 0
    success = ~fell & (first_done < MAX_STEPS - 1)  # ended before the time limit without falling: package at goal
    return returns.mean().item(), success.float().mean().item(), fell.float().mean().item()


def save_gif(env, actor, path):
    """Render evaluation environment 0 for one deterministic rollout."""
    frames = []
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        env.rollout(MAX_STEPS, actor, break_when_any_done=False,
                    callback=lambda env, td: frames.append(env._env.render(mode="rgb_array")))
    ImageSequenceClip(frames, fps=GIF_FPS).write_gif(path)


def train(algo, seed):
    family, centralized = ALGOS[algo]
    torch.manual_seed(seed)
    env, eval_env = make_env(NUM_ENVS, seed), make_env(EVAL_NUM_ENVS, seed + 1000)
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    act_dim = env.full_action_spec_unbatched[env.action_key].shape[-1]

    actor = make_actor(env, family, obs_dim, act_dim)
    critic = make_critic(env, family, centralized, obs_dim, act_dim)
    loss = make_loss(env, family, actor, critic)
    optim = torch.optim.Adam(loss.parameters(), lr=LR)

    behaviour = actor
    if family in ("ddpg", "td3"):  # deterministic actors explore with annealed Gaussian noise
        noise = AdditiveGaussianModule(spec=actor.spec, sigma_init=NOISE_SIGMA_INIT, sigma_end=NOISE_SIGMA_END,
                                       annealing_num_steps=NOISE_ANNEAL_FRAMES, action_key=env.action_key,
                                       device=DEVICE)
        behaviour = TensorDictSequential(actor, noise)
    on_policy = family == "ppo"
    if not on_policy:
        target_updater = SoftUpdate(loss, tau=TAU)

    collector = Collector(env, behaviour, frames_per_batch=FRAMES_PER_BATCH, total_frames=TOTAL_FRAMES,
                          device=DEVICE, storing_device=DEVICE)
    buffer = ReplayBuffer(storage=LazyTensorStorage(FRAMES_PER_BATCH if on_policy else BUFFER_SIZE, device=DEVICE),
                          sampler=SamplerWithoutReplacement() if on_policy else None,  # None = uniform random
                          batch_size=PPO_MINIBATCH_SIZE if on_policy else OFF_POLICY_BATCH_SIZE)
    n_updates = PPO_EPOCHS * (FRAMES_PER_BATCH // PPO_MINIBATCH_SIZE) if on_policy else OFF_POLICY_UPDATES

    log = [(algo, seed, 0, *evaluate(eval_env, actor))]
    for i, td in enumerate(collector):
        expand_done(td)
        if on_policy:  # advantages are computed once per batch, with the critic before the update
            with torch.no_grad():
                loss.value_estimator(td, params=loss.critic_network_params,
                                     target_params=loss.target_critic_network_params)
        buffer.extend(td.reshape(-1))

        for update in range(n_updates):
            losses = loss(buffer.sample())
            if family == "td3" and update % TD3_POLICY_DELAY != 0:  # TD3: delayed actor updates
                losses = losses.exclude("loss_actor")
            total = sum(value for key, value in losses.items() if key.startswith("loss_"))
            optim.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(loss.parameters(), MAX_GRAD_NORM)
            optim.step()
            if not on_policy:
                target_updater.step()

        if family in ("ddpg", "td3"):
            noise.step(FRAMES_PER_BATCH)
        collector.update_policy_weights_()

        if (i + 1) % EVAL_EVERY == 0:
            log.append((algo, seed, (i + 1) * FRAMES_PER_BATCH, *evaluate(eval_env, actor)))
            print("{:7s} seed {}  frames {:>10,}  return {:8.2f}  success {:.2f}  fell {:.2f}".format(*log[-1]))
    collector.shutdown()
    if SAVE_GIF:
        save_gif(eval_env, actor, f"{RESULTS_DIR}/{algo}_seed{seed}.gif")
    return log


if __name__ == "__main__":
    rows = [row for algo in ALGORITHMS for seed in SEEDS for row in train(algo, seed)]

    with open(f"{RESULTS_DIR}/compare.csv", "w", newline="") as f:
        csv.writer(f).writerows([("algorithm", "seed", "frames", "eval_return", "success_rate", "fall_rate"), *rows])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for column, ax, label in [(3, axes[0], "evaluation team return"), (4, axes[1], "success rate")]:
        for algo in ALGORITHMS:
            frames = [r[2] for r in rows if r[0] == algo and r[1] == SEEDS[0]]
            values = torch.tensor([[r[column] for r in rows if r[0] == algo and r[1] == s] for s in SEEDS])
            mean, std = values.mean(0), values.std(0, correction=0)
            ax.plot(frames, mean, label=algo)
            ax.fill_between(frames, mean - std, mean + std, alpha=0.2)
        ax.set_xlabel("environment frames")
        ax.set_ylabel(label)
    axes[0].legend()
    fig.suptitle(f"VMAS {SCENARIO}: mean ± std over {len(SEEDS)} seeds")
    fig.savefig(f"{RESULTS_DIR}/compare.png", dpi=150)