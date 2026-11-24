"""MAPPO on VMAS Navigation: native reward (A) vs SFO robustness reward (B), evaluated with the same metrics.

Run from the repository root:
    uv run python src/train.py                  # every reward and seed below, PARALLEL_RUNS at a time
    uv run python src/train.py robustness 0     # a single run
    uv run python src/analyze.py                # figures from all finished runs
Per run, in results/: <reward>_seed<seed>.csv (metrics per evaluation), _traces.npz (final evaluation, per step),
.pt (trained actor), .gif (one evaluation episode) and .log (printed progress).
"""
import csv
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import StepCounter, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from navigation_sfo import NavigationSFO

# =============================== PARAMETERS ===============================
REWARDS = ["native", "robustness"]  # A: VMAS's own reward, B: SFO robustness reward
SEEDS = [0, 1, 2]
PARALLEL_RUNS = 2      # runs trained at the same time, each its own process; lower it if memory runs out
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "results"
SAVE_GIF = True
GIF_FPS = 30

# Environment: VMAS Navigation, shared reward in both conditions
N_AGENTS = 4
MAX_STEPS = 150        # every episode lasts exactly this long
NUM_ENVS = 300         # parallel training environments
EVAL_NUM_ENVS = 200    # parallel evaluation environments (one deterministic episode each)
SPAWN_SPACING = 0.4    # minimum distance between agents and goals at the start

# SFO spec per agent i:  F G (|p_i - g_i| <= EPS)  and  G (for all j != i: |p_i - p_j| >= SAFETY_DISTANCE)
EPS = 0.1
SAFETY_DISTANCE = 0.3
LAMBDA1 = 1.0                    # robustness reward = sum over agents of LAMBDA1 * mu1 + LAMBDA2 * min(mu2, MU2_CAP)
LAMBDA2 = 1 / SAFETY_DISTANCE
MU2_CAP = 0.2                    # more separation than this earns nothing extra

# Training budget (counted in environment steps)
FRAMES_PER_BATCH = NUM_ENVS * MAX_STEPS
TOTAL_FRAMES = 5_000_000
EVAL_EVERY = 5         # evaluate every N iterations (and after the last one)

# MAPPO
SHARE_PARAMS = True    # one network shared by all agents
DEPTH = 2
NUM_CELLS = 256
LR = 3e-4
GAMMA = 0.99
MAX_GRAD_NORM = 1.0
PPO_EPOCHS = 10
PPO_MINIBATCH_SIZE = 4096
CLIP_EPS = 0.2
GAE_LAMBDA = 0.95
ENTROPY_COEF = 1e-4
# ==========================================================================


def make_env(num_envs, seed, reward):
    env = VmasEnv(scenario=NavigationSFO(), num_envs=num_envs, continuous_actions=True, device=DEVICE, seed=seed,
                  n_agents=N_AGENTS, shared_rew=True, reward=reward, eps=EPS, safety_distance=SAFETY_DISTANCE,
                  spawn_spacing=SPAWN_SPACING, lambda1=LAMBDA1, lambda2=LAMBDA2, mu2_cap=MU2_CAP)
    return TransformedEnv(env, StepCounter(max_steps=MAX_STEPS))  # time limit = truncation: the critic looks past it


def mlp(n_in, n_out, centralized):
    return MultiAgentMLP(n_agent_inputs=n_in, n_agent_outputs=n_out, n_agents=N_AGENTS, centralized=centralized,
                         share_params=SHARE_PARAMS, device=DEVICE, depth=DEPTH, num_cells=NUM_CELLS,
                         activation_class=torch.nn.Tanh)


def make_actor(env, obs_dim, act_dim):
    """Decentralised actor: each agent acts on its own observation."""
    spec = env.full_action_spec_unbatched
    net = TensorDictModule(torch.nn.Sequential(mlp(obs_dim, 2 * act_dim, False), NormalParamExtractor()),
                           in_keys=[("agents", "observation")], out_keys=[("agents", "loc"), ("agents", "scale")])
    return ProbabilisticActor(net, spec=spec, in_keys=[("agents", "loc"), ("agents", "scale")],
                              out_keys=[env.action_key], distribution_class=TanhNormal, return_log_prob=True,
                              distribution_kwargs={"low": spec[env.action_key].space.low,
                                                   "high": spec[env.action_key].space.high})


def make_critic(obs_dim):
    """Centralised critic: sees every agent's observation."""
    return TensorDictModule(mlp(obs_dim, 1, True), in_keys=[("agents", "observation")],
                            out_keys=[("agents", "state_value")])


def make_loss(env, actor, critic):
    loss = ClipPPOLoss(actor, critic, clip_epsilon=CLIP_EPS, entropy_coeff=ENTROPY_COEF, normalize_advantage=True)
    loss.set_keys(reward=env.reward_key, action=env.action_key, value=("agents", "state_value"),
                  sample_log_prob=("agents", "action_log_prob"), done=("agents", "done"),
                  terminated=("agents", "terminated"))
    loss.make_value_estimator(ValueEstimators.GAE, gamma=GAMMA, lmbda=GAE_LAMBDA)
    return loss


def expand_done(td):
    """VMAS gives one done flag per environment; the loss needs one per agent, like the reward."""
    shape = td.get(("next", "agents", "reward")).shape
    for key in ("done", "terminated"):
        td.set(("next", "agents", key), td.get(("next", key)).unsqueeze(-1).expand(shape))


def evaluate(env, actor):
    """One deterministic episode per evaluation env. Returns summary metrics and the per-step traces."""
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = env.rollout(MAX_STEPS, actor)
    traces = {key: value.squeeze(-1) for key, value in td["next", "agents", "info"].items()}  # [envs, steps, agents(, 2)]
    traces["reward"] = td["next", "agents", "reward"].squeeze(-1)
    mu1, mu2 = traces["mu1"], traces["mu2"]
    rho_reach = mu1[:, -1]                    # F G goal: the margin at the end of the episode     [envs, agents]
    rho_safe = mu2.min(dim=1).values          # G separation: the worst margin over the episode    [envs, agents]
    rho = torch.minimum(rho_reach, rho_safe)  # per-agent SFO robustness
    rho_team = rho.min(dim=-1).values         # for all agents
    all_at_goal = (mu1 >= 0).all(dim=-1)      # [envs, steps]
    time_to_goal = torch.where(all_at_goal.any(dim=1), all_at_goal.float().argmax(dim=1) + 1, MAX_STEPS)
    metrics = {
        "train_return": traces["reward"].sum(dim=1).mean(),        # per agent, in the reward it was trained on
        "native_return": traces["native_rew"].sum(dim=1).mean(),   # per agent, in VMAS's own reward
        "robustness_team": rho_team.mean(),
        "robustness_agent": rho.mean(),
        "robustness_goal": rho_reach.mean(),
        "robustness_separation": rho_safe.mean(),
        "spec_satisfied": (rho_team >= 0).float().mean(),          # whole team spec holds
        "success_rate": all_at_goal[:, -1].float().mean(),         # every agent at its goal at the end
        "violation_rate": (rho_safe.min(dim=-1).values < 0).float().mean(),  # some pair ever closer than d
        "collision_steps": traces["collisions"].sum(dim=(1, 2)).mean() / 2,  # touching pairs, summed over steps
        "min_separation": (rho_safe.min(dim=-1).values + SAFETY_DISTANCE).mean(),
        "time_to_goal": time_to_goal.float().mean(),
    }
    return {key: value.item() for key, value in metrics.items()}, traces


def save_gif(env, actor, path):
    """Draw evaluation environment 0 during one deterministic episode; dashed circles touch at the safety distance."""
    fig, ax = plt.subplots(figsize=(5, 5))
    frames = []

    def draw(env, td):
        ax.clear()
        ax.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), aspect="equal")
        ax.axis("off")
        world = env.base_env._env.world
        for agent in world.agents:
            (x, y), (gx, gy) = agent.state.pos[0].tolist(), agent.goal.state.pos[0].tolist()
            ax.add_patch(plt.Circle((gx, gy), agent.goal.shape.radius, color=agent.color, alpha=0.5))
            ax.add_patch(plt.Circle((x, y), agent.shape.radius, color=agent.color))
            ax.add_patch(plt.Circle((x, y), SAFETY_DISTANCE / 2, color=agent.color, fill=False, linestyle="--"))
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        env.rollout(MAX_STEPS, actor, callback=draw)
    plt.close(fig)
    ImageSequenceClip(frames, fps=GIF_FPS).write_gif(path, logger=None)


def save_log(log, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)


def train(reward, seed):
    torch.manual_seed(seed)
    env, eval_env = make_env(NUM_ENVS, seed, reward), make_env(EVAL_NUM_ENVS, seed + 1000, reward)
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    act_dim = env.full_action_spec_unbatched[env.action_key].shape[-1]

    actor, critic = make_actor(env, obs_dim, act_dim), make_critic(obs_dim)
    loss = make_loss(env, actor, critic)
    optim = torch.optim.Adam(loss.parameters(), lr=LR)
    collector = Collector(env, actor, frames_per_batch=FRAMES_PER_BATCH, total_frames=TOTAL_FRAMES,
                          device=DEVICE, storing_device=DEVICE)
    buffer = ReplayBuffer(storage=LazyTensorStorage(FRAMES_PER_BATCH, device=DEVICE),
                          sampler=SamplerWithoutReplacement(), batch_size=PPO_MINIBATCH_SIZE)

    name, start = f"{RESULTS_DIR}/{reward}_seed{seed}", time.time()

    def log_evaluation(frames):
        metrics, traces = evaluate(eval_env, actor)
        log.append({"reward": reward, "seed": seed, "frames": frames, "minutes": (time.time() - start) / 60, **metrics})
        save_log(log, f"{name}.csv")  # saved at every evaluation, so a crash or timeout keeps the curve so far
        print("{reward:10s} seed {seed}  frames {frames:>9,}  native {native_return:7.2f}  "
              "robustness {robustness_team:6.3f}  success {success_rate:.2f}  violations {violation_rate:.2f}  "
              "[{minutes:.1f} min]".format(**log[-1]), flush=True)
        return traces

    log = []
    log_evaluation(0)
    for i, td in enumerate(collector):
        expand_done(td)
        with torch.no_grad():  # advantages are computed once per batch, with the critic before the update
            loss.value_estimator(td, params=loss.critic_network_params,
                                 target_params=loss.target_critic_network_params)
        buffer.extend(td.reshape(-1))
        for _ in range(PPO_EPOCHS * (FRAMES_PER_BATCH // PPO_MINIBATCH_SIZE)):
            losses = loss(buffer.sample())
            total = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
            optim.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(loss.parameters(), MAX_GRAD_NORM)
            optim.step()
        collector.update_policy_weights_()

        frames = (i + 1) * FRAMES_PER_BATCH
        if (i + 1) % EVAL_EVERY == 0 or frames >= TOTAL_FRAMES:
            traces = log_evaluation(frames)
    collector.shutdown()

    np.savez(f"{name}_traces.npz", **{key: value.cpu().numpy() for key, value in traces.items()})
    torch.save(actor.state_dict(), f"{name}.pt")
    if SAVE_GIF:
        save_gif(eval_env, actor, f"{name}.gif")


if __name__ == "__main__":
    if len(sys.argv) == 3:  # a single run: python src/train.py <reward> <seed>
        train(sys.argv[1], int(sys.argv[2]))
    else:  # every run as its own process, PARALLEL_RUNS at a time, each printing to results/<reward>_seed<seed>.log
        todo = [(reward, seed) for reward in REWARDS for seed in SEEDS]
        for start in range(0, len(todo), PARALLEL_RUNS):
            runs = [subprocess.Popen([sys.executable, __file__, reward, str(seed)], stderr=subprocess.STDOUT,
                                     stdout=open(f"{RESULTS_DIR}/{reward}_seed{seed}.log", "w"))
                    for reward, seed in todo[start:start + PARALLEL_RUNS]]
            for run in runs:
                run.wait()