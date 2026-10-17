import torch, torch.nn as nn

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss, ValueEstimators



class Agent():
    def __init__(self):
        self.policy = None
        self.loss = None
        self.optim = None
        self.actor_net = None

    def make_loss(self, env, actor, critic, cfg):

        loss = ClipPPOLoss(
            actor, 
            critic, 
            clip_epsilon=cfg.CLIP_EPS, 
            entropy_coeff=cfg.ENTROPY_COEF, 
            normalize_advantage=True
            )
        
        loss.set_keys(
            reward=env.reward_key, 
            action=env.action_key, 
            value=("agents", "state_value"),
            sample_log_prob=("agents", "action_log_prob"), 
            done=("agents", "done"),
            terminated=("agents", "terminated")
            )
        
        loss.make_value_estimator(
            ValueEstimators.GAE, 
            gamma=cfg.GAMMA, 
            lmbda=cfg.LAMBDA
            )
        
        return loss

    def make_agent(self, env, cfg):

        A = env.n_agents
        obs_dim = env.observation_spec["agents", "observation"].shape[-1]
        act_dim = env.full_action_spec[env.action_key].shape[-1]
        act_spec = env.full_action_spec_unbatched

        actor_net = nn.Sequential(
            MultiAgentMLP(
                obs_dim, 
                2 * act_dim, 
                n_agents=A, 
                centralised=False,
                share_params=True, 
                depth=cfg.DEPTH, 
                num_cells=cfg.N_NODES,
                activation_class=cfg.get_activation(), 
                device=cfg.DEVICE
                ),
            NormalParamExtractor(),
        )

        policy = ProbabilisticActor(
            module=TensorDictModule(
                actor_net, 
                in_keys=[("agents", "observation")],
                out_keys=[("agents", "loc"), ("agents", "scale")]
                ),
            spec=act_spec,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[env.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": act_spec[env.action_key].space.low,
                                "high": act_spec[env.action_key].space.high},
            return_log_prob=True,
        )

        critic = ValueOperator(
            MultiAgentMLP(
                obs_dim, 
                1, 
                n_agents=A, 
                centralised=cfg.MAPPO,   # if we want to run MAPPO we set cfg.MAPPO=True as it needs a centralized critic. cfg.MAPPO=False for IPPO
                share_params=True, 
                depth=cfg.DEPTH, 
                num_cells=cfg.N_NODES,
                activation_class=cfg.get_activation(), 
                device=cfg.DEVICE
                ),
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "state_value")],
        )


        loss_module = self.make_loss(env=env, actor=policy, critic=critic, cfg=cfg)
        optim = torch.optim.AdamW(loss_module.parameters(), lr=cfg.LR)

        self.policy = policy 
        self.loss_module = loss_module
        self.optim = optim
        self.actor_net = actor_net