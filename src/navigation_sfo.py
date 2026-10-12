"""VMAS Navigation with SFO robustness terms, to compare the native reward (A) with a robustness reward (B).

Changes to the original scenario, applied in both conditions:
  - agents and goals spawn at least `spawn_spacing` apart, so the separation spec is satisfiable,
  - episodes never end early (only the time limit in train.py ends them),
  - info reports the native reward, collisions, positions and the SFO terms, whatever reward is trained on.

Per agent i the SFO spec is   F G (mu1_i >= 0)  and  G (mu2_i >= 0)   with
  mu1_i = eps - |p_i - g_i|                    (goal: >= 0 when within eps of the goal)
  mu2_i = min_{j != i} |p_i - p_j| - d         (separation: >= 0 when every other agent is at least d away)
"""
import torch
from vmas.scenarios.navigation import Scenario


class NavigationSFO(Scenario):
    def make_world(self, batch_dim, device, *, reward, eps, safety_distance, spawn_spacing, lambda1, lambda2, mu2_cap,
                   **kwargs):
        world = super().make_world(batch_dim, device, **kwargs)
        self.reward_type, self.eps, self.d = reward, eps, safety_distance
        self.lambda1, self.lambda2, self.mu2_cap = lambda1, lambda2, mu2_cap
        self.min_distance_between_entities = spawn_spacing
        return world

    def sfo_terms(self):
        """mu1 and mu2 for every agent, each of shape [envs, agents]."""
        pos = torch.stack([agent.state.pos for agent in self.world.agents], dim=1)
        goal = torch.stack([agent.goal.state.pos for agent in self.world.agents], dim=1)
        distances = torch.cdist(pos, pos) + 1e6 * torch.eye(pos.shape[1], device=pos.device)  # exclude self
        return self.eps - torch.linalg.vector_norm(pos - goal, dim=-1), distances.min(dim=-1).values - self.d

    def reward(self, agent):
        native = super().reward(agent)  # always computed, so info can report it in both conditions
        if self.reward_type == "native":
            return native
        if agent is self.world.agents[0]:  # once per step; every agent gets the team sum, like VMAS's shared_rew
            mu1, mu2 = self.sfo_terms()
            self.robust_rew = (self.lambda1 * mu1 + self.lambda2 * mu2.clamp(max=self.mu2_cap)).sum(dim=-1)
        return self.robust_rew

    def done(self):
        return torch.zeros(self.world.batch_dim, dtype=torch.bool, device=self.world.device)

    def info(self, agent):
        native = super().info(agent)
        mu1, mu2 = self.sfo_terms()
        i = self.world.agents.index(agent)
        return {"native_rew": native["pos_rew"] + native["final_rew"] + native["agent_collisions"],
                "collisions": native["agent_collisions"] / self.agent_collision_penalty,  # other agents touched
                "mu1": mu1[:, i], "mu2": mu2[:, i], "pos": agent.state.pos, "goal": agent.goal.state.pos}
