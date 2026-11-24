"""VMAS Navigation with SFO robustness terms, to compare the native reward (A) with a robustness reward (B).

Changes to the original scenario, applied in both conditions:
  - agents and goals spawn at least `spawn_spacing` apart, so the separation spec is satisfiable,
  - episodes never end early (only the time limit in train.py ends them),
  - info reports the native reward, collisions, positions and the SFO terms, whatever reward is trained on.

Optional danger zones (n_zones > 0) and inertia (agent_mass, drag):
  - n_zones circles of radius zone_radius per episode, placed on the straight spawn->goal path of randomly chosen
    agents (never on a spawn or goal). Agents can drive through them: they are not obstacles.
  - Each step inside a zone, an agent suffers a catastrophe with probability cat_prob: native reward -cat_penalty.
    There is no other signal about zones in the native reward.
  - Agents see zones through a second LiDAR (n_lidar_rays rays, range zone_lidar_range) appended to the observation;
    inside a zone every ray reads distance 0.
  - agent_mass and drag set the inertia; terminal speed scales with 1 / (mass * drag), so mass 2.5 with drag 0.1 keeps
    the default top speed (1 * 0.25) with about 2.5x the response time.

Per agent i the SFO spec is   F G (mu1_i >= 0)  and  G (mu2_i >= 0)   with
  mu1_i = eps - |p_i - g_i|                    (goal: >= 0 when within eps of the goal)
  mu2_i = min_{j != i} |p_i - p_j| - d         (separation: >= 0 when every other agent is at least d away)
and with zones also  G (mu3_i >= 0),  mu3_i = min_k |p_i - z_k| - r   (outside every zone).
"""
import math
import torch
from vmas.scenarios.navigation import Scenario

FAR = 100.0     # position of an unused zone (no valid placement found)


class NavigationSFO(Scenario):
    def make_world(self, batch_dim, device, *, reward, eps, safety_distance, spawn_spacing, lambda1, lambda2, mu2_cap,
                   n_zones=0, zone_radius=0.25, zone_lidar_range=0.6, cat_prob=0.0,
                   cat_penalty=0.0, agent_mass=1.0, drag=0.25, **kwargs):
        world = super().make_world(batch_dim, device, **kwargs)
        self.reward_type, self.eps, self.d = reward, eps, safety_distance
        self.lambda1, self.lambda2, self.mu2_cap = lambda1, lambda2, mu2_cap
        self.min_distance_between_entities = spawn_spacing

        self.n_zones, self.zone_r, self.zone_range = n_zones, zone_radius, zone_lidar_range
        self.cat_prob, self.cat_penalty = cat_prob, cat_penalty
        self.zone_pos = torch.full((batch_dim, n_zones, 2), FAR, device=device)
        angles = torch.arange(self.n_lidar_rays, device=device) * (2 * math.pi / self.n_lidar_rays)
        self.ray_dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)          # [M, 2]
        for agent in world.agents:
            agent.mass = agent_mass
            agent._drag = drag                                                              # per-agent drag (VMAS)
            agent.catastrophe = torch.zeros(batch_dim, device=device)
        return world

    def reset_world_at(self, env_index=None):
        super().reset_world_at(env_index)
        if self.n_zones:
            self._place_zones(env_index)

    def _place_zones(self, env_index, n_cand=64):
        """Zone centres on the spawn->goal segment (35-65 %) of random agents, at least zone_radius + 0.15 from every
        spawn and goal and 2 * zone_radius + 0.1 from the other zones; uniform in the arena if that fails."""
        pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)                # [B, A, 2]
        goal = torch.stack([a.goal.state.pos for a in self.world.agents], dim=1)
        if env_index is not None:
            pos, goal = pos[env_index:env_index + 1], goal[env_index:env_index + 1]
        B, A, dev = pos.shape[0], pos.shape[1], pos.device
        points = torch.cat([pos, goal], dim=1)                                             # [B, 2A, 2]
        zones = torch.full((B, self.n_zones, 2), FAR, device=dev)
        for k in range(self.n_zones):
            a = torch.randint(A, (B, n_cand), device=dev)
            s = torch.rand(B, n_cand, 1, device=dev) * 0.3 + 0.35
            p0 = torch.gather(pos, 1, a[..., None].expand(-1, -1, 2))
            p1 = torch.gather(goal, 1, a[..., None].expand(-1, -1, 2))
            on_path = p0 + s * (p1 - p0) + 0.05 * torch.randn(B, n_cand, 2, device=dev)
            uniform = (torch.rand(B, n_cand, 2, device=dev) * 2 - 1) * self.world_spawning_x
            cand = torch.cat([on_path, uniform], dim=1)                                    # path candidates first
            ok = (torch.cdist(cand, points) > self.zone_r + 0.15).all(-1)
            ok &= (torch.cdist(cand, zones) > 2 * self.zone_r + 0.1).all(-1)
            first = torch.where(ok.any(1), ok.float().argmax(1), torch.zeros_like(ok[:, 0], dtype=torch.long))
            chosen = cand[torch.arange(B, device=dev), first]
            zones[:, k] = torch.where(ok.any(1, keepdim=True), chosen, torch.full_like(chosen, FAR))
        if env_index is None:
            self.zone_pos = zones
        else:
            self.zone_pos[env_index] = zones[0]

    def zone_dist(self, agent):
        """Signed distance of the agent to the nearest zone boundary (< 0 inside), shape [envs]."""
        if not self.n_zones:
            return torch.full((self.world.batch_dim,), FAR, device=self.world.device)
        return (torch.linalg.vector_norm(agent.state.pos[:, None] - self.zone_pos, dim=-1) - self.zone_r).amin(-1)

    def zone_lidar(self, agent):
        """Distance along each ray to the nearest zone (0 inside a zone, zone_range if none) -> [envs, rays]."""
        w = self.zone_pos[:, :, None] - agent.state.pos[:, None, None]                     # [B, K, 1, 2]
        b = (w * self.ray_dirs).sum(-1)                                                     # [B, K, M]
        c = (w ** 2).sum(-1) - self.zone_r ** 2                                             # [B, K, 1]
        disc = b ** 2 - c
        s = b - torch.sqrt(disc.clamp(min=0))
        hit = (disc >= 0) & (s >= 0)
        dist = torch.where(c < 0, torch.zeros_like(s), torch.where(hit, s, torch.full_like(s, self.zone_range)))
        return dist.amin(1).clamp(max=self.zone_range)

    def observation(self, agent):
        obs = super().observation(agent)
        if self.n_zones:
            obs = torch.cat([obs, self.zone_range - self.zone_lidar(agent)], dim=-1)
        return obs

    def sfo_terms(self):
        """mu1 and mu2 for every agent, each of shape [envs, agents]."""
        pos = torch.stack([agent.state.pos for agent in self.world.agents], dim=1)
        goal = torch.stack([agent.goal.state.pos for agent in self.world.agents], dim=1)
        distances = torch.cdist(pos, pos) + 1e6 * torch.eye(pos.shape[1], device=pos.device)  # exclude self
        return self.eps - torch.linalg.vector_norm(pos - goal, dim=-1), distances.min(dim=-1).values - self.d

    def reward(self, agent):
        native = super().reward(agent)  # always computed, so info can report it in both conditions
        if self.n_zones:                # rare catastrophe inside a zone: the only zone signal in the native reward
            inside = self.zone_dist(agent) < 0
            agent.catastrophe = (inside & (torch.rand_like(native) < self.cat_prob)).float()
            native = native - self.cat_penalty * agent.catastrophe
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
        out = {"native_rew": native["pos_rew"] + native["final_rew"] + native["agent_collisions"],
               "collisions": native["agent_collisions"] / self.agent_collision_penalty,  # other agents touched
               "mu1": mu1[:, i], "mu2": mu2[:, i], "pos": agent.state.pos, "goal": agent.goal.state.pos}
        if self.n_zones:
            out["native_rew"] = out["native_rew"] - self.cat_penalty * agent.catastrophe
            out.update(catastrophe=agent.catastrophe, mu3=self.zone_dist(agent), zones=self.zone_pos.clone())
        return out
